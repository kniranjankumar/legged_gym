import pybullet as p, numpy as np
import time, pdb
import pybullet_data

ENABLE_BALLOON_FORCE = True
BUOYANCY_MASS = 0.202 * 6
# TIE_FEMUR_LOCAL = (0, 0.340, 0)
# TIE_MOTORARM_LOCAL = (0.016,0,0)
KNEE_SPRING_COEFF = 0.8 * 0.1409e-3 * 180 / np.pi * 5
KNEE_DAMPING_COEFF = 1.0e-2 # 0.1390e-3*180/np.pi # 0.1390e-3 N/deg from Misumi Korea
KNEE_PRELOAD = np.deg2rad(180 - 135 + 27.35)

KNEE_GAIN_P = 3.0 # * 2.4 * 5
KNEE_GAIN_D = 0.5 # * 2.4 * 1

# CLS0511H @ 3.5V
MOTOR_VELOCITY_MAX = np.rad2deg(60) / 0.07 # 0.07sec/60deg
MOTOR_TORQUE_MAX = (1.6) * 0.9 * 9.81 * 1e-2 * 1.5 # 0.9 (0.6) kg.cm # Tm = 90.4500e-1 -- N.m -- motor torque

_p = p # for debugging with pdb
ballu_id = None
joint_ids = {}
def main():
    global ballu_id, joint_ids

    # Make connection and Set default environment
    c = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane and ballu model
    pos0 = [0,0, 0.7005] # 0.69565 0.6666 0.6615
    ori0 = p.getQuaternionFromEuler([0,0,0])
    cam_pos0 = (pos0[0]+1, pos0[1], pos0[2]) # cam0 = p.getDebugVisualizerCamera()
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=180.0, cameraPitch=0.0, cameraTargetPosition=cam_pos0)
    plane_id = p.loadURDF("plane.urdf")
    ballu_id = p.loadURDF("./urdf/ballu.urdf", pos0, ori0, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL) # p.URDF_USE_INERTIA_FROM_FILE|


    # Collect Joint info
    joint_ids = {p.getJointInfo(ballu_id, jdx)[1].decode("utf8"):jdx for jdx in range(p.getNumJoints(ballu_id))}
    jt_pos0 = {'HIP_LEFT':np.deg2rad(1), 'HIP_RIGHT':np.deg2rad(1), 'KNEE_LEFT':np.deg2rad(27.35), 'KNEE_RIGHT':np.deg2rad(27.35), 'NECK':np.deg2rad(0), 'MOTOR_LEFT':np.deg2rad(10), 'MOTOR_RIGHT':np.deg2rad(10)} # (-10, 20, 0)
    # Apply initial pose
    [p.resetJointState(ballu_id, joint_ids[jn], jv, 0.0) for jn, jv in jt_pos0.items()]

    # # Collect Link info for URDF inertia debug
    # link_info = [p.getDynamicsInfo(ballu_id, jdx) for jdx in range(p.getNumJoints(ballu_id)) if jdx in [joint_ids['HIP_LEFT'], joint_ids['HIP_RIGHT'], joint_ids['KNEE_LEFT'], joint_ids['KNEE_RIGHT']]]
    # link_info_filt = [(li[0], li[2]) for li in link_info] # only mass and diag_inertia


    # CONSTRAINT DOES NOT WORK WELL!
    # hip_status = p.getLinkState(ballu_id, joint_ids['HIP_LEFT'])
    # tie_femur_left, _ = p.multiplyTransforms(hip_status[4], hip_status[5], TIE_FEMUR_LOCAL, (0,0,0,1))
    # motorarm_status = p.getLinkState(ballu_id, joint_ids['MOTOR_LEFT'])
    # tie_motorarm_left, _ = p.multiplyTransforms(motorarm_status[4], motorarm_status[5], TIE_MOTORARM_LOCAL,  (0,0,0,1))
    # # inv_trans, inv_rot = _p.invertTransform(hip_status[4], hip_status[5])
    # # _p.multiplyTransforms(inv_trans, inv_rot, _tie_pos, _tie_ori) == ()(0, 0.350, 0), (0,0,0,1))
    #
    # tendon_length0 = np.linalg.norm(np.array(tie_femur_left)-np.array(tie_motorarm_left))
    #
    # lk_const = p.createConstraint(ballu_id, joint_ids['HIP_LEFT'], ballu_id, joint_ids['MOTOR_LEFT'], p.JOINT_POINT2POINT, [0, 0, 0], TIE_FEMUR_LOCAL, TIE_MOTORARM_LOCAL)
    # p.changeConstraint(lk_const, maxForce=2)
    # const_line = p.addUserDebugLine(tie_femur_left, tie_motorarm_left, [1.0, 0.5, 0.5],  lineWidth=10.0, lifeTime=0.0) # replaceItemUniqueId

    # Run simulation
    for idx in range(int(1e4)):
        p.stepSimulation()
        time.sleep(1./240.)

        # Let debug camera follow the robot
        pos, ori = p.getBasePositionAndOrientation(ballu_id)
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=180.0, cameraPitch=0.0, cameraTargetPosition=(pos[0]+1, pos[1], pos[2]))

        # # Let debug cam follow the left ankle
        # link_state = p.getLinkState(ballu_id, joint_ids['MOTOR_LEFT'])
        # pos = link_state[0]
        # rpy = p.getEulerFromQuaternion(link_state[1])
        # p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=np.rad2deg(rpy[0]+0.5*np.pi), cameraPitch=0.0, cameraTargetPosition=(pos[0], pos[1], pos[2]+0.2))

        # Low-level Knee control
        # (1) Torque due to the knee springs
        spring_torque_left = calc_knee_spring_torque('LEFT', jt_pos0['KNEE_LEFT' ])
        spring_torque_right = calc_knee_spring_torque('RIGHT', jt_pos0['KNEE_RIGHT' ])

        # (2) Force Due to servo actuation
        # test position control

        if (idx//240) % 2:
            target_position = [np.deg2rad(170), np.deg2rad(10)]
        else:
            target_position = [np.deg2rad(10), np.deg2rad(170)]

        if (idx%240) == 0.0:
            print(target_position)

        motor_left, motor_right = p.getJointStates(ballu_id, [joint_ids['MOTOR_LEFT'], joint_ids['MOTOR_RIGHT']])

        motor_torque_left,  motor_velocity_left  = calc_knee_actuation('LEFT', target_position[0])
        motor_torque_right, motor_velocity_right = calc_knee_actuation('RIGHT', target_position[1])

        print(f"vel:({motor_velocity_left:3.2f}, {motor_velocity_right:3.2f}), spring_torque: ({spring_torque_left:3.2f}, {spring_torque_right:3.2f}), motor_torque: ({motor_torque_left:3.2f}, {motor_torque_right:3.2f})")
        # (3) Apply the calculated forces
        p.setJointMotorControl2(ballu_id, joint_ids['MOTOR_LEFT'],  p.POSITION_CONTROL, targetPosition=target_position[0], maxVelocity=MOTOR_VELOCITY_MAX)
        p.setJointMotorControl2(ballu_id, joint_ids['MOTOR_RIGHT'], p.POSITION_CONTROL, targetPosition=target_position[1], maxVelocity=MOTOR_VELOCITY_MAX)
        p.setJointMotorControlArray(ballu_id, [joint_ids['KNEE_LEFT'], joint_ids['KNEE_RIGHT']], p.VELOCITY_CONTROL,
                                    targetVelocities=[motor_velocity_left, motor_velocity_right],
                                    forces=[motor_torque_left+spring_torque_left, motor_torque_right+spring_torque_right])


        # CONSTRAINT DOES NOT WORK WELL!
        # # draw constraint line
        # hip_status = p.getLinkState(ballu_id, joint_ids['HIP_LEFT'])
        # tie_femur_left, _ = p.multiplyTransforms(hip_status[4], hip_status[5], TIE_FEMUR_LOCAL, (0,0,0,1))
        # motorarm_status = p.getLinkState(ballu_id, joint_ids['MOTOR_LEFT'])
        # tie_motorarm_left, _ = p.multiplyTransforms(motorarm_status[4], motorarm_status[5], TIE_MOTORARM_LOCAL, (0,0,0,1))
        # tendon_length = np.linalg.norm(np.array(tie_femur_left)-np.array(tie_motorarm_left))
        # if tendon_length < tendon_length0 * 0.90:
        #     p.changeConstraint(lk_const, maxForce=0)
        #     const_line = p.addUserDebugLine(tie_femur_left, tie_motorarm_left, [0.5, 0.1, 0.1], lifeTime=0.0, replaceItemUniqueId=const_line)
        # else:
        #     p.changeConstraint(lk_const, maxForce=2)
        #     const_line = p.addUserDebugLine(tie_femur_left, tie_motorarm_left, [1.0, 0.5, 0.5], lifeTime=0.0, replaceItemUniqueId=const_line)
        # print(tendon_length < tendon_length0 * 0.90, tendon_length, tendon_length0)
        #
        # lm_q = p.getJointState(ballu_id, joint_ids['MOTOR_LEFT'])[0]
        # lm_l, lm_h = p.getJointInfo(ballu_id, joint_ids['MOTOR_LEFT'])[8:10]
        # print(lm_l <= lm_q <= lm_h, lm_l, lm_q, lm_h)


        if ENABLE_BALLOON_FORCE:
            # calc forces from the balloons
            balloon_state = p.getLinkState(ballu_id, joint_ids['NECK'], 1)
            balloon_v = np.array(balloon_state[6])
            balloon_drag = -0.3 * balloon_v
            balloon_buoyancy = np.array([0, 0, BUOYANCY_MASS * 9.81])
            balloon_force = balloon_drag + balloon_buoyancy
            p.applyExternalForce(ballu_id, joint_ids['NECK'], balloon_force, balloon_state[0], flags=p.WORLD_FRAME)

    p.disconnect()

def calc_knee_spring_torque(lr:str, knee_pos0:float):
    # jn: either ('LEFT', 'RIGHT')
    # knee_pos0: the initial position of the knee
    knee_state = p.getJointState(ballu_id, joint_ids[f'KNEE_{lr}'])
    knee_pos = knee_state[0]
    knee_vel = knee_state[1]

    # Knee Spring is modeled with Stiffness and Damping
    spring_force  = -2 * KNEE_SPRING_COEFF * (knee_pos - knee_pos0 + KNEE_PRELOAD) - KNEE_DAMPING_COEFF * knee_vel
    return spring_force

knee_pos_err0 = {'KNEE_LEFT':0.0, 'KNEE_RIGHT':0.0}
def calc_knee_actuation(lr, target_position):
    global knee_pos_err0

    motor_limit = p.getJointInfo(ballu_id, joint_ids[f'MOTOR_{lr}'])[8:10]
    knee_limit = p.getJointInfo(ballu_id, joint_ids[f'KNEE_{lr}'])[8:10]

    motor_state = p.getJointState(ballu_id, joint_ids[f'MOTOR_{lr}'])
    knee_state = p.getJointState(ballu_id, joint_ids[f'KNEE_{lr}'])

    motor_pos, motor_vel = motor_state[0], motor_state[1]
    knee_pos, knee_vel = knee_state[0], knee_state[1]
    # print(f"[MOTOR_{lr}]: pos_err:{np.rad2deg(target_position-motor_pos):3.2f}")
    if motor_limit[0] + 1e-3 < motor_pos < motor_limit[1] - 1e-3:
        # At this time, using linear min-max transfrom from motor to knee
        # i.e., Assuming that the knee and the servo have proportional relationship
        knee_pos_target = min_max_mapping(target_position, motor_limit, knee_limit)
        print(f"[KNEE_{lr}] target:{np.rad2deg(knee_pos_target):3.2f}, pos:{np.rad2deg(knee_pos):3.2f}, vel:{np.rad2deg(knee_vel):3.2f}")
        knee_pos_err = knee_pos_target - knee_pos
        knee_vel_err = (knee_pos_err - knee_pos_err0[f'KNEE_{lr}'])/(1/240.)
        actuation = knee_pos_err * KNEE_GAIN_P + knee_vel_err * KNEE_GAIN_D
        actuation = np.clip(actuation, 0.0, MOTOR_TORQUE_MAX)
    else:
        print(f"MOTOR_{lr}: out of range")
        knee_pos_err = knee_vel_err = 0.0
        actuation = 0.0
    knee_pos_err0[f'KNEE_{lr}'] = knee_pos_err

    return actuation, MOTOR_VELOCITY_MAX

def min_max_mapping(val, from_range, to_range):
    # val: value to map
    # from: tuple of (min, max)
    # to: tuple of (min, max)
    return (val - from_range[0]) / (from_range[1] - from_range[0]) * (to_range[1] - to_range[0]) + to_range[0]

if __name__ == '__main__':
    main()
