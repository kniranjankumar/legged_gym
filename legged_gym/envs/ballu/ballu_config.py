from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BalluCfg( LeggedRobotCfg ):
    class env:
        num_envs = 1024
        num_observations = 7
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 2 + 5 # underactuated 5 joints
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        terrain_proportions = [0.5, 0.5, 0., 0., 0.]
           
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 2.72] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.1,   # [rad]
        #     'RL_hip_joint': 0.1,   # [rad]
        #     'FR_hip_joint': -0.1 ,  # [rad]
        #     'RR_hip_joint': -0.1,   # [rad]

        #     'FL_thigh_joint': 0.8,     # [rad]
        #     'RL_thigh_joint': 1.,   # [rad]
        #     'FR_thigh_joint': 0.8,     # [rad]
        #     'RR_thigh_joint': 1.,   # [rad]

        #     'FL_calf_joint': -1.5,   # [rad]
        #     'RL_calf_joint': -1.5,    # [rad]
        #     'FR_calf_joint': -1.5,  # [rad]
        #     'RR_calf_joint': -1.5,    # [rad]
        # }
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'HIP_LEFT': 0.,
            'KNEE_LEFT': 0.,
            'MOTOR_LEFT': 0.,   # [rad]
            'HIP_RIGHT': 0.,
            'KNEE_RIGHT': 0.,
            'MOTOR_RIGHT': 0.,
            'NECK': 0.
        }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        #for sim
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5} 
        # for real robot
        stiffness = {'MOTOR': 0., "HIP": 0, "KNEE": 0, "NECK": 0}  # [N*m/rad]
        damping = {'MOTOR': 0.5, "HIP": 0.5, "KNEE": 0.5, "NECK": 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/BALLU_urdf/urdf/urdf/ballu.urdf'
        name = "ballu"
        foot_name = "TIBIA"
        penalize_contacts_on = ["TIBIA", "FEMUR"]
        terminate_after_contacts_on = []
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        fix_base_link = True
        disable_gravity = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            
            
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

class BalluCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'ballu'
        load_run = "Apr28_13-55-30_"
        max_iterations = 3500
        

