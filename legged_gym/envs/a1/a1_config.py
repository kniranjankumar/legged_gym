# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
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

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'
        # load_run = "Jun03_14-08-37_"
        load_run = "Jul03_22-00-03_"
        max_iterations = 3500
        


class A1FlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.015

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'a1_flat'
        policy_class_name = 'SkillActorCritic'     
        # load_run = "Jun21_17-35-58_"
        load_run = "Aug19_10-58-19_"
class A1FlatCfg( A1RoughCfg):

    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        terrain_proportions = [0.5, 0.5, 0., 0., 0.]
        
    class env(A1RoughCfg.env ):
        # num_envs = 4096
        num_observations = 45
        spacing = 10.
    class viewer:
        ref_env = 0
        pos = [0, -2, 0.5]  # [m]
        lookat = [0., 0.5, 0.]  # [m]
        
    # class control( A1RoughCfg.control ):
    #     action_scale = 0.5

    class rewards:
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.0 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.3
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.1 #changed
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-0., 0.]        
    # class rewards:
    #     class scales:
    #         termination = -0.0
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 0.5
    #         lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         orientation = -0.
    #         torques = -0.00001
    #         dof_vel = -0.
    #         dof_acc = -2.5e-7
    #         base_height = -0. 
    #         feet_air_time =  1.0
    #         collision = -1.
    #         feet_stumble = -0.0 
    #         action_rate = -0.01
    #         stand_still = -0.
    #         # box_moved = 2.0

    #     only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     base_height_target = 1.
    #     max_contact_force = 100. # forces above this value are penalized

    # class noise:
    #     add_noise = False
    #     noise_level = 1.0 # scales other values
    #     class noise_scales:
    #         dof_pos = 0.01
    #         dof_vel = 1.5
    #         lin_vel = 0.1
    #         ang_vel = 0.2
    #         gravity = 0.05
    #         height_measurements = 0.1

    # class sim:
    #     dt =  0.005
    #     substeps = 1
    #     gravity = [0., 0. ,-9.81]  # [m/s^2]
    #     up_axis = 1  # 0 is y, 1 is z

    #     class physx:
    #         num_threads = 10
    #         solver_type = 1  # 0: pgs, 1: tgs
    #         num_position_iterations = 4
    #         num_velocity_iterations = 1
    #         contact_offset = 0.01  # [m]
    #         rest_offset = 0.0   # [m]
    #         bounce_threshold_velocity = 0.5 #0.5 [m/s]
    #         # max_depenetration_velocity = 100.0
    #         max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
    #         default_buffer_size_multiplier = 5
    #         contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

##############################################################################################################        
################################         Crouching        #################################################
##############################################################################################################
class A1CrouchingCfg(A1FlatCfg):
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, -0.0]    # min max [rad/s]
            heading = [-0.0, -0.0]
            
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 46
        env_spacing = 6.2
    
    class rewards:
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.0 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        class scales:
                termination = -0.0
                tracking_lin_vel = 1.0
                tracking_ang_vel = 0.3
                lin_vel_z = -2.0
                ang_vel_xy = -0.05
                orientation = -0.1
                dof_vel = -0.
                dof_acc = -2.5e-7
                base_height = -0.0
                feet_air_time =  0.0
                collision = -1.
                feet_stumble = -0.0 
                action_rate = -0.01
                stand_still = -0.
                torques = -0.0002
                dof_pos_limits = -10.0
            
            # termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -0.
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.1
            # feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0 
            # action_rate = -0.01
            # stand_still = -0.
            # torques = -0.0002
            # dof_pos_limits = -10.0
            
            # termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.0
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.5
            # orientation = -0.
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.1
            # feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0 
            # action_rate = -0.01
            # stand_still = -0.0
            # torques = -0.0002
            # dof_pos_limits = -10.0
            # door_angle = 0.6
            # box_moved = 2.0
class A1CrouchingCfgPPO( A1FlatCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.015
        residual_action_penalty_coef = 0.03#0.009

    # class runner( A1FlatCfgPPO.runner ):
    #     experiment_name = 'crouching'
    #     load_run = 'Jun17_18-18-34_'
        
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512,256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    # class runner( A1FlatCfgPPO.runner ):
    #     experiment_name = 'crouching'
        # load_run = 'Jun17_18-18-34_'
    class runner( LeggedRobotCfgPPO.runner ):
    #     run_name = ''
        # checkpoint = "600"
        algorithm_class_name = 'ResidualPPO'
        load_run = "Aug21_22-24-48_"
        max_iterations = 6000
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "height": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
        actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "height",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    ]
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "height",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                      ]
        experiment_name = 'crouching'
        policy_class_name = 'ResidualActorCritic'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/crouching/Jun17_18-18-34_/model_1500.pt"]
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]


##############################################################################################################        
################################         Turning        #################################################
##############################################################################################################

class A1TurnCfg(A1FlatCfg):
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [1.0, 1.0]    # min max [rad/s]
            heading = [3.14, 3.14]
            
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 45
        env_spacing = 6.2
class A1TurnCfgPPO( LeggedRobotCfgPPO ):

    class rewards:
            class scales:
                termination = -0.0
                tracking_lin_vel = 1.0
                tracking_ang_vel = 0.3
                lin_vel_z = -2.0
                ang_vel_xy = -0.05
                orientation = -0.
                dof_vel = -0.
                dof_acc = -2.5e-7
                base_height = -0.1
                feet_air_time =  1.0
                collision = -1.
                feet_stumble = -0.0 
                action_rate = -0.01
                stand_still = -0.5
                torques = -0.0002
                dof_pos_limits = -10.0
                # door_angle = 0.6
                # box_moved = 2.0

            only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
            tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
            soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
            soft_dof_vel_limit = 1.
            soft_torque_limit = 1.
            base_height_target = 0.25
            max_contact_force = 100. # forces above this value are penalized
            
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.009

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        checkpoint = "100" # -1 = last saved model
        # load_run = "May19_11-18-58_"
        # load_run = "Jun08_10-57-33_"
        load_run = "Jun24_13-41-47_"
        
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    # "target_position": 3,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
        actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    ]
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                      ]
        experiment_name = 'turning'
        policy_class_name = 'ResidualActorCritic'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]

##############################################################################################################        
################################         Door opening        #################################################
##############################################################################################################

class A1DoorOpeningCfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
 
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 48
        env_spacing = 6.2
    class viewer:
        ref_env = 0
        pos = [0, 10, 10]  # [m]
        lookat = [5., 5., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -5.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -1.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

class A1DoorOpeningCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.01

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512,256]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "750" # -1 = last saved model
        load_run = "Jul11_13-04-00_"
        
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "door_state": 2,
                    "door_angle": 1,
                    # "target_position": 3,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
        actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "door_state",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    ]
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "door_state",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                      ]
        experiment_name = 'dooropen'
        policy_class_name = 'ResidualActorCritic'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]

##############################################################################################################        
################################        Door openingv2       #################################################
##############################################################################################################

class A1DoorOpeningv2Cfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 50
        env_spacing = 6.2
        # num_envs = 2048
    class viewer:
        ref_env = 0
        pos = [0, 10, 10]  # [m]
        lookat = [5., 5., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 1.0
            # cross_door = 0.1 # makes the robot not open door
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

class A1DoorOpeningv2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.9 #0.05
        # max_iterations = 3500
        

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "door_open":[512, 256],
                             "residual": [512,256]}
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,4],
                             "door_open":[512, 256, 128,2]}
        skill_compositions = {"straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left"],
                              "door_open": ["straight_walk", "door_open"],
                              "residual": ["residual"]}
        meta_backbone_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        synthetic_obs_scales = {"target_position":3}
        
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "1700" # -1 = last saved model
        # load_run = "Jul18_12-51-06_"
        # load_run = "Jul19_18-49-59_"
        load_run = "Jul21_11-18-51_"
        max_iterations = 6000
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "door_position": 2,
                    "door_angle": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12,
                    "robot_position":2,
                    # synthetic observations after this point
                    "target_position": 2,
                    }
        actor_obs ={
                    "straight_walk":["scaled_base_lin_vel",
                                        "scaled_base_ang_vel",
                                        "projected_gravity",
                                        "relative_dof",
                                        "scaled_dof_vel",
                                        "actions"],
                    
                    "standing":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    "turn_left":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                    
                    "turn_right":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],

                    "target_reach":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "target_position",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                   
                    "door_open":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                   
                    "residual":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position"]
                    }
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "door_position",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions",
                    "robot_position"]
                      ]
        meta_network_obs = ["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position"]
        experiment_name = 'dooropenv2'
        policy_class_name = 'MultiSkillActorCriticv3'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
        #                 "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt", #turn left
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jun09_17-20-37_/model_1150.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/May19_14-20-14_/model_750.pt"]
        skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                        "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                        "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt"  
                }
##############################################################################################################
################################         Target Reach        #################################################
##############################################################################################################        
class A1MultiSkillReachCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.0 # was 0.05
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128], [256, 128], [256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # load_run = "May12_19-48-31_"
        # load_run = "May16_09-46-39_"
        # load_run = "May16_21-49-42_"
        # load_run = "May19_22-33-24_/model_3500.pt"
        # load_run = "Jun09_14-54-30_"
        # load_run = "Jun27_15-27-05_"
        # load_run = "Jun30_10-56-57_"
        # load_run = "Jul06_16-49-08_"
        load_run = "Jul07_10-24-55_"
        # resume = True
        max_iterations = 5500
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
        actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    # ["scaled_base_lin_vel",
                    # "scaled_base_ang_vel",
                    # "projected_gravity",
                    # "target_position",
                    # "relative_dof",
                    # "scaled_dof_vel",
                    # "actions"]
                    ]
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                      ]
        experiment_name = 'multiskill_targetreach'
        policy_class_name = 'MultiSkillActorCritic'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt" #turn left
        #                ]
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                       ]
class A1TargetReachCfg( A1FlatCfg):
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 47
        env_spacing = 5.
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.00
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            target_reach = 4.0
            # door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized


##############################################################################################################        
################################        Object pushing       #################################################
##############################################################################################################
class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.02

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512, 256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512,256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        max_iterations = 1500
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "object_position": 2,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
        actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "object_position", 
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    ]
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "object_position",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                      ]
        experiment_name = 'multiskill_object_push'
        policy_class_name = 'ResidualActorCritic'
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
        
class A1TargetObjectPushCfg( A1FlatCfg):
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 49
        env_spacing = 5.
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.4 #0.5
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.00
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            object_target_dist = 2.0
            # door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

##############################################################################################################
################################           Standing          #################################################
##############################################################################################################    

class A1StandingCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'standing'
        policy_class_name = 'SkillActorCritic'     
        
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

class A1StandingCfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 45
        env_spacing = 5.
    class viewer:
        ref_env = 0
        pos = [0, 10, 10]  # [m]
        lookat = [5., 5., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 1
        max_push_vel_xy = 1.
    
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.1
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.5
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            # door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.3 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

##############################################################################################################
##########################         Interactive Target Reach        ###########################################
##############################################################################################################        

class InteractiveTargetReachCfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 55
        env_spacing = 12.2
        episode_length_s = 40
        # num_envs = 2048
    class viewer:
        ref_env = 0
        pos = [20, 1, 10]  # [m]
        lookat = [20., 0., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 2.0 #4.0 maybe its 2->1?
            robot_target_dist = 5.0
            # cross_door = 0.1 # makes the robot not open door
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

class InteractiveTargetReachCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0 #1.5 #0.05
        gamma = 0.99
        
        # max_iterations = 3500
        

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "door_open":[512, 256],
                             "door_openv2": [512,256],
                            #  "residual": [512,256]
                             }
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,4],
                             "door_open":[512, 256, 128,2],
                             "door_openv2":[256,128,9]
                             }
        skill_compositions = {
                            #   "straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left"],
                              "door_open": ["straight_walk", "door_open"],
                              "door_openv2":["straight_walk", "standing","turn_right","turn_left","target_reach","door_open","door_openv2"],
                            #   "residual": ["residual"]
                              }
        meta_backbone_dims = [512, 256, 256]
        # meta_backbone_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        synthetic_obs_scales = {"synth_target_position":2.5}
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        save_interval = 500
        # checkpoint = "1700" # -1 = last saved model
        # load_run = "Jul18_12-51-06_"
        # load_run = "Jul26_15-15-48_"
        # load_run = "Jul26_07-10-21_" # no residual penalty
        # load_run = "Jul27_10-59-17_" # no residual network
        # load_run = "Aug05_10-29-06_"
        # load_run = "Aug09_20-25-14_"
        # load_run = "Aug10_11-27-03_" # 90% with table, continued from Aug09_20-25-14_
        # load_run = "Aug11_01-18-09_" #90% with table from scratch
        load_run = "Aug12_01-29-28_"

        # resume_path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_5750.pt"
        max_iterations = 15000
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "robot_angle": 1,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "door_position": 2,
                    # "table_position":2,
                    "door_angle": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12,
                    "robot_position":2,
                    "target_room":1,
                    "robot_room":1,
                    # synthetic observations after this point
                    "synth_target_position": 2,
                    }
        actor_obs ={
                    "straight_walk":["scaled_base_lin_vel",
                                        "scaled_base_ang_vel",
                                        "projected_gravity",
                                        "relative_dof",
                                        "scaled_dof_vel",
                                        "actions"],
                    
                    "standing":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    "turn_left":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                    
                    "turn_right":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],

                    "target_reach":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "synth_target_position",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                   
                    "door_open":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                   
                    "door_openv2":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position"],
                    
                    # "residual":["scaled_base_lin_vel",
                    #             "scaled_base_ang_vel",
                    #             "projected_gravity",
                    #             "synth_target_position",
                    #             "target_position",
                    #             "door_position",
                    #             "door_angle",
                    #             "relative_dof",
                    #             "scaled_dof_vel",
                    #             "actions",
                    #             "robot_position",
                    #             "target_room",
                    #             "robot_room"]
                    }
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "robot_angle",
                    "projected_gravity",
                    "synth_target_position",
                    "target_position",
                    "door_position",
                    # "table_position",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions",
                    "robot_position",
                    "target_room",
                    "robot_room"]
                      ]
        meta_network_obs = ["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "robot_angle",
                                "projected_gravity",
                                "target_position",
                                "door_position",
                                # "table_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position",
                                "target_room",
                                "robot_room"
                                ]
        experiment_name = 'interactive_targetreach'
        policy_class_name = 'MultiSkillActorCriticv3'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
        #                 "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt", #turn left
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jun09_17-20-37_/model_1150.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/May19_14-20-14_/model_750.pt"]
        skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                        "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                        "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt",
                        "door_openv2":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_6000.pt"  
                }


##############################################################################################################
##########################        Interactive Target Reachv2       ###########################################
##############################################################################################################        
        
class InteractiveTargetReachv2Cfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 62
        env_spacing = 12.2
        # num_envs = 2048
    class viewer:
        ref_env = 0
        pos = [0, 10, 10]  # [m]
        lookat = [5., 5., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 0.8
            robot_target_dist = 4.0
            # cross_door = 0.1 # makes the robot not open door
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

class InteractiveTargetReachv2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 1.5 #0.05
        # max_iterations = 15000
        

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "door_open":[512, 256],
                             "residual": [512,256]
                             }
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,4],
                             "door_open":[512, 256, 128,2]}
        skill_compositions = {"straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left"],
                              "door_open": ["straight_walk", "door_open"],
                              "residual": ["residual"]
                              }
        meta_backbone_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        synthetic_obs_scales = {"synth_target_position":3}
        synthetic_obs_ingredients = ["door_position",
                                    #  "table_position",
                                     "wall_corner_position_00",
                                     "wall_corner_position_10",
                                     "wall_corner_position_11",
                                     "wall_corner_position_12",
                                     "wall_corner_position_02",
                                     "wall_corner_position_01",
                                     ]
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "1700" # -1 = last saved model
        # load_run = "Jul18_12-51-06_"
        load_run = "Jul26_15-15-48_"
        # load_run = "Jul26_07-10-21_" # no residual penalty
        # load_run = "Jul27_10-59-17_" # no residual network
        # resume = True
        # resume_path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_5750.pt"
        max_iterations = 15000
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "door_position": 2,
                    "door_angle": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "wall_corner_position_00": 2,
                    "wall_corner_position_10": 2,
                    "wall_corner_position_11": 2,
                    "wall_corner_position_12": 2,
                    "wall_corner_position_02": 2,
                    "wall_corner_position_01": 2,
                    "actions": 12,
                    # "robot_position": 2,
                    # synthetic observations after this point
                    "synth_target_position": 2,
                    }
        actor_obs ={
                    "straight_walk":["scaled_base_lin_vel",
                                        "scaled_base_ang_vel",
                                        "projected_gravity",
                                        "relative_dof",
                                        "scaled_dof_vel",
                                        "actions"],
                    
                    "standing":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    "turn_left":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                    
                    "turn_right":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],

                    "target_reach":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "synth_target_position",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                   
                    "door_open":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                   
                    "residual":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "synth_target_position",
                                "target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                # "robot_position"
                                "wall_corner_position_00",
                                "wall_corner_position_10",
                                "wall_corner_position_11",
                                "wall_corner_position_12",
                                "wall_corner_position_02",
                                "wall_corner_position_01",                                
                                ]
                    }
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "synth_target_position",
                    "target_position",
                    "door_position",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "wall_corner_position_00",
                    "wall_corner_position_10",
                    "wall_corner_position_11",
                    "wall_corner_position_12",
                    "wall_corner_position_02",
                    "wall_corner_position_01",
                    "actions",
                    # "robot_position"
                    ]
                      ]
        meta_network_obs = ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "door_position",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "wall_corner_position_00",
                    "wall_corner_position_10",
                    "wall_corner_position_11",
                    "wall_corner_position_12",
                    "wall_corner_position_02",
                    "wall_corner_position_01",
                    "actions",
                    # "robot_position"
                    ]
        experiment_name = 'interactive_targetreachv2'
        policy_class_name = 'MultiSkillActorCriticSplit'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
        #                 "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt", #turn left
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jun09_17-20-37_/model_1150.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/May19_14-20-14_/model_750.pt"]
        skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                        "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                        "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt"  
                }
        
        
##############################################################################################################
##########################         Interactive Target Reach v3        ###########################################
##############################################################################################################        

class InteractiveTargetReachv3Cfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 60
        env_spacing = 12.2
        episode_length_s = 40
        # num_envs = 2048
    # class viewer:
    #     ref_env = 0
    #     pos = [20, 1, 10]  # [m]
    #     lookat = [20., 0., 0.]  # [m]
    class viewer:
        ref_env = 0
        pos = [0, -2, 0.5]  # [m]
        lookat = [0., 0.5, 0.]  # [m]
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0 #1.0
            tracking_ang_vel = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time = 1.0
            collision = -2 #-1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 2.0 #0.8 #4.0 maybe its 2->1?
            robot_target_dist = 5.0
            # target_reach = 800
            # cross_door = 0.1 # makes the robot not open door
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 100.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            # always_use_articulations = True

class InteractiveTargetReachv3CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0 #1.5 #0.05
        residual_weight_penalty_coef = 0
        gamma = 0.99
        
        # max_iterations = 3500
        

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "door_open":[512, 256],
                             "door_openv2": [512,256],
                             "crouching": [256,128],
                            #  "residual": [512,256]
                             }
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,4],
                             "door_open":[512, 256, 128,2],
                             "door_openv2":[256,128,9],
                             "crouching": [512,256,2]
                             }
        skill_compositions = {
                            #   "straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left"],
                              "door_open": ["straight_walk", "door_open"],
                              "door_openv2":["straight_walk", "standing","turn_right","turn_left","target_reach","door_open","door_openv2"],
                                "crouching": ["straight_walk","crouching"],
                            #   "residual": ["residual"]
                              }
        meta_backbone_dims = [512, 256, 256]
        # meta_backbone_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        synthetic_obs_scales = {"synth_target_position":2.5}
        synthetic_obs_ingredients = ["door_position",
                                     "table_position",
                                     "couch_position",
                                     "target_position",
                                     "robot_angle",
                                     "robot_position",
                                     "target_room",
                                     "robot_room"
                                    #  "wall_corner_position_00",
                                    #  "wall_corner_position_10",
                                    #  "wall_corner_position_11",
                                    #  "wall_corner_position_12",
                                    #  "wall_corner_position_02",
                                    #  "wall_corner_position_01",
                                     ]
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        save_interval = 500
        # checkpoint = "1700" # -1 = last saved model
        # load_run = "Jul18_12-51-06_"
        # load_run = "Jul26_15-15-48_"
        # load_run = "Jul26_07-10-21_" # no residual penalty
        # load_run = "Jul27_10-59-17_" # no residual network
        # load_run = "Aug05_10-29-06_"
        # load_run = "Aug09_20-25-14_"
        # load_run = "Aug10_11-27-03_" # 90% with table, continued from Aug09_20-25-14_
        # load_run = "Aug11_01-18-09_" #90% with table from scratch
        # load_run = "Aug15_13-24-32_" 
        # load_run = "Aug16_22-33-13_" # finetuned from Aug15_13-24-32_
        
        # load_run = "Aug17_19-54-25_"
        # load_run = "Aug18_12-53-21_"
        # load_run = "Aug20_01-05-14_"
        # load_run = "Aug23_17-42-04_"
        # load_run = "Aug27_10-34-57_"
        # load_run = "Aug28_20-11-06_"
        # load_run = "Sep05_15-03-25_"
        load_run = "Sep06_21-41-40_"
        # resume = True
        checkpoint = 5000
        # resume_path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_5750.pt"
        max_iterations = 15000
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "robot_angle": 1,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "door_position": 2,
                    "table_position":2,
                    "couch_position":2,
                    "door_angle": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12,
                    "robot_position":2,
                    "target_room":1,
                    "robot_room":1,
                    "height": 1,
                    # synthetic observations after this point
                    "synth_target_position": 2,
                    }
        actor_obs ={
                    "straight_walk":["scaled_base_lin_vel",
                                        "scaled_base_ang_vel",
                                        "projected_gravity",
                                        "relative_dof",
                                        "scaled_dof_vel",
                                        "actions"],
                    
                    "standing":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    "turn_left":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                    
                    "turn_right":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],

                    "target_reach":["scaled_base_lin_vel",
                                    "scaled_base_ang_vel",
                                    "projected_gravity",
                                    "synth_target_position",
                                    "relative_dof",
                                    "scaled_dof_vel",
                                    "actions"],
                   
                    "door_open":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                   
                    "door_openv2":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position"],
                    
                    "crouching":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "height",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    # "residual":["scaled_base_lin_vel",
                    #             "scaled_base_ang_vel",
                    #             "robot_angle",
                    #             "projected_gravity",
                    #             "synth_target_position",
                    #             "target_position",
                    #             "door_position",
                    #             "door_angle",
                    #             "relative_dof",
                    #             "scaled_dof_vel",
                    #             "actions",
                    #             "robot_position",
                    #             "target_room",
                    #             "robot_room"]
                    }
        critic_obs = [                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "robot_angle",
                    "projected_gravity",
                    "synth_target_position",
                    "target_position",
                    "door_position",
                    "table_position",
                    "couch_position",
                    "door_angle",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions",
                    "robot_position",
                    "target_room",
                    "robot_room",
                    "height"]
                      ]
        meta_network_obs = ["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "robot_angle",
                                "projected_gravity",
                                "target_position",
                                "door_position",
                                "table_position",
                                "couch_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position",
                                "target_room",
                                "robot_room",
                                # "height",
                                "synth_target_position"
                                ]
        experiment_name = 'interactive_targetreachv3'
        policy_class_name = 'MultiSkillActorCriticSplit'
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
        #                 "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt", #turn left
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jun09_17-20-37_/model_1150.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/May19_14-20-14_/model_750.pt"]
        skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                        "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                        "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt",
                        "door_openv2":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_6000.pt",
                        "crouching":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/crouching/Aug21_22-24-48_/model_6000.pt"
                }