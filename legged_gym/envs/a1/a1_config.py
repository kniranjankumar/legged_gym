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
import numpy as np
class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.7,     # [rad]
            'RL_thigh_joint': 0.7,   # [rad]
            'FR_thigh_joint': 0.7,     # [rad]
            'RR_thigh_joint': 0.7,   # [rad]

            'FL_calf_joint': -1.4,   # [rad]
            'RL_calf_joint': -1.4,    # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            'RR_calf_joint': -1.4,    # [rad]
        }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        #for sim
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5} 
        # for real robot
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 1.0}     # [N*m*s/rad]
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
        load_run = "Jun21_17-35-58_"
        # load_run = "Sep25_17-07-25_"
        # load_run = "Aug19_10-58-19_"
        # load_run = "Sep26_16-45-50_"
        load_run = "Sep26_17-16-44_"
        # load_run = "May08_07-29-42_"
        resume = False
        # checkpoint = 2000
class A1FlatCfg( A1RoughCfg):

    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        # terminate_after_contacts_on = ["face","base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        # penalize_contacts_on = ["face"]
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
        pos = [3, -1, 0.5]  # [m]
        lookat = [3., 0.5, 0.]  # [m]
        
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
            # tracking_lin_vel = 1.0
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
    
    
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        
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
        num_observations = 45
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
        # residual_action_penalty_coef = 0.03#0.009

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
        # load_run = "Aug21_22-24-48_" #low pid
        load_run = "Oct05_13-26-36_" # high_pid
        # resume = True
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

    # class runner(LeggedRobotCfgPPO.runner ):
    #     run_name = ''
    #     # experiment_name = 'a1_flat'
    #     policy_class_name = 'SkillActorCritic'     
    #     load_run = "Jun21_17-35-58_"
    #     load_run = "Sep26_17-16-44_"
    #     experiment_name = 'crouching'
    #     resume = True
        

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
        
    class viewer:
        ref_env = 0
        pos = [-2, -3.5, 2.5]  # [m]
        lookat = [0., 0.5, 0.]  # [m]
        
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
                stand_still = -0.0
                torques = -0.0002
                dof_pos_limits = -10.0
                # door_angle = 0.6
                # box_moved = 2.0

            only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
            tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
            soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
            soft_dof_vel_limit = 1.
            soft_torque_limit = 1.
            base_height_target = 0.29
            max_contact_force = 100. # forces above this value are penalized
            
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.009
        residual_weight_penalty_coef = 0.01

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "100" # -1 = last saved model
        # load_run = "May19_11-18-58_"
        # load_run = "Jun08_10-57-33_"
        # load_run = "Sep21_19-19-47_"
        # load_run = "Sep21_19-36-10_"
        load_run = "Jun24_13-48-02_"
        # load_run = "Jun24_13-41-47_"
        load_run = "Sep26_15-52-16_"
        load_run = "Sep26_16-17-42_" # better policy left
        # load_run = "Sep28_19-15-56_" # turn right old
        load_run = "Mar07_20-09-15_" # new turn right
        load_run = "Mar07_20-16-26_" # new turn left
        
        checkpoint=150
        # resume = True
        
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
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
        #high gain policy
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]
        
    # class runner(LeggedRobotCfgPPO.runner ): #curriculum
    #     run_name = ''
    #     # experiment_name = 'a1_flat'
    #     policy_class_name = 'SkillActorCritic'     
    #     load_run = "Jun21_17-35-58_"
    #     load_run = "Sep26_17-16-44_"
    #     experiment_name = 'turning'
    #     resume = True
        

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
        env_spacing = 12.2
    class viewer:
        ref_env = 0
        pos = [1, -1, 0.3]  # [m]
        lookat = [1., 0., 0.]  # [m]

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
        #straight walking
        # class scales:
        #     termination = -0.0
        #     tracking_lin_vel = 1.0
        #     tracking_ang_vel = 0.3
        #     lin_vel_z = -2.0
        #     ang_vel_xy = -0.05
        #     orientation = -1.0 #changed
        #     dof_vel = -0.
        #     dof_acc = -2.5e-7
        #     base_height = -1.0
        #     feet_air_time =  1.0
        #     collision = -1.
        #     feet_stumble = -0.0 
        #     action_rate = -0.01
        #     stand_still = -0.
        #     torques = -0.0002
        #     dof_pos_limits = -10.0
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time =  1.0
            collision = -5.0
            feet_stumble = -0.0 
            action_rate = -0.00 -0.5
            stand_still = -1.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 0.2
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
        entropy_coef = 0.02
        residual_action_penalty_coef = 0.01
        residual_weight_penalty_coef = 0.01

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512,256]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        load_run = "Oct27_17-20-31_" # warmstarting from walking
        load_run = "Oct28_14-42-32_"
        # load_run = "Oct27_20-40-02_"
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "750" # -1 = last saved model
        # load_run = "May19_14-20-14_"
        # load_run = "Jul11_13-04-00_"Oct14_10-20-05_
        # load_run = "Oct05_15-12-02_"
        # load_run = "Oct13_20-15-48_"
        # load_run = "Oct14_10-47-37_"
        load_run = "Oct14_14-40-56_" # best real robot
        load_run = "Mar09_17-40-47_"
        # load_run = "Mar10_15-04-01_"
        # load_run = "Mar10_09-41-36_"
        resume = True
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
    #     # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
    #     skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
    #     # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Oct14_10-20-05_/model_1500.pt"] #high pid walking
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]

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
        num_observations = 49
        env_spacing = 6.2
        # num_envs = 2048
    # class viewer:
    #     ref_env = 0
    #     pos = [0, 10, 10]  # [m]
    #     lookat = [5., 5., 0.]  # [m]
    class viewer:
        ref_env = 0
        pos = [0, 5.1, 9]  # [m]
        lookat = [0., 5., 0.]  # [m]
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
            collision = -0.5
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            door_angle = 0.5
            robot_door_dist = 1.0
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

# class A1DoorOpeningv2CfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.02
#         residual_action_penalty_coef = 0.9
#         residual_weight_penalty_coef = 0.01

#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [[512, 256, 128], [512,256]]
#         critic_hidden_dims = [512, 256, 128]
#         weight_network_dims = [512, 256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
#     class runner( LeggedRobotCfgPPO.runner ):
#         load_run = "Oct27_17-20-31_" # warmstarting from walking
#         load_run = "Oct28_14-42-32_"
#         # load_run = "Oct27_20-40-02_"
#         run_name = ''
#         algorithm_class_name = 'ResidualPPO'
#         # checkpoint = "750" # -1 = last saved model
#         # load_run = "May19_14-20-14_"
#         # load_run = "Jul11_13-04-00_"Oct14_10-20-05_
#         # load_run = "Oct05_15-12-02_"
#         # load_run = "Oct13_20-15-48_"
#         # load_run = "Oct14_10-47-37_"
#         load_run = "Oct14_14-40-56_" # best real robot
#         load_run = "Mar09_17-40-47_"
#         # load_run = "Mar10_15-04-01_"
#         # load_run = "Mar10_09-41-36_"
#         # resume = True
#         obs_sizes =  {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "projected_gravity": 3,
#                     "door_position": 2,
#                     "door_angle": 1,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12,
#                     "robot_angle":1
#                     # "robot_position":2,
#                     # synthetic observations after this point
#                     # "target_position": 2,
#                     }
#         actor_obs =[["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "door_position",
#                     "door_angle",
#                     "robot_angle",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                     ]
#         critic_obs = [                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "door_position",
#                     "door_angle",
#                     "robot_angle",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                       ]
#         experiment_name = 'dooropenv2'
#         policy_class_name = 'ResidualActorCritic'
#     #     # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
#     #     skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
#     #     # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Oct14_10-20-05_/model_1500.pt"] #high pid walking
#         skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]

class A1DoorOpeningv2CfgPPO( LeggedRobotCfgPPO ): # original
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.9 #0.05
        residual_weight_penalty_coef = 0.0
        # max_iterations = 3500
        

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "door_open":[512, 256],
                             "target_reach":[256,128],
                             "residual": [512,256]}
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,5],
                             "door_open":[512, 256, 128,2]}
        skill_compositions = {"straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left","target_reach"],
                              "door_open": ["straight_walk", "door_open"],
                              "residual": ["residual"]}
        meta_backbone_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        synthetic_obs_scales = {"target_position":3}
        
        
    class runner( LeggedRobotCfgPPO.runner ):
        load_run = "Oct28_15-39-15_"
        resume = True
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "1700" # -1 = last saved model
        # load_run = "Jul18_12-51-06_"
        # load_run = "Jul19_18-49-59_"
        # load_run = "Jul21_11-18-51_" oldbest?
        load_run = "Oct16_17-58-57_"
        max_iterations = 4500
        checkpoint = 3500
        obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "door_position": 2,
                    "door_angle": 1,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12,
                    "robot_angle":1,
                    # "robot_position":2,
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
                                "robot_angle",
                                # "robot_position"
                                ]
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
                    "robot_angle",
                    # "robot_position"
                    ]
                      ]
        meta_network_obs = ["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_angle",
                                # "robot_position"
                                ]
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
                        # "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Oct10_14-14-11_/model_4500.pt", #reach target with residual
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt"  
                }
##############################################################################################################
################################         Target Reach        #################################################
##############################################################################################################        
class A1MultiSkillReachCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 2.0 *0# was 0.05
        residual_weight_penalty_coef = 2.0 *0
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128], [256, 128], [256, 128],[256,128]] # added residual
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    
    
    # class runner( LeggedRobotCfgPPO.runner ): # vanilla
    #     max_iterations = 9000
    #     save_interval = 1000
    #     experiment_name = 'multiskill_targetreach'
    #     policy_class_name = 'SkillActorCritic'     
    #     # load_run = "Nov28_19-57-17_"
    #     # load_run = "Nov26_11-18-44_"
    #     load_run = "Nov28_20-43-29_"
    #     # load_run = "Mar08_19-08-08_"
    #     load_run = "Mar09_11-27-11_" # vanilla policy
    #     # load_run = "Mar08_19-08-08_"
    #     # load_run = "Mar11_13-20-36_"
    #     # resume = True
    #         # checkpoint = 1000
            
    #         # resume = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        save_interval = 500
        # load_run = "May12_19-48-31_"
        # load_run = "May16_09-46-39_"
        # load_run = "May16_21-49-42_"
        # load_run = "May19_22-33-24_/model_3500.pt"
        # load_run = "Jun09_14-54-30_"
        # load_run = "Jun27_15-27-05_"
        # load_run = "Jun30_10-56-57_"
        # load_run = "Jul06_16-49-08_"
        # load_run = "Jul07_10-24-55_"
        # load_run = "Sep28_20-15-41_"
        load_run = "Dec02_13-17-09_" # big policy
        # load_run = "Oct10_14-14-11_"
        # load_run = "Sep30_10-28-01_" # paper Table run 
        load_run = "Mar08_14-31-10_" # best run
        # load_run = "Mar11_13-20-36_" #naive MCP + residual
        # load_run = "Mar29_15-11-05_" # No residual
        # load_run = ""
        # resume = True
        max_iterations = 5500
        # checkpoint = 3500
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
                    "actions"],

                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
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
        # sim_results
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
        #                ]
        
        # hardware results
        # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Sep28_19-15-56_/model_150.pt", #turn right
        #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Sep26_15-52-16_/model_150.pt", #turn left
        #                ]
        # March 2023 experiments
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt",
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-09-15_/model_150.pt", #turn right
                       "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-16-26_/model_150.pt", #turn left
                       ]
class A1TargetReachCfg( A1FlatCfg):
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 47
        env_spacing = 5.
    
    class viewer:
        ref_env = 0
        pos = [-1, 3, 1]  # [m]
        lookat = [0., 0., 0.]  # [m]
    class rewards:

        # class scales: # mcp + residual
        #     termination = -0.0
        #     tracking_lin_vel = 0.0
        #     tracking_ang_vel = 0.0
        #     lin_vel_z = -2.0
        #     ang_vel_xy = -0.00
        #     orientation = -0.5
        #     dof_vel = -0.
        #     dof_acc = -2.5e-7
        #     base_height = -0.1
        #     feet_air_time =  1.0
        #     collision = -1.
        #     feet_stumble = -0.0 
        #     action_rate = -0.1
        #     stand_still = -0.
        #     torques = -0.0002
        #     dof_pos_limits = -10.0
        #     target_reach = 2.0        
        # class scales: # vanilla
        #     termination = -0.0
        #     tracking_lin_vel = 0.0
        #     tracking_ang_vel = 0.0
        #     lin_vel_z = -2.0
        #     ang_vel_xy = -0.00
        #     orientation = -0.5
        #     dof_vel = -0.
        #     dof_acc = -2.5e-7
        #     base_height = -0.1
        #     feet_air_time =  1.0
        #     collision = -1.
        #     feet_stumble = -0.0 
        #     action_rate = -0.1
        #     stand_still = -0.
        #     torques = -0.0002
        #     dof_pos_limits = -10.0
        #     target_reach = 2.0
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
            target_reach = 2.0
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
# class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#         # residual_action_penalty_coef = 0.015
#         # residual_action_penalty_coef = 5 #1.5 #0.05
#         # residual_weight_penalty_coef = 0.01
#         residual_weight_penalty_coef = 0.005*0#*0+0.02#0.09
#         residual_action_penalty_coef = 0.025*0

#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [[512, 256, 128], [512, 256,128]]
#         critic_hidden_dims = [512, 256, 128]
#         weight_network_dims = [512,256]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
#     class runner( LeggedRobotCfgPPO.runner ):
#         # run_name = ''
#         checkpoint = 2000
#         # load_run = "Nov02_22-20-21_"
#         # load_run = "Nov02_09-31-03_"
#         # load_run = "Nov02_13-32-38_"
#         # load_run = "Nov02_19-17-17_"
#         # load_run = "Nov02_22-20-21_" #1st real world
#         load_run = "Mar18_16-43-43_"
#         load_run = "Mar19_15-43-07_"
#         # load_run = "Mar19_18-28-25_"
#         load_run = "Mar21_11-01-03_"
#         load_run = "Mar22_14-22-46_"
#         # resume = True
#         algorithm_class_name = 'ResidualPPO'
#         max_iterations = 9000
#         save_interval = 500
#         obs_sizes = {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "projected_gravity": 3,
#                     "target_position": 2,
#                     "object_position": 2,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12}
#         actor_obs =[["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position", 
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                     ]
#         critic_obs = [                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                       ]
#         experiment_name = 'multiskill_object_push'
#         policy_class_name = 'ResidualActorCritic'
#         # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
#         skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]

# class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ): # doesnt train for even static target
#     # todo early termination if robot is too far from object
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#         residual_weight_penalty_coef = 0.09*0
#         residual_action_penalty_coef = 0.01
        
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = {"straight_walk":[512, 256, 128],
#                              "standing": [256,128],
#                              "turn_left":[256, 128],
#                              "turn_right":[256, 128],
#                              "target_reach":[256,128],
#                              "residual": [512,256]}
#         critic_hidden_dims = [512, 256, 128]
#         weight_hidden_dims = {"turn_left":[128,2],
#                               "turn_right":[128,2], 
#                              "target_reach":[512, 256, 128,5]}
#         skill_compositions = {"straight_walk": ["straight_walk"],
#                               "standing": ["standing"],
#                               "turn_left": ["straight_walk", "turn_left"],
#                               "turn_right": ["straight_walk", "turn_right"],
#                               "target_reach": ["straight_walk", "standing","turn_right","turn_left","target_reach"],
#                               "residual": ["residual"]}
#         meta_backbone_dims = [256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         synthetic_obs_scales = {"synth_target_position":3}
#         synthetic_obs_ingredients = ["target_position",
#                                     "object_position"]
        
        
#     class runner( LeggedRobotCfgPPO.runner ):
#         max_iterations = 4500
#         checkpoint = 3500
#         algorithm_class_name = 'ResidualPPO'
        
        
#         obs_sizes = {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "projected_gravity": 3,
#                     "target_position": 2,
#                     "object_position": 2,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12,
#                     # synthetic observations after this point
#                     "synth_target_position": 2,
#                     }
#         actor_obs ={
#                     "straight_walk":["scaled_base_lin_vel",
#                                         "scaled_base_ang_vel",
#                                         "projected_gravity",
#                                         "relative_dof",
#                                         "scaled_dof_vel",
#                                         "actions"],
                    
#                     "standing":["scaled_base_lin_vel",
#                                 "scaled_base_ang_vel",
#                                 "projected_gravity",
#                                 "relative_dof",
#                                 "scaled_dof_vel",
#                                 "actions"],
                    
#                     "turn_left":["scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"],
                    
#                     "turn_right":["scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"],

#                     "target_reach":["scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "synth_target_position",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"],
                   
#                     "residual":["scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "target_position",
#                                     "object_position",
#                                     "synth_target_position",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"]
#                     }
#         critic_obs = [                    
#                     [               "scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "target_position",
#                                     "object_position",
#                                     "synth_target_position",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"]
#                       ]
#         meta_network_obs = ["scaled_base_lin_vel",
#                                     "scaled_base_ang_vel",
#                                     "projected_gravity",
#                                     "target_position",
#                                     "object_position",
#                                     "synth_target_position",
#                                     "relative_dof",
#                                     "scaled_dof_vel",
#                                     "actions"]
#         experiment_name = 'multiskill_object_push'
#         policy_class_name = 'MultiSkillActorCriticSplit'
#         # # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt",
#         # #                 "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_12-04-19_/model_200.pt", #turn right
#         # #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun08_10-57-33_/model_200.pt", #turn left
#         # #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jun09_17-20-37_/model_1150.pt",
#         # #                "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/May19_14-20-14_/model_750.pt"]
#         skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
#                        "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
#                         "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
#                         "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
#                         "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Oct10_14-14-11_/model_4500.pt", #reach target with residual
#                 }        
#         # skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1550.pt",
#         #                "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
#         #                 "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-09-15_/model_150.pt", #turn right
#         #                 "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-16-26_/model_150.pt", #turn left
#         #                 "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Sep30_10-28-01_/model_5500.pt", #reach target with residual
# # #                 # }    
# Multiskill push paper version
# class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#         residual_action_penalty_coef = 0.5*0.01 # was 0.05
#         residual_weight_penalty_coef = 0.5*0.01
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [[512, 256, 128], [256, 128], [256, 128], [256, 128],[256,128]] # added residual
#         critic_hidden_dims = [512, 256, 128]
#         weight_network_dims = [512, 256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    

#     #         # resume = True
#     class runner( LeggedRobotCfgPPO.runner ):
#         run_name = ''
#         algorithm_class_name = 'ResidualPPO'
#         save_interval = 500
#         max_iterations = 5500
#         load_run = "Mar22_14-22-46_"
#         load_run = "Jul17_17-53-58_"
#         load_run = "Jul18_16-34-44_"
#         load_run = "Jul23_18-11-50_"
#         load_run = "Jul24_19-52-28_"
#         load_run = "Jul30_15-34-15_"
#         load_run = "Jul30_20-42-15_"
#         # load_run = "Apr04_18-41-45_"
#         # resume = True
#         # checkpoint = 3500
#         obs_sizes = {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "projected_gravity": 3,
#                     "target_position": 2,
#                     "object_position": 2,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12}
#         actor_obs =[["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                     ]
#         critic_obs = [                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                       ]
#         experiment_name = 'multiskill_object_push'
#         policy_class_name = 'MultiSkillActorCritic'
#         skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt",
#                        "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
#                        "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-09-15_/model_150.pt", #turn right
#                        "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Mar07_20-16-26_/model_150.pt", #turn left
#                        ]    
# # one step residual
class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # residual_action_penalty_coef = 0.015
        # residual_action_penalty_coef = 5 #1.5 #0.05
        # residual_weight_penalty_coef = 0.01
        # residual_weight_penalty_coef = 0.005#*0+0.02#0.09
        # residual_action_penalty_coef = 0.025
        residual_weight_penalty_coef = 0.5*0.02
        residual_action_penalty_coef = 0.5*0.02
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512, 256,128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512,256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        # run_name = ''
        checkpoint = 2000
        # load_run = "Nov02_22-20-21_"
        # load_run = "Nov02_09-31-03_"
        # load_run = "Nov02_13-32-38_"
        # load_run = "Nov02_19-17-17_"
        # load_run = "Nov02_22-20-21_" #1st real world
        load_run = "Mar18_16-43-43_"
        load_run = "Mar19_15-43-07_"
        # load_run = "Mar19_18-28-25_"
        load_run = "Mar21_11-01-03_"
        load_run = "Aug05_01-16-36_"
        load_run = "Aug07_12-58-09_"
        load_run = "Aug13_19-07-25_"
        load_run = "Aug14_01-37-25_"
        load_run = "Aug14_17-19-57_"
        load_run = "Aug15_02-49-24_"
        # resume = True
        algorithm_class_name = 'ResidualPPO'
        max_iterations = 20000
        save_interval = 500
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
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]


class A1TargetObjectPushCfg( A1FlatCfg):
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 49
        env_spacing = 10.
        episode_length_s = 5

    class viewer:
        ref_env = 0
        pos = [0, 3, 1]  # [m]
        lookat = [1., 1., 0.]  # [m]
# class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#         # residual_action_penalty_coef = 0.015
#         # residual_action_penalty_coef = 5 #1.5 #0.05
#         # residual_weight_penalty_coef = 0.01
#         residual_weight_penalty_coef = 0.005#*0+0.02#0.09
#         residual_action_penalty_coef = 0.025

#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [[512, 256, 128], [512, 256,128]]
#         critic_hidden_dims = [512, 256, 128]
#         weight_network_dims = [512,256]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
#     class runner( LeggedRobotCfgPPO.runner ):
#         # run_name = ''
#         checkpoint = 2000
#         # load_run = "Nov02_22-20-21_"
#         # load_run = "Nov02_09-31-03_"
#         # load_run = "Nov02_13-32-38_"
#         # load_run = "Nov02_19-17-17_"
#         # load_run = "Nov02_22-20-21_" #1st real world
#         load_run = "Mar18_16-43-43_"
#         load_run = "Mar19_15-43-07_"
#         # load_run = "Mar19_18-28-25_"
#         load_run = "Mar21_11-01-03_"
#         # resume = True
#         algorithm_class_name = 'ResidualPPO'
#         max_iterations = 9000
#         save_interval = 500
#         obs_sizes = {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "projected_gravity": 3,
#                     "target_position": 2,
#                     "object_position": 2,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12}
#         actor_obs =[["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position", 
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                     ]
#         critic_obs = [                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "target_position",
#                     "object_position",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"]
#                       ]
#         experiment_name = 'multiskill_object_push'
#         policy_class_name = 'ResidualActorCritic'
#         # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
#         skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]

    #     terminate_after_contacts_on = ["face","base", "hip", "thigh"]
    #     # penalize_contacts_on = ["face"]
    #     self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-0., 0.]       
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.4*0+0.01 #0.5
            tracking_ang_vel = 0.4*0+0.01
            lin_vel_z = -2.0
            ang_vel_xy = -0.00
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1*5
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002*0.5
            dof_pos_limits = -10.0
            object_target_dist = 2.0
            # robot_target_dist = 2 
            # object_robot_dist = 0.5
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
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
            feet_air_time =  3.0
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
        # terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
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
        pos = [0, -2, 3.0]  # [m]
        lookat = [0., 1.5, 0.]  # [m]
    # class viewer:
    #     ref_env = 0
    #     pos = [64, 0.1, 9]  # [m]
    #     lookat = [64., 0., 0.]  # [m]
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
            tracking_lin_vel = 3.0 #1.0
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
            torques = -0.002
            dof_pos_limits = -10.0
            door_angle = 2.0 #0.8 #4.0 maybe its 2->1?
            robot_target_dist = 5.0
            feet_contact_forces = -0.5
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
        residual_action_penalty_coef = 5 #1.5 #0.05
        residual_weight_penalty_coef = 5
        gamma = 0.99
        
        # max_iterations = 3500
        

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = {"straight_walk":[512, 256, 128],
                             "standing": [256,128],
                             "turn_left":[256, 128],
                             "turn_right":[256, 128],
                             "target_reach": [256, 128],
                             "door_open":[512, 256],
                             "door_openv2": [512,256],
                             "crouching": [256,128],
                             "residual": [512,256,128]
                             }
        critic_hidden_dims = [512, 256, 128]
        weight_hidden_dims = {"turn_left":[128,2],
                              "turn_right":[128,2], 
                             "target_reach":[512, 256, 128,5],
                             "door_open":[512, 256, 128,2],
                             "door_openv2":[256,128,9],
                             "crouching": [512,256,2]
                             }
        skill_compositions = {
                            #   "straight_walk": ["straight_walk"],
                              "standing": ["standing"],
                              "turn_left": ["straight_walk", "turn_left"],
                              "turn_right": ["straight_walk", "turn_right"],
                              "target_reach": ["straight_walk", "standing","turn_right","turn_left","target_reach"],
                              "door_open": ["straight_walk", "door_open"],
                              "door_openv2":["straight_walk", "standing","turn_right","turn_left","target_reach","door_open","door_openv2"],
                              "crouching": ["straight_walk","crouching"],
                              "residual": ["residual"]
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
                                    #  "robot_position",
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
        # load_run="Oct29_19-04-07_"
        # resume = True
    #     run_name = ''
        algorithm_class_name = 'ResidualPPO'
        save_interval = 500
    #     # checkpoint = "1700" # -1 = last saved model
    #     # load_run = "Jul18_12-51-06_"
    #     # load_run = "Jul26_15-15-48_"
    #     # load_run = "Jul26_07-10-21_" # no residual penalty
    #     # load_run = "Jul27_10-59-17_" # no residual network
    #     # load_run = "Aug05_10-29-06_"
    #     # load_run = "Aug09_20-25-14_"
    #     # load_run = "Aug10_11-27-03_" # 90% with table, continued from Aug09_20-25-14_
    #     # load_run = "Aug11_01-18-09_" #90% with table from scratch
    #     # load_run = "Aug15_13-24-32_" 
    #     # load_run = "Aug16_22-33-13_" # finetuned from Aug15_13-24-32_
        
    #     # load_run = "Aug17_19-54-25_"
    #     # load_run = "Aug18_12-53-21_"
    #     # load_run = "Aug20_01-05-14_"
    #     # load_run = "Aug23_17-42-04_"
    #     # load_run = "Aug27_10-34-57_"
    #     # load_run = "Aug28_20-11-06_"
    #     # load_run = "Sep05_15-03-25_"
    #     load_run = "Sep09_15-04-40_" # LAst working policy
    #     # load_run = "Oct04_18-39-17_"
    #     load_run = "Oct17_16-32-28_"
        load_run = "Oct19_21-08-53_"
    #     resume = True
    #     # checkpoint = 5000
    #     # resume_path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Jul21_11-18-51_/model_5750.pt"
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
                                "synth_target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_angle"
                                # "robot_position"
                                ],
                    
                    "crouching":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "projected_gravity",
                                "height",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions"],
                    
                    "residual":["scaled_base_lin_vel",
                                "scaled_base_ang_vel",
                                "robot_angle",
                                "projected_gravity",
                                "synth_target_position",
                                "target_position",
                                "door_position",
                                "door_angle",
                                "relative_dof",
                                "scaled_dof_vel",
                                "actions",
                                "robot_position",
                                "target_room",
                                "robot_room"]
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
        # results from Spet 2022
        # skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
        #                "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
        #                 "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
        #                 "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
        #                 "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Jul07_10-24-55_/model_1500.pt", #reach target
        #                 "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt",
        #                 "door_openv2":"/home/niranjan/PrTrue
        # unified models, door randomness
        skill_paths = {"straight_walk":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt",
                       "standing":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/standing/Jun30_13-09-00_/model_100.pt", #standing
                        "turn_right":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-41-47_/model_100.pt", #turn right
                        "turn_left":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/turning/Jun24_13-48-02_/model_100.pt", #turn left
                        "target_reach":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/multiskill_targetreach/Oct10_14-14-11_/model_4500.pt", #reach target
                        "door_open":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropen/Jul11_13-04-00_/model_1500.pt",
                        "door_openv2":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/dooropenv2/Oct16_17-58-57_/model_6000.pt",
                        "crouching":"/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/crouching/Aug21_22-24-48_/model_6000.pt"
                }
        
        
# #MCP + residual

# class InteractiveTargetReachv3CfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#         # residual_action_penalty_coef = 0.015
#         # residual_action_penalty_coef = 5 #1.5 #0.05
#         # residual_weight_penalty_coef = 0.01
#         residual_weight_penalty_coef = 0.005*0#*0+0.02#0.09
#         residual_action_penalty_coef = 0.025*0

#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [[512, 256, 128], [512, 256,128]]
#         critic_hidden_dims = [512, 256, 128]
#         weight_network_dims = [512,256]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
#     class runner( LeggedRobotCfgPPO.runner ):
#         algorithm_class_name = 'ResidualPPO'
#         max_iterations = 9000
#         save_interval = 500
#         obs_sizes = {"scaled_base_lin_vel": 3,
#                     "scaled_base_ang_vel": 3,
#                     "robot_angle": 1,
#                     "projected_gravity": 3,
#                     "target_position": 2,
#                     "door_position": 2,
#                     "table_position":2,
#                     "couch_position":2,
#                     "door_angle": 1,
#                     "relative_dof": 12,
#                     "scaled_dof_vel": 12,
#                     "actions": 12,
#                     "robot_position":2,
#                     "target_room":1,
#                     "robot_room":1,
#                     "height": 1}
#         actor_obs =[["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "projected_gravity",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions"],
                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "robot_angle",
#                     "projected_gravity",
#                     # "synth_target_position",
#                     "target_position",
#                     "door_position",
#                     "table_position",
#                     "couch_position",
#                     "door_angle",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions",
#                     "robot_position",
#                     "target_room",
#                     "robot_room",
#                     "height"]
#                     ]
#         critic_obs = [                    
#                     ["scaled_base_lin_vel",
#                     "scaled_base_ang_vel",
#                     "robot_angle",
#                     "projected_gravity",
#                     # "synth_target_position",
#                     "target_position",
#                     "door_position",
#                     "table_position",
#                     "couch_position",
#                     "door_angle",
#                     "relative_dof",
#                     "scaled_dof_vel",
#                     "actions",
#                     "robot_position",
#                     "target_room",
#                     "robot_room",
#                     "height"]
#                       ]
#         experiment_name = 'multiskill_object_push'
#         policy_class_name = 'ResidualActorCritic'
#         # skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]
#         skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Sep26_17-16-44_/model_1500.pt"]

####################################################################################################################################################
##############################          Two leg balance         ####################################################################################
####################################################################################################################################################

class TwoLegBalanceCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.001

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'two_leg_balance'
        policy_class_name = 'SkillActorCritic'     
        load_run = "Apr09_16-49-21_" # standing
        load_run = "Apr09_18-30-03_" # standing 3 layer 512,256,128
        load_run = "Apr10_13-00-39_"
        # load_run = "Apr06_18-00-37_"
        # load_run = "Mar27_14-58-33_"
        # load_run = "Mar27_16-24-20_"
        # load_run = "Mar27_19-46-48_"
        # load_run = "Mar28_11-38-57_"
        # load_run = "Apr06_13-52-16_"
        # load_run = "Apr06_14-50-05_" 
        # load_run = "Apr07_18-44-25_" 
        # load_run = "Apr09_15-01-17_"
        load_run = "Apr10_13-42-23_" # with base height and feet air time terms and possibly low gain
        # load_run = "Apr10_15-15-52_" # started from Apr10_13-42-23_ - successful balancing no DR
        # load_run ="Apr17_01-33-33_" # started from Apr10_13-42-23_ - successful balancing with mass -1,1 DR
        # load_run = "Apr17_13-10-02_" # started from Apr10_13-42-23_ - successful balancing with mass [-1,1], friction [0.5, 1.25] DR
        # load_run = "Apr25_18-15-45_"
        # load_run = "Apr28_18-23-46_"
        # load_run = "Apr29_19-22-00_"
        load_run = "May02_20-25-46_" # standing policy used to warmstart
        # load_run = "May03_11-41-40_"  # balancing policy with no DR
        # load_run = "May03_20-45-06_" # balancing policy with mass DR [-1,1]
        # load_run = "May04_16-32-21_" # balancing policy with mass DR [-1,1] and setting body props
        # load_run = "May05_17-52-22_" # # standing policy used to warmstart with low gain
        load_run = "May07_17-36-07_" # # standing policy used to warmstart with low gain
        load_run = "May09_00-00-04_"
        load_run = "May09_16-07-58_"
        load_run = "May10_11-08-49_"
        load_run = "May10_17-17-42_"
        load_run = "May11_11-02-20_"
        load_run = "May15_19-33-32_"
        # load_run = "May20_13-56-03_"
        load_run = "May21_16-29-35_"
        load_run = "May20_13-56-03_"
        load_run = "May23_17-38-31_"
        load_run = "May27_19-22-33"
        load_run = "May28_12-49-19_"
        load_run = "May28_18-23-48_"
        # load_run = "May22_18-38-51_"
        save_interval = 1000
        # max_iterations = 15000
        max_iterations = 20000
        # checkpoint = 10000
        resume = False
    
    class policy:
        init_noise_std = 0.001
        actor_hidden_dims = [512,256, 256, 128]
        critic_hidden_dims = [512,256, 256,128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

class TwoLegBalanceCfg( A1RoughCfg):
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        #for sim
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5} 
        # for real robot
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class init_state( A1RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.7,     # [rad]
            'RL_thigh_joint': 0.7,   # [rad]
            'FR_thigh_joint': 0.7,     # [rad]
            'RR_thigh_joint': 0.7,   # [rad]

            'FL_calf_joint': -1.4,   # [rad]
            'RL_calf_joint': -1.4,    # [rad]
            'FR_calf_joint': -1.4,  # [rad]
            'RR_calf_joint': -1.4,    # [rad]
        }

    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

  

    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_mass_shift.urdf'
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip", "face"]#,"front_mass","back_mass"]
        penalize_contacts_on = ["thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
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
        randomize_friction = False
        friction_range = [0.5, 1.25]
        friction_range_stddev = [0.01,0.5]
        # friction_range = [0.75, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        added_mass_range_stddev = [0.01, 0.5]
        push_robots = True
        push_interval_s = 1
        max_push_vel_xy = 1.
        randomize_com = True
        com_shift_mass_range = [0.1, 3.0]
        com_shift_mass_range_stddev = [0.01, 2.0]
        com_sampled = [np.random.uniform(*com_shift_mass_range),np.random.uniform(*com_shift_mass_range_stddev)]
        distribution = 'truncated_normal' # uniform, truncated_normal
    
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.1
            lin_vel_z = -2.0*0
            ang_vel_xy = -0.05
            orientation = -0.1 -0.4
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -5.5
            feet_air_time =  3.0*0 # this was 0 before 16:44 May 2
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002
            dof_pos_limits = -10.0
            foot_contact = 1.0
            foot_lift = 10.0
            # action_magnitude= -1.0
            # door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.3 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100. # forces above this value are penalized

    class noise:
        add_noise = True
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
            
class A1BayrnPushCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_weight_penalty_coef = 0.5*0.02
        residual_action_penalty_coef = 0.5*0.02
    class policy:
        init_noise_std = 1.0*5
        actor_hidden_dims = [[512, 256, 128], [512, 256,128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512,256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        # run_name = ''
        checkpoint = 2000
        load_run = "Aug13_19-07-25_"
        load_run = "Aug14_01-37-25_"
        load_run = "Aug14_17-19-57_"
        load_run = "Aug15_02-49-24_"
        load_run = "Aug24_12-14-07_"
        resume = True
        checkpoint= 3000
        algorithm_class_name = 'ResidualPPO'
        max_iterations = 1
        save_interval = 500
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
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/a1_flat/Jun21_17-35-58_/model_1500.pt"]


class A1BayrnPushCfg( A1FlatCfg):
    class asset( A1RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_face.urdf'
        name = "a1"
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        penalize_contacts_on = ["face"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 49
        env_spacing = 10.
        episode_length_s = 5

    class viewer:
        ref_env = 0
        pos = [0, 3, 1]  # [m]
        lookat = [1., 1., 0.]  # [m]

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-0., 0.]       

    class domain_rand:
        push_robots = True
        push_interval_s = 1
        max_push_vel_xy = 1.
        randomize_base_mass = False
        randomize_friction = False
        friction_range = [0.5, 1.25]
        friction_range_stddev = [0.01,0.5]
        # friction_range = [0.75, 1.0]
        
        randomize_com = False
        com_shift_mass_range = [0.1, 3.0]
        com_shift_mass_range_stddev = [0.01, 2.0]
        com_sampled = [np.random.uniform(*com_shift_mass_range),np.random.uniform(*com_shift_mass_range_stddev)]
        
        randomize_puck_mass = True
        puck_mass_range = [0.1, 1.]
        puck_mass_range_stddev = [0.01, 0.1]
        puck_mass_sampled = [np.random.uniform(*puck_mass_range),np.random.uniform(*puck_mass_range_stddev)]

        distribution = 'truncated_normal' # uniform, truncated_normal
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.4*0+0.01 #0.5
            tracking_ang_vel = 0.4*0+0.01
            lin_vel_z = -2.0
            ang_vel_xy = -0.00
            orientation = -0.
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.1*5
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            torques = -0.0002*0.5
            dof_pos_limits = -10.0
            object_target_dist = 2.0*2
            # robot_target_dist = 2 
            # object_robot_dist = 0.1
            # door_angle = 0.6
            # box_moved = 2.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        #for sim
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5} 
        # for real robot
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4