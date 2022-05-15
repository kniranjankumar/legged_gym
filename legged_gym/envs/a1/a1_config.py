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

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'


class A1FlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'reachtarget'
        policy_class_name = 'SkillActorCritic'     
        
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
           
class A1FlatCfg( A1RoughCfg):
    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        
    class asset( A1RoughCfg.asset ):
        terminate_after_contacts_on = ["base", "FL_hip", "FR_hip", "RL_hip", "RR_hip"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 48
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
            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-0, 0]
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.1
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
            # door_angle = 0.6
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
            
class A1MultiSkillCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.009

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [256, 128]]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        checkpoint = "750" # -1 = last saved model
        load_run = "May10_21-50-29_"
        
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
        experiment_name = 'multiskill'
        policy_class_name = 'MultiSkillActorCritic'
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
        
class A1MultiSkillReachCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.05
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512, 256, 128]]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        algorithm_class_name = 'ResidualPPO'
        # checkpoint = "750" # -1 = last saved model
        load_run = "May12_19-48-31_"
        max_iterations = 3500
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
        skill_paths = ["/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/straight_walker/May06_14-52-51_/model_550.pt"]
        
class A1TargetReachCfg( A1FlatCfg):
    class env(A1RoughCfg.env ):
        num_envs = 4096
        num_observations = 47
        env_spacing = 5.
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.5
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

class A1MultiSkillObjectPushCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.02

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512, 256, 128]]
        critic_hidden_dims = [512, 256, 128]
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
        policy_class_name = 'MultiSkillActorCritic'
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
