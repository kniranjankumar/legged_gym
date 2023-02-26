from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.envs.a1.a1_config import A1FlatCfg, A1RoughCfgPPO
import numpy as np
import torch
import pybullet as p
            
class TurningRobot(LeggedRobot):
    def __init__(self,*args,**kwargs):
        super(TurningRobot, self).__init__(*args,**kwargs)
        self.observation_types = ["scaled_base_lin_vel",
                                  "scaled_base_ang_vel",
                                  "projected_gravity",
                                  "relative_dof",
                                  "scaled_dof_vel",
                                  "actions"]
        
    # def update_command_curriculum(self, env_ids):
    #     """ Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     print(self.command_ranges["ang_vel_yaw"][0],"is the command ang_velocity")
    #     # If the tracking reward is above 80% of the maximum, increase the range of commands
    #     if torch.mean(self.episode_sums["lin_vel_x"][env_ids]) / self.max_episode_length > 0.5 * self.reward_scales["lin_vel_x"]:
    #         # self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
    #         # self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    #         if self.command_ranges["lin_vel_x"][0]<3.0:
    #             self.command_ranges["lin_vel_x"][0]+=0.5
    #             self.command_ranges["lin_vel_x"][1]+=0.5
    #         print("==============================Tracking velocity:", self.command_ranges["lin_vel_x"])
    def compute_observations(self):
        """Compute observations by using the skill' observation_types
        """
        # print(self.projected_gravity[0].tolist())
        # print([item[0].tolist() for item in get_euler_xyz(self.base_quat)])
        # print(self.base_ang_vel[0])
        self.scaled_base_lin_vel = self.base_lin_vel * self.obs_scales.lin_vel
        self.scaled_base_ang_vel = self.base_ang_vel  * self.obs_scales.ang_vel
        self.relative_dof = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        self.scaled_dof_vel = self.dof_vel * self.obs_scales.dof_vel
        obs_list = [self.__getattribute__(obs_name) for obs_name in self.observation_types]
        self.obs_buf = torch.cat(obs_list, dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            
            
    