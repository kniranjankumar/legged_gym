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
            
class StraightWalkingRobot(LeggedRobot):
    def __init__(self,*args,**kwargs):
        super(StraightWalkingRobot, self).__init__(*args,**kwargs)
        self.observation_types = ["scaled_base_lin_vel",
                                  "scaled_base_ang_vel",
                                  "projected_gravity",
                                #   "filler",
                                  "relative_dof",
                                  "scaled_dof_vel",
                                  "actions"]
        
    
    def compute_observations(self):
        """Compute observations by using the skill' observation_types
        """
        # print(self.contact_forces[0])
        
        self.scaled_base_lin_vel = self.base_lin_vel * self.obs_scales.lin_vel
        # self.scaled_base_lin_vel[:,0] = 1.0
        self.scaled_base_ang_vel = self.base_ang_vel  * self.obs_scales.ang_vel
        self.relative_dof = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        self.scaled_dof_vel = self.dof_vel * self.obs_scales.dof_vel
        # self.filler = torch.zeros_like(self.scaled_base_lin_vel)[:,:-1]
        obs_list = [self.__getattribute__(obs_name) for obs_name in self.observation_types]
        self.obs_buf = torch.cat(obs_list, dim=-1)
        # print(self.projected_gravity[0,:])
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # print(self.scaled_dof_vel[0,:])
        # print(self.p_gains, self.d_gains)
    
    # def step(self, actions):
    #     clip_actions = self.cfg.normalization.clip_actions
    #     actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    #     diff = torch.clip(self.actions-actions,-0.5,0.5)
    #     actions = self.actions-diff
    #     return super().step(actions)
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # robot_angle = torch.tensor([[ 0.0348995, 0, 0, 0.9993908 ]], device=self.device)
        # g_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).view(1,-1)
        # print(robot_angle,g_vec)
        # print(quat_rotate_inverse(robot_angle, g_vec ))
        # if 1 in env_ids:
        #     self.p_gains = torch.rand_like(self.p_gains)*20+20
        #     self.d_gains = torch.rand_like(self.d_gains)*0.3+0.4
            
            
            
    