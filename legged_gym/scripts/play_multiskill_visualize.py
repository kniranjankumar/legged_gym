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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
print("failed", env_ids, success)
from matplotlib import pyplot as plt
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
print("failed", env_ids, success)
import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    # env_cfg.terrain.curriculum = True
    # env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.push_robots = True
    # env_cfg.viewer.lookat = [1.2,0,0.2]
    # env_cfg.viewer.pos = [1.2,-2,0.6]
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_multiskill_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    #### Code to plot weights
    fig, (ax, ax1) = plt.subplots(2,1)
    # ax.set_aspect('equal')
    bars = ('Walk', 'Stand', 'Right', 'Left', 'Residual')
    # ax.hold(True)
    # plt.show(False)
    plt.draw()
    num_skills = len(train_cfg.runner.skill_paths)
    chart = ax.bar(bars, [1.0]*(num_skills+1))
    # ax.xticks(range(num_skills+1), bars)
    weights_list = []
    success_count = 0
    total_count = 0
    
    for i in range(int(env.max_episode_length)):
        actions = policy(obs.detach())
        # print(ppo_runner.alg.actor_critic.visualize_weights)
        
        #### update weights plot
        for rect, weight in zip(chart, ppo_runner.alg.actor_critic.visualize_weights):
            rect.set_height(weight)
        weights_list.append(ppo_runner.alg.actor_critic.visualize_weights[0])
        # chart.set_data([1,2], ppo_runner.alg.actor_critic.visualize_weights)
        fig.canvas.draw()
        plt.pause(0.0001)
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        if "success" in infos["episode"].keys():
            success_count += torch.sum(infos["episode"]["success"])
            total_count = infos["episode"]["success"].size()[0]
        if dones[0] == True:
            ax1.clear()
            num_pts2plot = min(len(weights_list)-1, 500)
            ax1.plot(weights_list[:num_pts2plot])        
            weights_list = []
            # print("Should be plotting now")    
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
    print("success/total", success_count, total_count, success_count/total_count,infos["episode"]["success"].size())
    print(done)
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
