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

from matplotlib import pyplot as plt
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import cv2
import numpy as np
import torch
import pandas as pd


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
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
    ppo_runner, train_cfg = task_registry.make_multiskill_alg_runnerv2(env=env, name=args.task, args=args, train_cfg=train_cfg)
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
    camera_vel = np.array([1., 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    #### Code to plot weights
    fig, (ax, ax2) = plt.subplots(2,1)
    # ax.set_aspect('equal')
    # ax.hold(True)
    # plt.show(False)
    plt.draw()
    # skills = set(train_cfg.policy.weight_hidden_dims.keys()).union(train_cfg.policy.skill_compositions.keys())
    skills = train_cfg.policy.skill_compositions.keys()
    num_skills = len(skills)
    chart = ax.bar(skills, [1.0]*(num_skills))
    ax.set_xticklabels(list(skills))
    weights_list = []
    trajectory_buffer = np.zeros([50,4000,2]) # 50 trajectories, 1000 steps, 2 dimensions
    all_weights = np.zeros([50,4000,7])
    dones_tracker = np.zeros([50]).astype("bool")
    map_bw = cv2.imread("./resources/2room_map_smaller.png")*0
    trajectory_length = 2000
    xs = torch.tensor([-3, -2.0, -1, 0, 1, 3, 4, 5, 6,7])
    ys = torch.tensor([-2.0, -1, 0, 1, 2])
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    x = x.numpy().reshape(-1)
    y = y.numpy().reshape(-1)
    print(x.shape, y.shape)
    # early_termination
    for i in range(3*int(env.max_episode_length)):
        actions = policy(obs.detach())
        if i>0:
            mask = np.tile(dones_tracker.astype("float").reshape(-1,1),(1,2))
            # print(mask.shape)
            trajectory_buffer[:,i,:] = env.get_robot_position().detach().cpu().numpy()*(1-mask)+ trajectory_buffer[:,i-1,:]*(mask)
        else:
            trajectory_buffer[:,i,:] = env.get_robot_position().detach().cpu().numpy()
        # print(ppo_runner.alg.actor_critic.visualize_weights)
        
        #### update weights plot
        for rect, weight in zip(chart, ppo_runner.alg.actor_critic.visualize_weights):
            rect.set_height(weight)
        weights_list.append(ppo_runner.alg.actor_critic.visualize_weights[0])
        all_weights[:,i,:] = ppo_runner.alg.actor_critic.all_weights.detach().cpu().numpy()
        # chart.set_data([1,2], ppo_runner.alg.actor_critic.visualize_weights)
        fig.canvas.draw()
        plt.pause(0.0001)
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        for j, done in enumerate(dones.detach().cpu().numpy()):
            if done and dones_tracker[j]==False:
                if np.linalg.norm(trajectory_buffer[j,i,:]-np.array([x[j],y[j]]))<0.2:
                    dones_tracker[j] = True
                else:
                    trajectory_buffer[j,:i+1,:] = env.get_robot_position().detach().cpu().numpy()[j,:]
                    # dones_tracker = np.logical_or(dones_tracker, dones.detach().cpu().numpy())
        if dones[0] == True:
            # ax1.clear()
            num_pts2plot = min(len(weights_list)-1, 500)
            # ax1.plot(weights_list[:num_pts2plot])        
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
    
        if i == trajectory_length:
            #### Plot trajectories
            ax2.imshow(map_bw, extent=[0, 1000, 0, 500])
            for j in range(50):
                # if not dones_tracker[j]:
                    x_image = trajectory_buffer[j,10:trajectory_length,0]*(1000//12)+330
                    y_image = trajectory_buffer[j,10:trajectory_length,1]*(500//6)+250
                    ax2.plot(x_image, y_image)
            # np.savetxt("x.csv", trajectory_buffer[:,:,0], delimiter=",")
            # np.savetxt("y.csv", trajectory_buffer[:,:,1], delimiter=",")
            weights = {"weights_"+str(w):all_weights[:,20:trajectory_length,w].reshape(-1) for w in range(7)}
            weights["x"] = trajectory_buffer[:,20:trajectory_length,0].reshape(-1)*(1000//12)+330
            weights["y"] = trajectory_buffer[:,20:trajectory_length,1].reshape(-1)*(500//6)+250
            df=pd.DataFrame(weights)
            df.to_csv('weights_dooropen.csv', header=False, index=False)
            break    

    plt.show()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
