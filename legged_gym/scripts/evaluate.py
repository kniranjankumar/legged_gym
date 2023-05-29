import numpy as np
import os
from datetime import datetime
from abc import ABC, abstractmethod

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
# update_cfg_from_args
import torch
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt import UtilityFunction
from multiprocessing import Process, Queue
from typing import Tuple, Union
from statistics import mean, stdev


def train(args, env_cfg,train_cfg=None):
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # rew_buffer = ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # evaluate and return the mean reward for Bayesian optimization
    # write eval code for target env
    mean_reward = 0
    return mean_reward, ppo_runner, train_cfg
    
def evaluate(true_parameters, runner):
    print("Evaluating in true environment...")
    env = runner.env
    # runner.resume=True
    set_parameters(env.cfg.domain_rand, true_parameters)
    print(env.cfg.domain_rand.friction_range)
    env.reset_dr_params()
    with torch.inference_mode():
        obs_,_ = env.reset()
    obs = obs_.clone()
    print(env.device)
    # policy = runner.alg.act
    policy = runner.get_inference_policy()
    mean_total_rewards = []
    # print(int(env.max_episode_length))
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    with torch.inference_mode():
 
        for i in range(2*int(env.max_episode_length)):
            # print(i)
            
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            cur_reward_sum += rews
            new_ids = (dones > 0).nonzero(as_tuple=False)
            mean_total_rewards.extend(cur_reward_sum[new_ids][:,0].tolist())
            cur_reward_sum[new_ids] = 0
        
    # runner.alg.actor_critic.train()
    return np.mean(mean_total_rewards)

def set_parameters(cfg, parameters):
    print("setting parameters...", parameters)
    # cfg.friction_range = [parameters['friction']]*2
    # cfg.added_mass_range = [parameters['mass']]*2
    # cfg.com_shift_mass_range = [parameters['com_shift']]*2
    # cfg.friction_range_stddev = [parameters['friction_stddev']]*2
    # cfg.added_mass_range_stddev = [parameters['mass_stddev']]*2
    # cfg.com_shift_mass_range_stddev = [parameters['com_shift_stddev']]*2
    cfg.com_sampled = [parameters['com_shift'], parameters['com_shift_stddev']]
    
    return cfg
    
if __name__ == '__main__':
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    # print()
    # print(args)
    # initialize Bayesian optimization
    bounds = {"friction":(0.5,1.25), "mass":(-1,1), "com_shift":(0.1,2),
              "friction_stddev":(0.01,0.5), "mass_stddev":(0.01,0.5), "com_shift_stddev":(0.01,0.5)
              }
    # true_parameters = {"friction":0.75, "mass":0.3, "com_shift":0.3,
    #           "friction_stddev":0.1, "mass_stddev":0.1, "com_shift_stddev":0.1
    #           }
    true_parameters = {
                    #    "friction":0.75, 
                    #    "mass":0.3, 
                       "com_shift":args.com_mean,
                    #    "friction_stddev":0.1, 
                    #    "mass_stddev":0.1, 
                       "com_shift_stddev":0.1
              }

    checkpoints = [i*1000 for i in range(20)]
    # checkpoints = [11000]
    _, ppo_runner, train_cfg = train(args,env_cfg)
    rewards = []
    for ckpt in checkpoints:
        path = "/nethome/nkannabiran3/Projects/curious_dog_isaac/legged_gym//logs/two_leg_balance/May28_18-26-34_/model_"+str(ckpt)+".pt"
        ppo_runner.load(path, load_optimizer=True)
    
        target = evaluate(true_parameters, ppo_runner)
        print("Mean eps reward at",ckpt,": ", target)
        rewards.append((ckpt,target))
    print(rewards)
            


    