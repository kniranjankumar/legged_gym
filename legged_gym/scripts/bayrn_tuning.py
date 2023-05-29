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

class TuningStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.outcomes = []

    def add_outcome(self, parameters: np.ndarray, result: float):
        self.outcomes.append((parameters, result))

    @abstractmethod
    def get_tuning_for(self, parameters: np.ndarray) -> Tuple[Union[int, None], int]:
        """
        Find how to tune for the given set of parameters.

        The first returned value is the index of the model to be tuned from. The index is with respect to
        self.outcomes. If the index is None, then the model should be trained from scratch.

        The second returned value is the number of timesteps to train the model for.
        """
        pass

class InfiniteChainTuningStrategy(TuningStrategy):
    def __init__(self, init_timesteps: int, following_timesteps) -> None:
        super().__init__()
        self.init_timesteps = init_timesteps
        self.following_timesteps = following_timesteps

    def get_tuning_for(self, parameters: np.ndarray) -> Tuple[Union[int, None], int]:
        if len(self.outcomes) == 0:
            return None, self.init_timesteps
        else:
            return len(self.outcomes) - 1, self.following_timesteps


class BestOnlyTuningStrategy(TuningStrategy):
    def __init__(self, initial_timesteps: int, following_timesteps) -> None:
        super().__init__()
        self.initial_timesteps = initial_timesteps
        self.following_timesteps = following_timesteps

    def get_tuning_for(self, parameters: np.ndarray) -> Tuple[Union[int, None], int]:
        if len(self.outcomes) == 0:
            return None, self.initial_timesteps
        else:
            best_index = 0
            for i, (_, reward) in enumerate(self.outcomes):
                if reward >= self.outcomes[best_index][1]:
                    best_index = i
            return best_index, self.following_timesteps
        
def train(args, env_cfg,train_cfg=None):
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root="/srv/share/nkannabiran3/Bayrn_a1_experiments/Bayrn")
    rew_buffer = ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # evaluate and return the mean reward for Bayesian optimization
    # write eval code for target env
    mean_reward = 0
    return mean_reward, ppo_runner, train_cfg
    
def evaluate(true_parameters, runner):
    print("Evaluating in true environment...")
    env = runner.env
    # runner.resume=True
    set_parameters(env.cfg.domain_rand, true_parameters)
    # print(env.cfg.domain_rand.friction_range)
    env.reset_dr_params()
    with torch.inference_mode():
        obs_,_ = env.reset()
    obs = obs_.clone()
    # print(env.device)
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
    bounds = {
            #   "friction":(0.5,1.25), 
            #   "mass":(-1,1), 
              "com_shift":tuple(env_cfg.domain_rand.com_shift_mass_range),
            #   "friction_stddev":(0.01,0.5), 
            #   "mass_stddev":(0.01,0.5), 
              "com_shift_stddev":tuple(env_cfg.domain_rand.com_shift_mass_range_stddev)
              }
    true_parameters = {
                    #    "friction":0.75, 
                    #    "mass":0.3, 
                       "com_shift":args.com_mean,
                    #    "friction_stddev":0.1, 
                    #    "mass_stddev":0.1, 
                       "com_shift_stddev":0.1
              }
    utility = UtilityFunction(kind='ucb',
                        kappa=2.576,
                        xi=0.0,
                        kappa_decay=1,
                        kappa_decay_delay=0)
    optimizer = BayesianOptimization(
        f=None,
        pbounds=bounds,
        random_state=np.random.RandomState(45321),
    )
    logger = JSONLogger(path="Bayrn/"+datetime.now().strftime('%b%d_%H-%M-%S')+".json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    num_threads = 2
    num_iterations = 10
    # processes = []
    model_paths = []
    tuningStrategy = BestOnlyTuningStrategy(5000,1000)
    train_cfg = None
    for i in range(num_iterations):
        # optimizer.maximize(init_points=5, n_iter=2000000)
            next_point = optimizer.suggest(utility)
            # next_point1 = optimizer.suggest(utility)
            print("next point:", next_point)
            model_index, n_steps = tuningStrategy.get_tuning_for(list(next_point.values()))
            if train_cfg is None:
                ## choose a nice start point
                init_point = {
                            # "friction":0.875, 
                            # "mass":0., 
                            "com_shift":np.mean(env_cfg.domain_rand.com_shift_mass_range),
                            # "friction_stddev":0.5, 
                            # "mass_stddev":0.5, 
                            "com_shift_stddev":np.mean(env_cfg.domain_rand.com_shift_mass_range_stddev)
                            }
                next_point = init_point
                set_parameters(env_cfg.domain_rand, init_point)
                print(env_cfg.domain_rand.friction_range)
                _, ppo_runner, train_cfg = train(args,env_cfg)
                target = evaluate(true_parameters, ppo_runner)
            # p = Process(target=train, args=((reward_queue,args),))
            else:
                # target, model_path, train_cfg = train(args,env_cfg, train_cfg)
                iteration = ppo_runner.current_learning_iteration
                print("loading model:", model_index)
                ppo_runner.load(model_paths[model_index], load_optimizer=True)
                ppo_runner.current_learning_iteration = iteration
                # ppo_runner.resume=True
                set_parameters(ppo_runner.env.cfg.domain_rand, next_point)
                ppo_runner.env.reset_dr_params()
                ppo_runner.learn(num_learning_iterations=n_steps, init_at_random_ep_len=True)      
                target = evaluate(true_parameters, ppo_runner)
            model_path = os.path.join(ppo_runner.log_dir, str(len(model_paths))+'_model_{}.pt'.format(ppo_runner.current_learning_iteration))
            ppo_runner.save(model_path)
            model_paths.append(model_path)
            tuningStrategy.add_outcome(np.array(next_point.values()), target)
            
            optimizer.register(params=next_point, target=target)
    
    print({"highest reward": optimizer.max, 
          "model_paths":model_paths})
            


    