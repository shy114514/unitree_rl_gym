import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import time

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    '''
    env = task_class(   cfg=env_cfg,
                    sim_params=sim_params,
                    physics_engine=args.physics_engine,
                    sim_device=args.sim_device,
                    headless=args.headless)
    env_cfg 是GO2RoughCfg类的一个实例
    '''

if __name__ == '__main__':
    # test
    args = get_args()
    train(args)
    # python legged_gym/scripts/train.py --task=go2 --experiment first_test --num_envs 8 --sim_device=cpu --headless
