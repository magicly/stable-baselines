import gym
import logging

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn2 import DQN
from stable_baselines.deepq import MlpPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1)

replay_buffer = ReplayBuffer(50000)

total_timesteps = 10_0000


def get_action(obs, action, obs2, reward, done, info, reset):
    try:
        if action is not None:
            replay_buffer.add(obs, action, reward, obs2, float(done))
        action = model.learn(replay_buffer=replay_buffer,
                             env_info=(obs, reward, done, info, reset),
                             total_timesteps=total_timesteps,
                             reset_num_timesteps=False)
    except:
        logging.exception('got exception....')
    return action


def predict(obs):
    action, _states = model.predict(obs)
    return action
