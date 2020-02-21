import time
import gym

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn2 import DQN
from stable_baselines.deepq import MlpPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1)

replay_buffer = ReplayBuffer(50000)

total_time = time.time()
total_train_time = 0
total_env_time = 0
total_timesteps = 10_000
for i in range(total_timesteps):
    t1 = time.time()
    obs = env.reset()
    t2 = time.time()
    total_env_time += t2 - t1

    reset = True
    t1 = time.time()
    action = model.learn(replay_buffer=replay_buffer,
                         env_info=(obs, 0, False, {}, reset),
                         total_timesteps=total_timesteps,
                         reset_num_timesteps=False)
    t2 = time.time()
    total_train_time += t2 - t1
    total_over = False
    while True:
        if model.num_timesteps >= total_timesteps:
            total_over = True
            break
        # done之后最后一步需要训练，否则reward错误
        reset = False
        t1 = time.time()
        new_obs, rew, done, info = env.step(action)
        t2 = time.time()
        total_env_time += t2 - t1
        # print(type(new_obs), type(rew), type(info), type(action))
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        t1 = time.time()
        action = model.learn(replay_buffer=replay_buffer,
                             env_info=(obs, rew, done, info, reset),
                             total_timesteps=total_timesteps,
                             reset_num_timesteps=False)
        t2 = time.time()
        total_train_time += t2 - t1

        if done:
            break

    if total_over:
        print(f'i: {i}, model.num: {model.num_timesteps}')
        break
print(
    f'total_time: {time.time() - total_time}, total_env_time: {total_env_time}, total_train_time: {total_train_time}'
)

obs = env.reset()
total_rewards = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_rewards += rewards
    if dones:
        break
    # env.render()

print(f'total_rewards: {total_rewards}')
env.close()