import gym

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.deepq import MlpPolicy

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
print(f'model.num: {model.num_timesteps}')

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