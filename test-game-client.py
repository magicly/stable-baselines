import gym
import requests
import pickle
import logging
import numpy
import time

env = gym.make('CartPole-v1')

url = 'http://localhost:5000/'


def send(command, data):
    payload = pickle.dumps(data)
    response = requests.post(url + command, data=payload)
    if response.status_code != 200:
        logging.error(f"Request failed {response.text}: {data}")
    response.raise_for_status()
    parsed = pickle.loads(response.content)
    return parsed


def get_action(obs, action, obs2, reward, done, info, reset):
    return send(
        'get_action', {
            'obs': obs,
            'action': action,
            'obs2': obs2,
            'reward': reward,
            'done': done,
            'info': info,
            'reset': reset,
        })['action']


def predict(obs):
    return send('predict', {'obs': obs})['action']


# total_timesteps = 10_0000
total_timesteps = 10000
num_timesteps = 0
total_time = time.time()
total_env_time = 0
for i in range(total_timesteps):
    t1 = time.time()
    obs = env.reset()
    t2 = time.time()
    total_env_time += t2 - t1

    reset = True

    action = get_action(obs=obs,
                        action=None,
                        obs2=None,
                        reward=0,
                        done=False,
                        info=None,
                        reset=reset)

    total_over = False
    while True:
        num_timesteps += 1
        if num_timesteps % 1000 == 0:
            print(num_timesteps)
        if num_timesteps >= total_timesteps:
            total_over = True
            break
        # done之后最后一步需要训练，否则reward错误
        reset = False

        t1 = time.time()
        obs2, reward, done, info = env.step(action)
        t2 = time.time()
        total_env_time += t2 - t1

        assert type(obs) == numpy.ndarray
        assert type(action) == numpy.int64
        assert type(obs2) == numpy.ndarray
        assert type(done) == bool

        action = get_action(obs=obs,
                            action=action,
                            obs2=obs2,
                            reward=reward,
                            done=done,
                            info=info,
                            reset=reset)

        obs = obs2

        if done:
            break

    if total_over:
        print(f'i: {i}, model.num: {num_timesteps}')
        break

print(
    f'total_time: {time.time() - total_time}, total_env_time: {total_env_time}'
)

obs = env.reset()
total_rewards = 0
for i in range(1000):
    action = predict(obs)

    obs, rewards, dones, info = env.step(action)
    total_rewards += rewards
    if dones:
        break
    # env.render()

print(f'total_rewards: {total_rewards}')
env.close()