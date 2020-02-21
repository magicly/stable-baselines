import logging
import pickle
from flask import Flask, request
import time

import test_train

config = {
    'httpHost': 'localhost',
    'httpPort': '5000',
}

total_train_time = 0


def http_server(debug):
    print('=====http serve')

    app = Flask(__name__)

    @app.route('/get_action', methods=['GET', 'POST'])
    def get_action():
        global total_train_time
        infos = request.data
        parsed = pickle.loads(infos)

        obs = parsed['obs']
        action = parsed['action']
        reward = parsed['reward']
        obs2 = parsed['obs2']
        done = parsed['done']
        info = parsed['info']
        reset = parsed['reset']

        t1 = time.time()
        action = test_train.get_action(obs=obs,
                                       action=action,
                                       obs2=obs2,
                                       reward=reward,
                                       done=done,
                                       info=info,
                                       reset=reset)
        t2 = time.time()
        total_train_time += t2 - t1

        return pickle.dumps({'action': action})

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        global total_train_time
        if total_train_time:
            print(f'total_train_time: {total_train_time}')
            total_train_time = 0
        infos = request.data
        parsed = pickle.loads(infos)

        obs = parsed['obs']
        action = test_train.predict(obs)

        return pickle.dumps({'action': action})

    app.run(host=config['httpHost'], port=config['httpPort'], debug=debug)


if __name__ == '__main__':
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format=
    #     '%(asctime)s %(levelname)s [%(funcName)s] %(message)s (%(filename)s:%(lineno)s)',
    #     handlers=[
    #         logging.FileHandler('log.log', mode='w'),
    #         logging.StreamHandler()
    #     ])
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.ERROR)

    http_server(False)
