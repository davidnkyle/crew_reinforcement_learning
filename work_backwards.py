import time

import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from create_dataset import MAX_TURNS, create_games
from crew import predict
from evaluate_dataset import LEARNING_RATE, MAX_DEPTH

num_episodes = 10000
players = 3
num_goals = 1

if __name__ == '__main__':
    start = time.time()

    model = None

    for steps_from_done in range(MAX_TURNS):
        print('steps from done: ', steps_from_done)
        print('generate games')
        my_data, win_rate = create_games(model, num_episodes, steps_from_done, players, num_goals)
        print('win rate: ', win_rate)

        X = my_data[:, :-1]
        y = np.vstack(my_data[:, -1])

        train_idx = int(X.shape[0] * 0.7)

        X_train = X[0:train_idx]
        X_test = X[train_idx:]

        y_train = y[0:train_idx]
        y_test = y[train_idx:]

        model_stat = XGBRegressor(
            eta=LEARNING_RATE,
            max_depth=MAX_DEPTH,
        )

        print('fitting model_stat')
        model_stat.fit(X_train, y_train)

        print('calculating error')
        pred = predict(model_stat, X)

        pred_train = pred[0:train_idx]
        train_error = mean_squared_error(y_train, pred_train)
        print(train_error)

        pred_cv = pred[train_idx:]
        cv_error = mean_squared_error(y_test, pred_cv)
        print(cv_error)

        model = XGBRegressor(
            eta=LEARNING_RATE,
            max_depth=MAX_DEPTH,
        )

        print('fitting true model')
        model.fit(X, y)

    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")



