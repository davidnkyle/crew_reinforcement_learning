from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import time
import pickle

from crew import predict

LEARNING_RATE = 0.5
MAX_DEPTH = 6
MAX_STEPS_FROM_DONE = 4


if __name__=='__main__':
    start = time.time()
    print('reading data')

    data_file_name = 'data_done{}.npy'.format(MAX_STEPS_FROM_DONE)
    my_data = np.load(data_file_name)
    X = my_data[:,:-1]
    y = np.vstack(my_data[:, -1])

    train_idx = int(X.shape[0]*0.6)
    cv_idx = int(X.shape[0]*0.8)

    X_train = X[0:train_idx]
    X_cv = X[train_idx:cv_idx]
    X_test = X[cv_idx:]

    y_train = y[0:train_idx]
    y_cv = y[train_idx:cv_idx]
    y_test = y[cv_idx:]

    model = XGBRegressor(
        eta=LEARNING_RATE,
        max_depth=MAX_DEPTH,
    )

    print('fitting model')
    model.fit(X_train, y_train)

    print('calculating error')
    pred = predict(model, X)

    pred_train = pred[0:train_idx]
    train_error = mean_squared_error(y_train, pred_train)
    print(train_error)

    pred_cv = pred[train_idx:cv_idx]
    cv_error = mean_squared_error(y_cv, pred_cv)
    print(cv_error)

    print('writing model')

    model_file_name = "xgb_done{}.pkl".format(MAX_STEPS_FROM_DONE)
    pickle.dump(model, open(model_file_name, "wb"))

    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
