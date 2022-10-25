import time

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

from create_dataset import GAMMA
from crew import CrewState,  ACTIONS

MAX_TURN = 40
EPSILON=0.01

if __name__ == '__main__':
    start = time.time()

    num_actions = len(ACTIONS)
    state_size = 510
    layer_1_size = 2000
    layer_2_size = 1000
    layer_3_size = 500

    # Q network
    q_network = Sequential([
        Input(state_size),
        Dense(layer_1_size, activation='relu'),
        Dense(layer_2_size, activation='relu'),
        Dense(layer_3_size, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # target Q network
    target_q_network = Sequential([
        Input(state_size),
        Dense(layer_1_size, activation='relu'),
        Dense(layer_2_size, activation='relu'),
        Dense(layer_3_size, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # create data
    print('generating games')

    min_turn = 39
    num_episodes = 10000
    players=3
    max_goals=1

    this_state_list = []
    this_state_reward_list = []
    this_state_done_list = []
    this_action_list = []
    next_state_list = []
    next_allowable_action_list = []

    for i in range(num_episodes):

        start_turn = np.random.randint(min_turn, MAX_TURN)
        env = CrewState.gen_mid_game(players=players, max_goals=max_goals, turn=start_turn)
        allowable_actions = [ACTIONS.index(a) for a in env.get_legal_actions()]
        state_vec = env.to_vector()

        for turn in range(min_turn, MAX_TURN):
            this_state_list.append(state_vec)
            reward = env.reward()
            this_state_reward_list.append(reward)
            done = env.done()
            this_state_done_list.append(done)
            # if np.random.random() > EPSILON:
            #     pass
            #     # add some stuff here later
            # else:
            action = np.random.choice(allowable_actions)

            this_action_list.append(action)
            env.move(ACTIONS[action])
            state_vec = env.to_vector()
            next_state_list.append(state_vec)
            allowable_actions = [ACTIONS.index(a) for a in env.get_legal_actions()]
            next_actions_binary = np.zeros(len(ACTIONS))
            next_actions_binary[allowable_actions] = 1
            next_allowable_action_list.append(next_actions_binary)

    states = np.array(this_state_list)
    rewards = np.array(this_state_reward_list)
    done_vals = np.array(this_state_done_list)
    actions = np.array(this_action_list)
    next_states = np.array(next_state_list)
    next_allowable_actions = np.array(next_allowable_action_list)

    # train model
    print('training model')

    opt = tf.keras.optimizers.Adam(0.001)
    mse = tf.keras.losses.MSE

    # define target output
    max_qsa = tf.reduce_max((target_q_network(next_states) + (next_allowable_actions - 1) * 10000), axis=-1)
    y_targets = rewards + (1 - done_vals) * GAMMA * max_qsa

    SIZE = int(states.shape[0]*0.7)
    batch = 100
    for i in range(10):
        for j in range(0, SIZE, batch):
            with tf.GradientTape() as tape:
                end_idx = min(SIZE, j+batch)
                q_values = q_network(states[j:end_idx])
                q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), # this just gets the Q values associated with the single action
                                                            tf.cast(actions[j:end_idx], tf.int32)], axis=1)) # this could be a useful function for the above
                loss = mse(y_targets[j:end_idx], q_values)
            grads = tape.gradient(loss, q_network.trainable_variables)
            processed_grads = [g for g in grads]
            grads_and_vars = zip(processed_grads, q_network.trainable_variables)
            opt.apply_gradients(grads_and_vars)

    print('evaluation')

    q_values_eval = q_network(states)
    q_values_eval = tf.gather_nd(q_values_eval, tf.stack(
        [tf.range(q_values_eval.shape[0]),  # this just gets the Q values associated with the single action
         tf.cast(actions, tf.int32)], axis=1))

    train_mse = mse(y_targets[:SIZE], q_values_eval[:SIZE]).numpy()
    test_mse = mse(y_targets[SIZE:], q_values_eval[SIZE:]).numpy()
    print(train_mse)
    print(test_mse)
    print(np.sqrt(train_mse))
    print(np.sqrt(test_mse))

    # target Q network
    # c_network = Sequential([
    #     Input(state_size),
    #     Dense(layer_1_size, activation='relu'),
    #     Dense(layer_2_size, activation='relu'),
    #     Dense(layer_3_size, activation='relu'),
    #     Dense(1, activation='linear')
    # ])
    #
    # c_network.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MSE)
    # c_network.fit(x=states[:SIZE], y=y_targets[:SIZE])
    #
    # c_eval = c_network(states)
    # print(mse(y_targets[:SIZE], tf.reshape(c_eval[:SIZE], SIZE)).numpy())
    # print(mse(y_targets[SIZE:], tf.reshape(c_eval[SIZE:], states.shape[0]-SIZE)).numpy())









