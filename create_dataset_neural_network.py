import random
import time
from collections import namedtuple

import numpy as np
from keras.models import clone_model
from keras.saving.save import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

from create_dataset import GAMMA
from crew import CrewState,  ACTIONS

MAX_TURN = 40
EPSILON=0.01

def qsa(network, states, actions):
    q_values_all = network(states)
    q_values = tf.gather_nd(q_values_all, tf.stack(
        [tf.range(q_values_all.shape[0]),  # this just gets the Q values associated with the single action
         tf.cast(actions, tf.int32)], axis=1))
    return q_values


def get_experiences(memory_buffer, episodes):
    batch_ratio = np.sqrt(episodes/len(memory_buffer))
    mini_batch_size = int(batch_ratio*len(memory_buffer))
    experiences = random.sample(memory_buffer, k=mini_batch_size)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),
                                       dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    allowable_actions_of_next_state_vals = tf.convert_to_tensor(
        np.array([e.next_actions for e in experiences if e is not None]), dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals, allowable_actions_of_next_state_vals)


if __name__ == '__main__':
    start = time.time()
    num_episodes = 1000
    players = 3
    max_goals = 1

    for min_turn in reversed(range(MAX_TURN)):
        print(min_turn)

        if min_turn == MAX_TURN-1:

            # target Q network
            target_q_network = Sequential([
                Input(CrewState.state_size),
                Dense(500, activation='relu'),
                Dense(200, activation='relu'),
                Dense(100, activation='relu'),
                Dense(50, activation='relu'),
                Dense(len(ACTIONS), activation='linear')
            ])
        else:
            # network_read_path = 'model{}.h5'.format(min_turn+1)
            # target_q_network = load_model(network_read_path)
            target_q_network = q_network

        ##
        # Generate Data
        #
        print('generating games')

        experience = namedtuple("Experience", field_names=[
            "state", "action", "reward", "next_state", "done", "next_actions"])
        experiences = []

        for i in range(num_episodes):

            start_turn = np.random.randint(min_turn, MAX_TURN)
            env = CrewState.gen_mid_game(players=players, max_goals=max_goals, turn=start_turn)
            state_vec = env.to_vector()
            allowable_actions = [ACTIONS.index(a) for a in env.get_legal_actions()]
            next_actions_binary = np.zeros(len(ACTIONS))
            next_actions_binary[allowable_actions] = 1

            for turn in range(min_turn, MAX_TURN):
                reward = env.reward()
                done = env.done()
                if np.random.random() > EPSILON:
                    q_values_all = target_q_network(np.expand_dims(state_vec, axis=0))
                    action = np.argmax(q_values_all + (next_actions_binary - 1) * 10000)
                else:
                    action = np.random.choice(allowable_actions)

                env.move(ACTIONS[action])
                next_state_vec = env.to_vector()
                allowable_actions = [ACTIONS.index(a) for a in env.get_legal_actions()]
                next_actions_binary = np.zeros(len(ACTIONS))
                next_actions_binary[allowable_actions] = 1
                experiences.append(experience(state=state_vec, action=action, reward=reward,
                                              next_state=next_state_vec, done=done, next_actions=next_actions_binary))
                state_vec = next_state_vec
                if done:
                    break

        states, actions, rewards, next_states, done_vals, next_allowable_actions = get_experiences(experiences, num_episodes)
        num_rows = states.shape[0]

        # define target output
        max_qsa = tf.reduce_max((target_q_network(next_states) + (next_allowable_actions - 1) * 10000), axis=-1)
        y_targets = rewards + (1 - done_vals) * GAMMA * max_qsa

        # print('saving data')
        # table = np.hstack([states, actions.numpy().reshape((num_rows, 1)), y_targets.numpy().reshape((num_rows, 1))])
        # np.save('data{}.npy'.format(min_turn), table)

        ##
        # train model
        #
        print('training evaluation model')

        mse = tf.keras.losses.MSE

        def train_model(network, size):
            opt = tf.keras.optimizers.Adam(0.001)

            # @tf.function
            def _train_model():
                batch = 100
                for i in range(10):
                    for j in range(0, size, batch):
                        with tf.GradientTape() as tape:
                            end_idx = min(size, j+batch)
                            q_values = qsa(network, states[j:end_idx], actions[j:end_idx])
                            loss = mse(y_targets[j:end_idx], q_values)
                        grads = tape.gradient(loss, network.trainable_variables)
                        processed_grads = [g for g in grads]
                        grads_and_vars = zip(processed_grads, network.trainable_variables)
                        opt.apply_gradients(grads_and_vars)
            _train_model()

        test_idx = int(num_rows * 0.7)
        eval_q_network = clone_model(target_q_network)
        train_model(eval_q_network, test_idx)

        q_values_eval = qsa(eval_q_network, states, actions)

        train_mse = mse(y_targets[:test_idx], q_values_eval[:test_idx]).numpy()
        test_mse = mse(y_targets[test_idx:], q_values_eval[test_idx:]).numpy()
        # print(train_mse)
        # print(test_mse)
        print(np.sqrt(train_mse))
        print(np.sqrt(test_mse))

        q_network = eval_q_network
        # print('train real model')
        #
        # q_network = clone_model(target_q_network)
        # train_model(q_network, num_rows)

        # print('save model')
        # network_write_path = 'model{}.h5'.format(min_turn)
        # q_network.save(network_write_path, save_format='h5')

    print('Win rate')
    wins = 0
    games = 1000
    for _ in range(games):
        game = CrewState.generate(players, num_goals=max_goals)
        game.random_move()
        while game.game_result is None:
            state_vec = game.to_vector()
            allowable_actions = [ACTIONS.index(a) for a in game.get_legal_actions()]
            next_actions_binary = np.zeros(len(ACTIONS))
            next_actions_binary[allowable_actions] = 1
            q_values_all = q_network(np.expand_dims(state_vec, axis=0))
            action = np.argmax(q_values_all + (next_actions_binary - 1) * 10000)
            game.move(ACTIONS[action])
        wins += game.game_result
    print(wins/games)


    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")









