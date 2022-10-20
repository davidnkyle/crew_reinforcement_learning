import time
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
import utils

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

from collections import namedtuple
from copy import copy

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MSE, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import utils
import time

from crew import CrewState, DECK_SIZE, ACTIONS


# dont use communication for the time being
# first create a network that uses a simple implication function
# eventually this will have to be trained as well


def compute_loss(experiences, gamma, q_network, target_q_network, sigmoid=False):
    """
    Calculates the loss for the Q networks

    Args:
      experiences: (tuple) tuple of ["true_state", "known_state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    # these are arrays with 10000 or so values inside of them
    states, actions, rewards, next_states, done_vals, next_allowable_actions = experiences

    # Compute max Q^(s,a)
    # use get legal actions
    max_qsa = tf.reduce_max((target_q_network(next_states) + (next_allowable_actions-1)*10000), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa

    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), # this just gets the Q values associated with the single action
                                                tf.cast(actions, tf.int32)], axis=1)) # this could be a useful function for the above

    # Compute the loss
    # if sigmoid:
    #     bce = BinaryCrossentropy(from_logits=True)
    #     loss = bce(y_targets, q_values)
    # else:
    loss = MSE(y_targets, q_values)

    return loss


if __name__ == '__main__':

    num_actions = 41
    state_size = 510
    # unknown_state_size = 120
    layer_1_size = 512
    layer_2_size = 256

    # Q network
    q_network = Sequential([
        Input(state_size),
        Dense(layer_1_size, activation='relu'),
        Dense(layer_2_size, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # target Q network
    target_q_network = Sequential([
        Input(state_size),
        Dense(layer_1_size, activation='relu'),
        Dense(layer_2_size, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # # I network
    # i_network = Sequential([
    #     Input(unknown_state_size),
    #     Dense(layer_1_size, activation='relu'),
    #     Dense(layer_2_size, activation='relu'),
    #     Dense(num_actions, activation='linear')
    # ])
    #
    # # target I network
    # target_i_network = Sequential([
    #     Input(unknown_state_size),
    #     Dense(layer_1_size, activation='relu'),
    #     Dense(layer_2_size, activation='relu'),
    #     Dense(num_actions, activation='linear')
    # ])

    optimizer = Adam()


    @tf.function
    def agent_learn(experiences, gamma):
        """
        Updates the weights of the Q networks.

        Args:
          experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
          gamma: (float) The discount factor.

        """

        # Calculate the loss
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma, q_network, target_q_network)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, q_network.trainable_variables)

        # Update the weights of the q_network.
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

        # update the weights of target q_network
        utils.update_target_network(q_network, target_q_network)


    # Store experiences as named tuples
    experience = namedtuple("Experience",
                            field_names=["state", # 511
                                         "action", # single value 0-39
                                         "reward", # reward for current state: float
                                         "next_state", # 511
                                         "done", # if this current state is done: 0-1
                                         "allowable_actions_of_next_state", # array of indices of allowable actions
                                         ])

    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000
    MEMORY_SIZE = 100_000  # size of memory buffer
    GAMMA = 0.995  # discount factor
    ALPHA = 1e-3  # learning rate
    NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

    total_point_history = []

    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        np.random.seed(i)
        players = 3
        num_goals = 1
        env = CrewState.generate(players, num_goals)
        v = env.to_vector()
        prob_vector = np.full(120, 1.0 / players)
        prob_vector_priv = np.full(120, 1.0 / (players-1))
        for idx in range(players):
            val = 0
            if idx == env.captain:
                val = 1
            prob_vector[(idx + 1) * DECK_SIZE - 1] = val
            prob_vector_priv[(idx + 1) * DECK_SIZE - 1] = val

        true_hands = v[0:120] #120
        public_hands = prob_vector.copy() #120
        private_hands = np.array([prob_vector_priv, prob_vector_priv, prob_vector_priv]) #120x3
        for pl in range(players):
            true_hand = true_hands[pl*DECK_SIZE: (pl+1)*DECK_SIZE]
            private_hands[pl, pl*DECK_SIZE: (pl+1)*DECK_SIZE] = true_hand
            mask = np.concatenate([true_hand, true_hand, true_hand])
            private_hands[pl, np.where(mask==1)] = 0
            private_hands[pl, pl * DECK_SIZE: (pl + 1) * DECK_SIZE] = true_hand
        state_other = v[120:]  # 271
        allowable_actions = [ACTIONS.index(a) for a in env.get_legal_actions()]

        total_points = 0

        for t in range(max_num_timesteps):

            # From the current state S choose an action A using an ε-greedy policy
            this_state = np.concatenate([private_hands[env.turn, :], public_hands, state_other]) # construct inputs to q network
            state_qn = np.expand_dims(this_state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            done = env.done()
            action = utils.get_action(q_values, allowable_actions, epsilon)
            new_env = env.move(ACTIONS[action])

            # Take action A and receive reward R and the next state S'
            reward = env.reward()

            # maybe instead of using a huge named tuple you could construct next state take the form of the next
            # action to be used in the Q function...
            v = new_env.to_vector()
            next_true_hands = v[0:120]  # 120
            next_public_hands = utils.imply(public_hands, action)  # 120
            priv_list = []
            for pl in range(players):
                priv_list.append(utils.imply(private_hands[pl, :], action))
            next_private_hands = np.array(priv_list)  # 120x3
            next_state_other = v[120:]  # 271
            next_allowable_actions = [ACTIONS.index(a) for a in new_env.get_legal_actions()]
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            next_state = np.concatenate([next_private_hands[new_env.turn, :], next_public_hands, next_state_other])
            next_actions_binary = np.zeros(num_actions)
            next_actions_binary[next_allowable_actions] = 1
            memory_buffer.append(experience(this_state, action, reward, next_state, done, next_actions_binary))

            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)

                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)

            env = new_env
            true_hands = next_true_hands
            public_hands = next_public_hands  # 120
            private_hands = next_private_hands  # 120x3
            state_other = next_state_other # 271
            allowable_actions = next_allowable_actions
            total_points += reward

            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])

        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i + 1} episodes!")
            q_network.save('lunar_lander_model.h5')
            break

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")