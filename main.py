from collections import namedtuple

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MSE, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import utils
import time


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
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa

    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Compute the loss
    if sigmoid:
        bce = BinaryCrossentropy(from_logits=True)
        loss = bce(y_targets, q_values)
    else:
        loss = MSE(y_targets, q_values)

    return loss


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


if __name__ == '__main__':

    num_actions = 143
    unknown_state_size = 120

    # Q network
    q_network = Sequential([
        Input(unknown_state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # target Q network
    target_q_network = Sequential([
        Input(unknown_state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # I network
    i_network = Sequential([
        Input(unknown_state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    # target I network
    target_i_network = Sequential([
        Input(unknown_state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

    optimizer = Adam()

    # Store experiences as named tuples
    experience = namedtuple("Experience",
                            field_names=["true_state", "known_state", "action", "reward", "next_state", "done"])

    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []

    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        state = env.reset()
        total_points = 0

        for t in range(max_num_timesteps):

            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)

            # Take action A and receive reward R and the next state S'
            next_state, reward, done, _ = env.step(action)

            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))

            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)

                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)

            state = next_state.copy()
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