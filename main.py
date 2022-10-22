import random

from xgboost import XGBRegressor
from crew import CrewState, ACTIONS, action_vec
import time
from collections import deque, namedtuple
import numpy as np

if __name__ == '__main__':
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000
    GAMMA = 0.995  # discount factor
    NUM_STEPS_FOR_UPDATE = 1000 # perform a learning update every C time steps
    MINIBATCH_SIZE = 300  # mini-batch size
    MEMORY_SIZE = 100_000
    E_DECAY = 0.999  # ε decay rate for ε-greedy policy
    E_MIN = 0.01  # minimum ε value for ε-greedy policy

    total_point_history = []
    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy
    players = 3
    num_goals = 1
    round = 13
    cards_in_trick = 1
    if round == 13 and cards_in_trick == 3:
        cumulative_x = np.empty((0, 551))
        cumulative_y = np.empty((0, 1))
    else:
        my_data = np.genfromtxt(r'dataset_{}pl_goals{}_round{}_trick{}.csv'.format(players,num_goals, round, cards_in_trick+1), delimiter=',')
        cumulative_x = my_data[:,:-1]
        cumulative_y = np.vstack(my_data[:, -1])

    memory_buffer = []
    # XGBRegressor model
    model = None


    def get_experiences(memory_buffer):
        experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        done_vals = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)
        allowable_actions_of_next_state_vals = np.array([e.next_allowable_actions for e in experiences if e is not None])
        return (states, actions, rewards, next_states, done_vals, allowable_actions_of_next_state_vals)

    def train_model(experiences, model, gamma):
        states, actions, rewards, next_states, done_vals, next_allowable_actions = experiences
        max_qsa = []
        for ns, naa in zip(next_states, next_allowable_actions):
            best_reward = -np.inf
            for action in np.where(naa==1):
                x = np.concatenate([ns, action_vec(action)])
                if model is None:
                    y_pred = 0
                else:
                    y_pred = model.predict(x.reshape((1, 551)))[0]
                if y_pred > best_reward:
                    best_reward = y_pred
            max_qsa.append(best_reward)
        y_targets = rewards + (1 - done_vals) * gamma * np.array(max_qsa)
        new_x = np.hstack([states, actions])
        global cumulative_x
        global cumulative_y
        cumulative_x = np.vstack([cumulative_x, new_x])
        cumulative_y = np.vstack([cumulative_y, np.vstack(y_targets)])
        new_model = XGBRegressor()
        new_model.fit(cumulative_x, cumulative_y)
        return new_model

    # Store experiences as named tuples
    experience = namedtuple("Experience",
                            field_names=["state",  # 550
                                         "action",  # single value 0-40
                                         "reward",  # reward for current state: float
                                         "next_state",  # 550
                                         "done",  # if this current state is done: 0, 1
                                         "next_allowable_actions"
                                         ])

    j = 0
    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        np.random.seed(i)

        env = CrewState.gen_mid_game(players, num_goals, round=round, cards_in_trick=cards_in_trick)

        total_points = 0

        for t in range(max_num_timesteps):
            this_state = env.to_vector()
            done = env.done()
            action = env.choose_action(model=model, epsilon=epsilon)
            new_env = env.move(ACTIONS[action])
            reward = env.reward()
            next_state = new_env.to_vector()
            next_allowable_actions = [ACTIONS.index(a) for a in new_env.get_legal_actions()]
            next_actions_binary = np.zeros(len(ACTIONS))
            next_actions_binary[next_allowable_actions] = 1
            memory_buffer.append(experience(this_state, action_vec(action), reward, next_state, done, next_actions_binary))
            j += 1
            if j > NUM_STEPS_FOR_UPDATE:
                j = 0
                experiences = get_experiences(memory_buffer)
                new_model = train_model(experiences, model, GAMMA)
                model = new_model
                memory_buffer = []
            env = new_env
            total_points += reward

            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])

        # Update the ε value
        epsilon = max(E_MIN, E_DECAY * epsilon)

        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f} epsilon: {epsilon:.3f} data size: {cumulative_y.shape[0]}")

    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
    export = np.hstack([cumulative_x, cumulative_y])
    np.savetxt(r'dataset_{}pl_goals{}_round{}_trick{}.csv'.format(players,num_goals, round, cards_in_trick), export, delimiter=",")