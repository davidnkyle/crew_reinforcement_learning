import pickle
import time
import numpy as np

from crew import CrewState, ACTIONS, action_vec

num_episodes = 10000
GAMMA = 0.995  # discount factor
EPSILON = .01  # minimum ε value for ε-greedy policy
MAX_STEPS_FROM_DONE = 5
MAX_TURNS = 40
players=3
num_goals=1

def create_games(model, num_episodes, max_steps_from_done, players=3, num_goals=1):
    data_list = []
    wins = 0

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        # np.random.seed(i)

        env = CrewState.generate(players, num_goals=num_goals)

        envs = []
        actions = []

        for turn in range(MAX_TURNS):
            prev_env = env
            if turn < MAX_TURNS - max_steps_from_done:
                e = 2
            else:
                e = EPSILON
            action = prev_env.choose_action(model=model, epsilon=e)
            envs.append(prev_env)
            actions.append(action)
            env = prev_env.move(ACTIONS[action])
            done = prev_env.done()
            if done:
                break

        wins += env.game_result
        steps_backward = np.random.randint(min(max_steps_from_done + 1, len(envs)))
        this_env = envs[-steps_backward - 1]
        state = this_env.to_vector()
        ac_vec = action_vec(actions[-steps_backward - 1])
        maxqsa = 0
        if steps_backward > 0:
            next_env = envs[-steps_backward]
            maxqsa = next_env.max_qsa(model)
        y_pred = this_env.reward() + GAMMA * maxqsa
        new_row = np.concatenate([state, ac_vec, np.array([y_pred])])
        data_list.append(new_row)

    table = np.array(data_list)
    return table, wins/num_episodes


if __name__ == '__main__':
    start = time.time()

    if MAX_STEPS_FROM_DONE==0:
        model = None
    else:
        file_name = 'xgb_done{}.pkl'.format(MAX_STEPS_FROM_DONE-1)
        model = pickle.load(open(file_name, "rb"))

    table, win_rate = create_games(model, num_episodes, MAX_STEPS_FROM_DONE, players, num_goals)

    np.save('data_done{}.npy'.format(MAX_STEPS_FROM_DONE), table)

    print('win rate: {}'.format(win_rate))
    tot_time = time.time() - start
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")




