import time

from crew import CrewState

if __name__ == '__main__':
    start = time.time()

    model = None
    min_turn = 38
    num_episodes = 100
    players=3
    max_goals=1

    for i in range(num_episodes):

        round = min_turn//players
        cards_in_trick =
        env = CrewState.gen_mid_game(players=players, max_goals=max_goals,)