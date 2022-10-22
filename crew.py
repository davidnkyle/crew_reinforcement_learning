import itertools
from copy import copy, deepcopy

import numpy as np
import random

import utils

non_trump = 'bgpy'
trump = 'z'
SUITS = non_trump + trump
DECK_NON_TRUMP = ['{}{}'.format(color, number) for color in non_trump for number in range(1, 10)]
DECK = DECK_NON_TRUMP + ['{}{}'.format(trump, number) for number in range(1, 5)]
COMMS = ['{}{}{}'.format(color, number, modifier) for modifier in 'loh' for color in non_trump for number in range(1, 10)]
DECK_ARRAY = np.array(DECK)
DECK_SIZE = len(DECK)
ACTIONS = DECK + ['-']

# remove communication for now

##
# Game state vector for 3 players
#
# 0-39 player 1's hand private (40) 0, 0.5, 1
# 40-79 player 2's hand private (40) 0, 0.5, 1
# 80-119 player 3's hand private (40) 0, 0.5, 1
# 120-159 player 1's hand public (40) 0, 0.5, 1
# 160-199 player 2's hand public (40) 0, 0.5, 1
# 200-239 player 3's hand public (40) 0, 0.5, 1
# 240-275 goal cards on the table (36) 0, 1
# 276-311 player 1's goals (36) 0, 1
# 312-347 player 2's goals (36) 0, 1
# 348-383 player 3's goals (36) 0, 1
# 384-386 player leading (3) 0, 1
# 387-426 player 1 card in trick (40) 0, 1
# 427-466 player 2 card in trick (40) 0, 1
# 467-506 player 3 card in trick (40) 0, 1
# 507-509 players turn (3) 0, 1
# 510-550 action (41) 0, 1
#


def action_vec(action: int):
    v = np.zeros(len(ACTIONS))
    v[action] = 1
    return v

def evaluate_trick(trick):
    if len(trick) == 0:
        raise ValueError('No trick to evaluate')
    suit = trick[0][0]
    cards = [c for c in trick if c[0] in (suit, 'z')]
    cards.sort(reverse=True)
    return trick.index(cards[0])


class CrewState():
    def __init__(self, hands, goal_cards):
        self.players = len(hands)
        if (self.players < 3) or (self.players > 5):
            raise ValueError('Only allow between 3 and 5 players')
        self.hands = hands
        self.hands_public = np.full((DECK_SIZE, self.players), 0.5)
        self.hands_private = [np.full((DECK_SIZE, self.players), 0.5) for _ in range(self.players)]
        for pl, hand in enumerate(hands):
            idxs = [idx for idx, c in enumerate(DECK) if c in hand]
            self.hands_private[pl][:,pl] = 0 # this player knows they have no other cards
            self.hands_private[pl][idxs, :] = 0 # this player knows everyone else doesn't have their cards
            self.hands_private[pl][idxs, pl] = 1 # this player knows they have their cards
        self.discard = []
        self.captain = self.player_with(DECK[-1])
        self.player_has(self.captain, DECK_SIZE-1)
        self.leading = self.captain
        self.turn = self.captain
        self.num_goals = len(goal_cards)
        self.select_goals_phase = True
        self.communication_phase = False
        self.coms = []
        self.goal_cards = goal_cards
        self.goals = [[] for _ in range(self.players)]
        self.total_rounds = DECK_SIZE//self.players
        self.rounds_left = DECK_SIZE//self.players
        self.trick = []
        self.game_result = None

    def player_has(self, player, idx):
        self.nobody_has(idx)
        for pl in range(self.players):
            self.hands_private[pl][idx, player] = 1
        self.hands_public[idx, player] = 1

    def nobody_has(self, idx):
        for pl in range(self.players):
            self.hands_private[pl][idx, :] = 0
        self.hands_public[idx, :] = 0

    def player_shortsuited(self, player, suit):
        suit_idx = SUITS.index(suit)
        start = suit_idx*9
        end = min((suit_idx+1)*9, 40)
        for pl in range(self.players):
            self.hands_private[pl][start:end, player] = 0
        self.hands_public[start:end, player] = 0


    @classmethod
    def generate(cls, players: int = 3, num_goals: int = 2):
        deck = copy(DECK)
        np.random.shuffle(deck)
        hands = [deck[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
        no_trump_deck = copy(DECK[:-4])
        np.random.shuffle(no_trump_deck)
        goals_cards = no_trump_deck[0:num_goals]
        state = cls(hands, goals_cards)
        return state

    @classmethod
    def gen_mid_game(cls, players: int = 3, max_goals: int = 1, round: int = 1, cards_in_trick: int = 0):
        deck = copy(DECK)
        some_non_trump_cards = False
        max_tries = 100
        while not some_non_trump_cards and max_tries > 0:
            np.random.shuffle(deck)
            active_deck_size = DECK_SIZE - players*(round-1)
            all_cards = deck[0:active_deck_size]
            non_trump_cards = [c for c in all_cards if 'z' not in c]
            some_non_trump_cards = (len(non_trump_cards) > 0)
            max_tries -= 1

        hands = [deck[(i * active_deck_size) // players:((i + 1) * active_deck_size) // players] for i in range(players)]

        initial_board_state = cls(hands=hands, goal_cards=[])
        initial_board_state.discard = deck[active_deck_size:]
        for c in initial_board_state.discard:
            initial_board_state.nobody_has(ACTIONS.index(c))
        rounds_left = len(deck) // players - round + 1
        initial_board_state.rounds_left = rounds_left
        initial_board_state.leading = np.random.randint(players)
        initial_board_state.turn = initial_board_state.leading
        initial_board_state.select_goals_phase = False
        num_goals = np.random.randint(min(len(non_trump_cards), max_goals)) + 1
        np.random.shuffle(non_trump_cards)
        goals = non_trump_cards[:num_goals]
        max_num_players_with_goals = min(rounds_left, players)
        players_with_goals = np.random.choice(range(players), max_num_players_with_goals, replace=False)
        for goal in goals:
            pl_idx = np.random.randint(max_num_players_with_goals)
            initial_board_state.goals[players_with_goals[pl_idx]].append(goal)

        # simulate a few random turns
        for _ in range(cards_in_trick):
            action = initial_board_state.choose_action(model=None, epsilon=2)
            initial_board_state = initial_board_state.move(ACTIONS[action])
        return initial_board_state

    def reward(self):
        score = 0
        if len(self.trick) == self.players:
            completed_goals = [g for g in self.goals[self.turn] if g in self.trick]
            score += len(completed_goals)
        if self.game_result == 1:
            score += 100
        if self.game_result == 0:
            score -= 100
        return score

    def to_vector(self):
        v = np.zeros(510)

        idx_start = 0

        # hands private
        hands_private = self.hands_private[self.turn]
        for pl in range(self.players):
            v[idx_start: idx_start+DECK_SIZE] = hands_private[:, pl]
            idx_start += DECK_SIZE

        # hands public
        for pl in range(self.players):
            v[idx_start: idx_start+DECK_SIZE] = self.hands_public[:, pl]
            idx_start += DECK_SIZE

        # goals
        # unassigned goals
        for idx, card in enumerate(DECK_NON_TRUMP):
            if card in self.goal_cards:
                v[idx_start + idx] = 1
        idx_start += len(DECK_NON_TRUMP)
        # assigned goals
        for pl in range(self.players):
            for idx, card in enumerate(DECK_NON_TRUMP):
                if card in self.goals[pl]:
                    v[idx_start + idx] = 1
            idx_start += len(DECK_NON_TRUMP)

        # leading
        v[idx_start + self.leading] = 1
        idx_start += self.players

        # cards in trick
        for pl in range(self.players):
            trick_idx = (self.leading + pl) % self.players
            try:
                trick_card = self.trick[trick_idx]
            except(IndexError):
                idx_start += DECK_SIZE
                continue
            card_idx = DECK.index(trick_card)
            v[idx_start + card_idx] = 1
            idx_start += DECK_SIZE

        # turn
        v[idx_start + self.turn] = 1
        idx_start += self.players

        # communication phase
        # if self.communication_phase:
        #     v[idx_start] = 1
        # idx_start += 1

        return v
    #
    # @staticmethod
    # def from_vector(v, captain=None, num_goals=None, coms=None):
    #     # only supports 3 player game
    #
    #     # hands
    #     hands = []
    #     start_idx = 0
    #     section = DECK_SIZE
    #     for _ in range(3):
    #         hands.append(list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1]))
    #         start_idx += section
    #
    #     # goals
    #     # unassigned goals
    #     section = len(DECK_NON_TRUMP)
    #     goal_cards = list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1])
    #     state = CrewState(hands, goal_cards)
    #     # assigned goals
    #     goals = []
    #     for _ in range(3):
    #         goals.append(list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1]))
    #         start_idx += section
    #     state.goals = goals
    #
    #     # leading
    #     section = 3
    #     state.leading = np.where(v[start_idx: start_idx + section]==1)[0]
    #     start_idx += section
    #
    #     # card in trick
    #     section = DECK_SIZE
    #     for idx in range(3):
    #         pl = (state.leading + idx)% 3
    #         card_idxs = np.where(v[start_idx + pl*section: start_idx + (pl+1)*section]==1)
    #         if len(card_idxs) > 0:
    #             state.trick.append(DECK[card_idxs[0]])
    #     start_idx += section*3
    #
    #     # turn
    #     section = 3
    #     state.turn = np.where(v[start_idx: start_idx + section]==1)[0]
    #     start_idx += section
    #
    #     # communication phase
    #     # if v[start_idx] == 1:
    #     #     state.communication_phase = True
    #     # start_idx += 1
    #
    #     # other things
    #     state.captain = captain
    #     state.num_goals = num_goals
    #     cards_in_play = list(itertools.chain(state.hands)) + state.trick
    #     state.discard = [c for c in DECK if c not in cards_in_play]
    #     if len(state.goal_cards) > 0:
    #         state.select_goals_phase = True
    #     state.coms = coms
    #     state.rounds_left = state.total_rounds - len(state.discard)//3
    #
    #     return state

    def done(self):
        if self.game_result is None:
            return 0
        return 1


    def player_with(self, card):
        player = None
        for pl in range(self.players):
            if card in self.hands[pl]:
                player = pl
        return player


    def _determine_game_result(self):
        if self.game_result is None:
            if not self.select_goals_phase:
                if len(self.trick) == self.players:
                    players_with_goals_left = 0
                    for pl, players_goals in enumerate(self.goals):
                        if pl == self.turn:
                            players_goals = [g for g in players_goals if g not in self.trick]
                        if len(players_goals) > 0:
                            players_with_goals_left += 1
                            for c in players_goals:
                                if c in self.trick:
                                    self.game_result = 0
                                    return
                    if players_with_goals_left == 0:
                        self.game_result = 1
                        return
                    if players_with_goals_left > self.rounds_left:
                        self.game_result = 0

    def is_game_over(self):
        return self.game_result is not None

    def move(self, move):
        new = deepcopy(self)
        if move == '-':
            return new
        if new.select_goals_phase:
            new.goals[new.turn].append(move)
            new.goal_cards.remove(move)
            new.turn = (new.turn + 1) % self.players
            if len(new.goal_cards) == 0:
                new.select_goals_phase = False
                # new.communication_phase = True
                new.turn = new.captain
            return new
        # if new.communication_phase:
        #     new.coms.append(move)
        #     new.turn = (new.turn + 1) % self.players
        #     if len(new.coms) == 3:
        #         new.communication_phase = False
        #         new.turn = new.captain
        #     return new
        if len(new.trick) == self.players:
            new.goals[new.turn] = [g for g in new.goals[new.turn] if g not in new.trick]
            new.discard += new.trick
            new.trick = []
            new.leading = new.turn

        # if player is short suited remove that from the list of possibilities
        if len(new.trick) > 0:
            leading_suit = new.trick[0][0]
            if move[0] != leading_suit:
                new.player_shortsuited(new.turn, leading_suit)
        new.trick.append(move)
        new.hands[new.turn].remove(move)
        new.nobody_has(DECK.index(move))
        if len(new.trick) < new.players:
            new.turn = (new.turn + 1) % self.players
            return new
        winner = (evaluate_trick(new.trick) + new.leading) % new.players
        # new.goals[winner] = [g for g in new.goals[winner] if g not in new.trick]
        # new.discard += new.trick # add trick to discard
        # new.trick = []
        new.rounds_left -= 1
        # new.leading = winner
        new.turn = winner
        if len(new.trick) == self.players:
            new._determine_game_result() # update game result variable
        return new

    def is_move_legal(self, move):
        if self.select_goals_phase:
            return move in self.goal_cards
        full_hand = self.hands[self.turn]
        if self.communication_phase:
            if move[0] not in 'bgpy':
                return False
            in_suit = [c for c in full_hand if c[0]==move[0]]
            in_suit.sort()
            if len(in_suit) == 1:
                if move[2] == 'o':
                    return in_suit[0] == move
            elif len(in_suit) > 1:
                if move[2] == 'h':
                    return in_suit[-1] == move
                elif move[2] == 'l':
                    return in_suit[0] == move
            return False
        if not move in full_hand: # you dont have this card in your hand
            return False
        if len(self.trick) not in [0, self.players]:
            leading_suit = self.trick[0][0] # you must follow suit if you can
            if leading_suit in [c[0] for c in full_hand]:
                return move[0] == leading_suit
        return True

    def get_legal_actions(self):
        full_hand = self.hands[self.turn]
        if self.select_goals_phase:
            return copy(self.goal_cards)
        if self.communication_phase:
            allowable = []
            if self.coms[self.turn] is None:
                sort_hand = copy(full_hand)
                sort_hand.sort()
                for suit in 'bgpy':
                    in_suit = [c for c in sort_hand if c[0] == suit]
                    if len(in_suit) == 1:
                        allowable.append(in_suit[0] + 'o')
                    elif len(in_suit) > 1:
                        allowable.append(in_suit[0] + 'l')
                        allowable.append(in_suit[-1] + 'h')
            return allowable
        list_of_actions = [c for c in full_hand if self.is_move_legal(c)]
        if len(list_of_actions) == 0:
            list_of_actions = ['-']
        return list_of_actions

    def choose_action(self, model=None, epsilon=-1):
        allowable_actions = [ACTIONS.index(a) for a in self.get_legal_actions()]
        if model is None:
            epsilon = 2
        if random.random() > epsilon:
            v = self.to_vector()
            best_action = '-'
            best_reward = -np.inf
            for action in allowable_actions:
                x = np.concatenate([v, action_vec(action)])
                y_pred = model.predict(x.reshape((1, 551)))[0]
                if y_pred > best_reward:
                    best_action = action
                    best_reward = y_pred
            return best_action
        else:
            return random.choice(allowable_actions)


    def sort_hands(self):
        for h in self.hands:
            h.sort()

    def print(self):
        # just implemented for 3 players
        self.sort_hands()

        #unassigned goals
        if len(self.goal_cards) > 0:
            print('{:<15}:{:^75s}'.format('UNASSIGNED', '  '.join(self.goal_cards)))
        # assigned goals
        goal_str = [' '.join(g) for g in self.goals]
        print('{:<15}:{:^24s}|{:^24s}|{:^24s}'.format('ASSIGNED', *goal_str))

        # table
        trick_str = ['  ', '  ', '  ']
        for idx, c in enumerate(self.trick):
            trick_str[(self.leading + idx) % self.players] = c
        trick_str[self.leading] += '*'
        trick_str[self.turn] = '[' + trick_str[self.turn] + ']'
        print('{:<15}:{:^24s}|{:^24s}|{:^24s}'.format('TABLE', *trick_str))


if __name__ == '__main__':
    np.random.seed(10000)
    state = CrewState.generate(3, 4)
    state.print()
    state = state.move('y8')
    state = state.move('b1')
    state = state.move('g9')
    state = state.move('g4')
    state.print()
    # state = state.move('p1l')
    # state = state.move('b4o')
    # state = state.move('p5h')
    state = state.move('g7')
    state = state.move('g4')
    state = state.move('g5')
    state.print()
    state = state.move('y8')
    state.print()

    env = CrewState.gen_mid_game(3, 1, 13, 3)

