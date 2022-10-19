import itertools
from copy import copy, deepcopy

import numpy as np

non_trump = 'bgpy'
trump = 'z'
SUITS = non_trump + trump
DECK_NON_TRUMP = ['{}{}'.format(color, number) for color in non_trump for number in range(1, 10)]
DECK = DECK_NON_TRUMP + ['{}{}'.format(trump, number) for number in range(1, 5)]
COMMS = ['{}{}{}'.format(color, number, modifier) for modifier in 'loh' for color in non_trump for number in range(1, 10)]
DECK_ARRAY = np.array(DECK)
DECK_SIZE = len(DECK)
ACTIONS = DECK + COMMS

##
# Game state vector for 3 players
#
# 0-39 player 1's hand
# 40-79 player 2's hand
# 80-119 player 3's hand
# 120-155 goal cards on the table
# 156-191 player 1's goals
# 192-227 player 2's goals
# 228-263 player 3's goals
# 264-266 players turn
# 267-306 player 1 card in trick
# 307-346 player 2 card in trick
# 347-386 player 3 card in trick
# 387 communication phase
#

##
# action vector
#
# 0-35 play card or select goal
# 36-71 communicate lowest
# 72-107 communicate only
# 108-143 communicate highest
#

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
        self.discard = []
        self.captain = self.player_with(DECK[-1])
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

    @staticmethod
    def generate(players: int = 3, num_goals: int = 2):
        deck = copy(DECK)
        np.random.shuffle(deck)
        hands = [DECK[(i * DECK_SIZE) // players:((i + 1) * DECK_SIZE) // players] for i in range(players)]
        no_trump_deck = copy(DECK[:-4])
        np.random.shuffle(no_trump_deck)
        goals_cards = no_trump_deck[0:num_goals]
        state = CrewState(hands, goals_cards)
        return state

    def to_vector(self):
        v = np.zeros(363)

        # hands
        idx_start = 0
        for pl in range(self.players):
            for idx, card in enumerate(DECK):
                if card in self.hands[pl]:
                    v[idx_start + idx] = 1
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

        # turn
        v[idx_start + self.turn] = 1
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

        # communication phase
        if self.communication_phase:
            v[idx_start] = 1
        idx_start += 1

        return v

    @staticmethod
    def from_vector(v, captain=None, num_goals=None, coms=None):
        # only supports 3 player game

        # hands
        hands = []
        start_idx = 0
        section = DECK_SIZE
        for _ in range(3):
            hands.append(list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1]))
            start_idx += section

        # goals
        # unassigned goals
        section = len(DECK_NON_TRUMP)
        goal_cards = list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1])
        state = CrewState(hands, goal_cards)
        # assigned goals
        goals = []
        for _ in range(3):
            goals.append(list(DECK_ARRAY[np.where(v[start_idx: start_idx+section])==1]))
            start_idx += section
        state.goals = goals

        # turn
        section = 3
        state.turn = np.where(v[start_idx: start_idx + section]==1)[0]
        start_idx += section

        # card in trick
        section = DECK_SIZE
        plays_in_trick_so_far = v[start_idx: start_idx + section*3].sum()
        state.leading = (state.turn - plays_in_trick_so_far) % 3
        for idx in range(3-1):
            pl = (state.leading + idx)% 3
            card_idxs = np.where(v[start_idx + pl*section: start_idx + (pl+1)*section]==1)
            if len(card_idxs) > 0:
                state.trick.append(DECK[card_idxs[0]])
        start_idx += section*3

        # communication phase
        if v[start_idx] == 1:
            state.communication_phase = True
        start_idx += 1

        # other things
        state.captain = captain
        state.num_goals = num_goals
        cards_in_play = list(itertools.chain(state.hands)) + state.trick
        state.discard = [c for c in DECK if c not in cards_in_play]
        if len(state.goal_cards) > 0:
            state.select_goals_phase = True
        state.coms = coms
        state.rounds_left = state.total_rounds - len(state.discard)//3

        return state


    def player_with(self, card):
        player = None
        for pl in range(self.players):
            if card in self.hands[pl]:
                player = pl
        return player

    @property
    def goals_remaining(self):
        return sum([len(goal) for goal in self.goals])

    @property
    def game_result(self):
        if not self.select_goals_phase:
            players_with_goals_left = 0
            for pl in self.goals:
                if len(pl) > 0:
                    players_with_goals_left += 1
                    for c in pl:
                        if c in self.discard:
                            return 0 # if the goal is still active and in the discard pile, there is no way to win
            if players_with_goals_left == 0:
                return 1
            if players_with_goals_left > self.rounds_left:
                return 0
        return None

    def is_game_over(self):
        return self.game_result is not None

    def move(self, move):
        new = deepcopy(self)
        if new.select_goals_phase:
            new.goals[new.turn].append(move)
            new.goal_cards.remove(move)
            new.turn = (new.turn + 1) % self.players
            if len(new.goal_cards) == 0:
                new.select_goals_phase = False
                new.communication_phase = True
                new.turn = 0
            return new
        if new.communication_phase:
            new.coms.append(move)
            new.turn = (new.turn + 1) % self.players
            if len(new.coms) == 3:
                new.communication_phase = False
                new.turn = new.captain
            return new
        new.trick.append(move)
        new.hands[new.turn].remove(move)
        if len(new.trick) < new.players:
            new.turn = (new.turn + 1) % self.players
            return new
        winner = (evaluate_trick(new.trick) + new.leading) % new.players
        new.goals[winner] = [g for g in new.goals[winner] if g not in new.trick]
        new.discard += new.trick # add trick to discard
        new.trick = []
        new.rounds_left -= 1
        new.leading = winner
        new.turn = winner
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
        if len(self.trick) > 0:
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
        return [c for c in full_hand if self.is_move_legal(c)]

    def get_all_actions(self):
        if self.select_goals_phase:
            return copy(self.goal_cards)
        if self.communication_phase:
            if self.coms[self.turn] is None:
                return copy(COMMS)
            return [None]
        return copy(DECK)


