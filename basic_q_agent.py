"""
File: basic_q_agent.py 
Last modified: 3/5/22 by Peyton
Basic Q-learning with a reduced state space.

Author: Michelle Fu, Peyton Lee
"""
import sys
import numpy as np
import pandas
from math import inf
from tqdm import tqdm

from classes import Card, Trick, Player

NUM_TRAINING_PASSES = 5

DEBUGGING = False

# Adapted from project 2.
class QLearner():
    def __init__(self, state_space, action_space, discount, alpha):
        self.state_space = []
        self.action_space = []
        self.discount = None
        self.Q = {}
        self.alpha = None
        self.trace = {}
        
    # Updates Q based on observation
    def update(self, s, a, r, sp):
        self.trace[s, a] += 1
        self.Q[s, a] += self.alpha * (r + self.discount * np.max(self.Q[sp]) - self.Q[s, a])
        
    # Interpolates values of Q function to unvisited state-action pairs based on 1 nearest neighbor.
    def interpolate_Q(self):
        for s in self.state_space:
            for a in self.action_space:
                if self.trace[s, a] == 0:
                    seen, queue = set(), []
                    closest = self.bfs_Q(s, a, seen, queue)
                    self.Q[s, a] = closest
                    self.trace[s, a] += 1
           
    # Helper for finding the nearest neighbor in self.Q.         
    def bfs_Q(self, s, a, seen, queue):
        seen.add((s, a))
        queue.append((s, a))
        while queue:
            curr = queue.pop(0)
            if self.trace[curr[0], curr[1]] != 0:
                return self.Q[curr[0], curr[1]]
            
            neighbors = [(curr[0]-1, curr[1]), (curr[0]+1, curr[1]), (curr[0], curr[1]-1), (curr[0], curr[1]+1)]
            for neighbor in neighbors:
                try:
                    if self.Q[neighbor[0], neighbor[1]] and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
                except:
                    pass
                
    # Generates a policy based on self.Q. Assumes an existing Q for every state-action pair.
    def extract_policy(self):
        for s in self.state_space:
            if np.sum(self.trace[s]) > 0:
                max_a = self.action_space[0]
                max_Q = -inf
                for a in self.action_space:
                    if self.trace[s, a] > 0 and self.Q[s, a] > max_Q:
                        max_a = a
                        max_Q = self.Q[s, a]
                self.policy[s] = max_a    
        
class BasicQAgent(Player):
    def __init__(self, pos: int):
        super().__init__(pos)
        state_space = [i for i in range(4 * 13 ** 4)]
        action_space = [i for i in range(52)]
        discount = 1
        alpha = 0.1
        self.q_learner = QLearner(state_space, action_space, discount, alpha)
        self.q_learner.Q = np.zeros((len(state_space), len(action_space)))
        self.q_learner.trace = np.zeros((len(state_space), len(action_space)))
        
    # Generates the corresponding state integer based on the state of the game.
    # s = num_spades_played * 13^0 + num_diamonds_played * 13^1 + num_hearts_played * 13^2
    #     num_clubs_played * 13^3 + hand_var * 13^4
    # hand_var = [0: first to play in trick, 
    #             1: doesn't have trick suit,
    #             2: has suit but all cards are lower than the current winning card, 
    #             3: has suit and a potential winning card]
    def get_state_int(self, trick: Trick, tricks: list[Trick], legal_moves: list[Card]):
        if not trick.suit:
            hand_var = 0
        elif any(c.suit != trick.suit for c in legal_moves):
            hand_var = 1
        else:
            winning_card = None
            for card in trick.cards.values():
                if card.suit == trick.suit and (not winning_card or card.rank > winning_card.rank):
                    winning_card = card 
            if all(c.rank < winning_card.rank for c in legal_moves):
                hand_var = 2
            else:
                hand_var = 3
        suit_counts = [0, 0, 0, 0]
        for t in tricks:
            for card in t:
                suit_counts[card.id // 13] += 1
        return suit_counts[0] + suit_counts[1] * 13 + suit_counts[2] * 13 ** 2 + suit_counts[3] * 13 ** 3 + hand_var * 13 ** 4
    
    # Employs batch Q-learning with actions constrained by rules
    def take_turn(self, trick: Trick, tricks: list[Trick], hearts_broken: bool) -> Card:
        legal_moves = self.get_legal_moves(trick, hearts_broken)
        state = self.get_state_int(trick, tricks, legal_moves)
        optimal_card, optimal_Q = None, -inf
        for card in legal_moves:
            if self.q_learner.Q[state, card.id] > optimal_Q:
                optimal_card, optimal_Q = card, self.q_learner.Q[state, card.id]
        if DEBUGGING:          
            hand = sorted([card.name for card in self.hand], key=lambda x: x[-1])
            choices = sorted([card.name for card in legal_moves], key=lambda x: x[-1])
            print("Hand: ", hand)
            print("From ", choices, ", picked ", optimal_card.name)
        self.hand.remove(optimal_card)
        return optimal_card
    
    # Trains model on a batch of simulated games
    def batch_q_learn(self, filename: str):
        input_df = pandas.read_csv(filename)
        for _ in range(NUM_TRAINING_PASSES):
            for idx, data in tqdm(input_df.iterrows()):
                self.q_learner.update(data["s"], data["a"], data["r"], data["sp"])
        self.q_learner.interpolate_Q()
        
        
# Simulates games and saves transition/reward data in a csv
def generate_random_data(num_games: int) -> str:
    pass