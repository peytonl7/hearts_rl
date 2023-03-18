"""
File: basic_q_agent.py 
Last modified: 3/5/23 by Peyton
Basic Q-learning with a reduced state space.

Author: Michelle Fu, Peyton Lee
"""
import numpy as np
import pandas
from math import inf
from tqdm import tqdm
from matplotlib import pyplot as plt

from classes import Card, Trick, Player, StateRecord
from baseline_agents import BaselineAgent, GreedyBaseline
from play import play
from evaluate import evaluate

NUM_TRAINING_PASSES = 1
NUM_TRAINING_GAMES = 10000

DEBUGGING = True

# Adapted from project 2.
class QLearner():
    def __init__(self, state_space, action_space, discount, alpha):
        self.state_space = state_space
        self.action_space = action_space
        self.discount = discount
        self.Q = np.zeros((len(state_space), len(action_space)))
        self.alpha = alpha
        self.trace = np.zeros((len(state_space), len(action_space)))
        
    # Updates Q based on observation
    def update(self, s, a, r, sp):
        self.trace[s, a] += 1
        self.Q[s, a] += self.alpha * (r + self.discount * np.max(self.Q[sp]) - self.Q[s, a])
                    
    # Interpolates values of Q function to unvisited state-action pairs based on 1 nearest neighbor.
    def interpolate_Q(self):
        print("Interpolating Q...")
        for s in tqdm(self.state_space):
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
                    if self.Q[neighbor[0], neighbor[1]] is not None and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
                except:
                    pass
                
        
class BasicQAgent(Player):
    def __init__(self, pos: int):
        super().__init__(pos)
        state_space = [i for i in range(4 * 14 ** 4)]
        action_space = [i for i in range(52)]
        self.q_learner = QLearner(state_space, action_space, discount=1, alpha=0.01)
        
        # needed for plotting
        self.win_percent_b = []
        self.lose_percent_b = []
        self.orw_p_b = []
        self.orl_p_b = []

        self.win_percent_g = []
        self.lose_percent_g = []
        self.orw_p_g = []
        self.orl_p_g = []
    
    # Employs batch Q-learning with actions constrained by rules
    def take_turn(self, trick: Trick, tricks: 'list[Trick]', players: 'list[Player]', hearts_broken: bool) -> Card:
        legal_moves = self.get_legal_moves(trick, hearts_broken)
        state = self.get_state(trick, tricks, legal_moves)
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
    def batch_q_learn(self, df: pandas.DataFrame):
        for i in range(NUM_TRAINING_PASSES):
            print(f"Learning epoch {i}")
            for idx, data in tqdm(df.iterrows()):
                self.q_learner.update(data["s"], data["a"], data["r"], data["sp"])
                # if idx % 5200 == 0:
                #     self.evaluate_performance()
        self.q_learner.interpolate_Q()
        
    def plot_stats(self, type):
        plt.figure(1)
        x_t = np.array([i * 100 for i in range(len(self.win_percent_b))])
        if type == 'baseline':
            wins_t = np.array(self.win_percent_b)
            losses_t = np.array(self.lose_percent_b)
            orw_t = np.array(self.orw_p_b)
            orl_t = np.array(self.orl_p_b)
        elif type == 'greedy':
            wins_t = np.array(self.win_percent_g)
            losses_t = np.array(self.lose_percent_g)
            orw_t = np.array(self.orw_p_g)
            orl_t = np.array(self.orl_p_g)

        plt.title('Percentage of wins/losses over time')
        plt.xlabel('Episode')
        plt.ylabel('Percentage')
        plt.plot(x_t, wins_t, label = "wins")
        plt.plot(x_t, losses_t, label = "losses")
        plt.plot(x_t, orw_t, label = "one round wins")
        plt.plot(x_t, orl_t, label = "one round losses")
        plt.legend()
        
        plt.savefig("stats" + type + ".png")
        plt.clf()
    
    def evaluate_performance(self):
        eval_players_b = [self, BaselineAgent(1), BaselineAgent(2), BaselineAgent(3)]
        print("Evaluating against baseline...")
        one_round_wins, one_round_losses = evaluate(eval_players_b, end_threshold=0, num_evals=5000)
        full_game_wins, full_game_losses = evaluate(eval_players_b, end_threshold=100, num_evals=500)

        self.win_percent_b.append(full_game_wins[0])
        self.lose_percent_b.append(full_game_losses[0])
        self.orw_p_b.append(one_round_wins[0])
        self.orl_p_b.append(one_round_losses[0])

        eval_players_g = [self, GreedyBaseline(1), GreedyBaseline(2), GreedyBaseline(3)]
        print("Evaluating against greedy...")
        one_round_wins, one_round_losses = evaluate(eval_players_g, end_threshold=0, num_evals=5000)
        full_game_wins, full_game_losses = evaluate(eval_players_g, end_threshold=100, num_evals=500)

        self.win_percent_g.append(full_game_wins[0])
        self.lose_percent_g.append(full_game_losses[0])
        self.orw_p_g.append(one_round_wins[0])
        self.orl_p_g.append(one_round_losses[0])
        
        
# Simulates games and trains the BasicQAgent based on the simulated transition/reward data.
# Saves the training data to a csv. Evaluates the BasicQAgent against random baseline agents.
def main():
    state_record = StateRecord()
    train_players = [BaselineAgent(0), GreedyBaseline(1), BaselineAgent(2), GreedyBaseline(3)]
    print("Generating training data...")
    for i in tqdm(range(NUM_TRAINING_GAMES)):
        play(train_players, 0, False, state_record)
    state_record.write_to_csv('data/train_data.csv')
    df = pandas.DataFrame(state_record.record[1:], columns=['player', 's', 'a', 'r', 'sp'])
    del state_record
    
    q_agent = BasicQAgent(0)
    q_agent.batch_q_learn(df)
    
    eval_players = [q_agent, BaselineAgent(1), BaselineAgent(2), BaselineAgent(3)]
    # print("Evaluating...")
    # one_round_wins, one_round_losses = evaluate(eval_players, end_threshold=0, num_evals=100000)
    # print("One round win percentages: ", one_round_wins)
    # print("One round loss percentages: ", one_round_losses)
    # full_game_wins, full_game_losses = evaluate(eval_players, end_threshold=100, num_evals=10000)
    # print("Full game win percentages: ", full_game_wins)
    # print("Full game loss percentages: ", full_game_losses)
    
    evaluate(eval_players, end_threshold=0, num_evals=20, console_game=True)
    
    # q_agent.plot_stats('baseline')
    # q_agent.plot_stats('greedy')
    
if __name__ == '__main__':
    main()