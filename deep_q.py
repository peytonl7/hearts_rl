"""
File: deep_q.py
Last update: 03/17/23 by Michelle
Attributions: I followed the PyTorch deep-Q tutorial (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
but typed all code myself and modified it for the Hearts environment

Contains code for the Deep Q-Learning Hearts agent
Author: Michelle Fu
"""

import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from classes import Card, Trick, Player
from utils import action_from_tsr, state_to_vec, CARD_TO_IND
from simulate_transition import get_starting_state, simulate_transition # comment this out to play against DQN
from baseline_agents import BaselineAgent, GreedyBaseline
from evaluate import evaluate # comment this out to play against DQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'legal_mask'))
NUM_PLAYERS = 4
AGENT_INDEX = 0

# stores transitions for experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    # save a transition
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    # sample a transition
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
    
    # called on one element (to determine next action) or batch
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class Trainer():
    def __init__(self) -> None:
        self.batch_size = 800
        self.gamma = 1
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.0001 # update rate of target network
        self.lr = 0.00001

        # needed for plotting
        self.win_percent_b = []
        self.lose_percent_b = []
        self.orw_p_b = []
        self.orl_p_b = []

        self.win_percent_g = []
        self.lose_percent_g = []
        self.orw_p_g = []
        self.orl_p_g = []

        n_actions = 52 # one for each card
        # 52 to encode cards in play, 52 to encode cards in hand, 52 to encode what cards have been previously played
        # 4 to encode player order, 4 to encode suit of current hand, 4 to encode which players have won points
        n_observations = 168

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(1300)
        self.steps_done = 0

    def plot_stats(self, type):
        plt.figure(1)
        x_t = torch.tensor([i * 100 for i in range(len(self.win_percent_b))], dtype=torch.int)
        if type == 'baseline':
            wins_t = torch.tensor(self.win_percent_b, dtype=torch.float)
            losses_t = torch.tensor(self.lose_percent_b, dtype=torch.float)
            orw_t = torch.tensor(self.orw_p_b, dtype=torch.float)
            orl_t = torch.tensor(self.orl_p_b, dtype=torch.float)
        elif type == 'greedy':
            wins_t = torch.tensor(self.win_percent_g, dtype=torch.float)
            losses_t = torch.tensor(self.lose_percent_g, dtype=torch.float)
            orw_t = torch.tensor(self.orw_p_g, dtype=torch.float)
            orl_t = torch.tensor(self.orl_p_g, dtype=torch.float)

        plt.title('Percentage of wins/losses over time')
        plt.xlabel('Episode')
        plt.ylabel('Percentage')
        plt.plot(x_t.numpy(), wins_t.numpy(), label = "wins")
        plt.plot(x_t.numpy(), losses_t.numpy(), label = "losses")
        plt.plot(x_t.numpy(), orw_t.numpy(), label = "one round wins")
        plt.plot(x_t.numpy(), orl_t.numpy(), label = "one round losses")
        plt.legend()

        plt.savefig("stats" + type + ".png")
        plt.clf()

    def get_action(self, state, legal_actions):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # roll RNG
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                actions = self.policy_net(state).squeeze()
                legal_indices = torch.LongTensor([CARD_TO_IND[card.name] for card in legal_actions]).to(device)
                legal_moves_vals = torch.index_select(actions, 0, legal_indices)
                action_ind = legal_indices[torch.argmax(legal_moves_vals).item()]
                suit, rank, id = action_from_tsr(action_ind) 
                action = Card(rank, suit, id)
                action_tsr = torch.tensor([action_ind], device=device, dtype = torch.long)
                return action, action_tsr
        else:
            action = random.choice(legal_actions)
            action_tsr = [CARD_TO_IND[action.name]]
            return action, torch.tensor(action_tsr, device=device, dtype = torch.long)
    
    def get_max(self, indices, mask):
        action_values = self.target_net(indices)
        action_values = action_values.masked_fill(mask, -10000)
        return action_values.max(1)[0]
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
    
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        mask_batch = torch.stack(batch.legal_mask)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.get_max(non_final_next_states, mask_batch[non_final_mask])
        # compute expected Q values
        expected_state_action_values = torch.add((next_state_values * self.gamma).unsqueeze(1),reward_batch)
        # compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def evaluate_performance(self):
        q_agent = deepQAgent(0, self.policy_net)
        eval_players_b = [q_agent, BaselineAgent(1), BaselineAgent(2), BaselineAgent(3)]
        print("Evaluating against baseline...")
        one_round_wins, one_round_losses = evaluate(eval_players_b, end_threshold=0, num_evals=1300)
        full_game_wins, full_game_losses = evaluate(eval_players_b, end_threshold=100, num_evals=100)

        self.win_percent_b.append(full_game_wins[0])
        self.lose_percent_b.append(full_game_losses[0])
        self.orw_p_b.append(one_round_wins[0])
        self.orl_p_b.append(one_round_losses[0])

        eval_players_g = [q_agent, GreedyBaseline(1), GreedyBaseline(2), GreedyBaseline(3)]
        print("Evaluating against greedy...")
        one_round_wins, one_round_losses = evaluate(eval_players_g, end_threshold=0, num_evals=1300)
        full_game_wins, full_game_losses = evaluate(eval_players_g, end_threshold=100, num_evals=100)

        self.win_percent_g.append(full_game_wins[0])
        self.lose_percent_g.append(full_game_losses[0])
        self.orw_p_g.append(one_round_wins[0])
        self.orl_p_g.append(one_round_losses[0])

    def train(self, num_epochs):
        for i in tqdm(range(num_epochs)):
            if i % 100 == 0:
                self.evaluate_performance()
            curr_trick = Trick(NUM_PLAYERS)
            tricks = []
            hearts_broken = False
            state, players = get_starting_state()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            while True: # play a whole game
                this_reward = 0
                legal_actions = players[AGENT_INDEX].get_legal_moves(curr_trick, hearts_broken)
                action, action_tsr = self.get_action(state, legal_actions)
                curr_trick, tricks, players, hearts_broken, next_state, reward = \
                    simulate_transition(curr_trick, tricks, players, hearts_broken, action) # play the trick to the end and get the next state

                if next_state != None:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                this_reward += reward
                reward_tsr = torch.tensor([reward], dtype=torch.float32, device=device)
                mask = torch.zeros(52).to(device)
                legal_indices = torch.LongTensor([CARD_TO_IND[card.name] for card in legal_actions]).to(device)
                mask = ~(mask.scatter(0, legal_indices, 1.).bool())
                self.memory.push(state, action_tsr, next_state, reward_tsr, mask)
                state = next_state
                self.optimize_model()

                if state is None:
                    break

            # soft update target net weights
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

        self.evaluate_performance()
        torch.save(self.policy_net, 'deepq-policy.pt')

class deepQAgent(Player):
    def __init__(self, pos: int, policy_net: DQN):
        super().__init__(pos)
        self.policy_net = policy_net
    
    def take_turn(self, trick: Trick, tricks: 'list[Trick]', players: 'list[Player]', hearts_broken: bool) -> Card:
        state = state_to_vec(trick=trick, players=players)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            legal_actions = players[AGENT_INDEX].get_legal_moves(trick, hearts_broken)
            with torch.no_grad():
                actions = self.policy_net(state).squeeze()
                legal_indices = torch.LongTensor([CARD_TO_IND[card.name] for card in legal_actions]).to(device)
                legal_moves_vals = torch.index_select(actions, 0, legal_indices)
                action_ind = legal_indices[torch.argmax(legal_moves_vals).item()]
                suit, rank, id = action_from_tsr(action_ind) 
                for c in self.hand:
                    if c.suit == suit and c.rank == rank:
                        self.hand.remove(c)
                        return c
        
def main():
    num_epochs = 200
    if len(sys.argv) == 2:
        num_epochs = int(sys.argv[1])
    trainer = Trainer()
    trainer.train(num_epochs)
    policy_net = torch.load('deepq-policy.pt')
    policy_net = policy_net.to(device)

    q_agent = deepQAgent(0, policy_net)
    eval_players = [q_agent, BaselineAgent(1), BaselineAgent(2), BaselineAgent(3)]
    print("Evaluating against baseline...")
    one_round_wins, one_round_losses = evaluate(eval_players, end_threshold=0, num_evals=10000)
    full_game_wins, full_game_losses = evaluate(eval_players, end_threshold=100, num_evals=1000)
    print("one round wins:", one_round_wins)
    print("one round losses:", one_round_losses)
    print("full game wins:", full_game_wins)
    print("full game losses", full_game_losses)

    eval_players = [q_agent, GreedyBaseline(1), GreedyBaseline(2), GreedyBaseline(3)]
    print("Evaluating against greedy...")
    one_round_wins, one_round_losses = evaluate(eval_players, end_threshold=0, num_evals=10000)
    full_game_wins, full_game_losses = evaluate(eval_players, end_threshold=100, num_evals=1000)
    print("one round wins:", one_round_wins)
    print("one round losses:", one_round_losses)
    print("full game wins:", full_game_wins)
    print("full game losses", full_game_losses)

    trainer.plot_stats('baseline')
    trainer.plot_stats('greedy')
    
if __name__ == '__main__':
    main()
