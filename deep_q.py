"""
File: deep_q.py
Last update: 03/09 by Michelle
Attributions: I followed the PyTorch deep-Q tutorial (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
but typed all code myself and modified it for the Hearts environment

Contains code for the Deep Q-Learning Hearts agent
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from classes import Card, Trick, Player
from utils import card_to_index
from simulate_transition import get_starting_state, simulate_transition

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    # called on one element (to determine next action) or batch
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class Trainer():
    def __init__(self) -> None:
        self.batch_size = 128
        self.gamma = 1
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005 # update rate of target network
        self.lr = 0.0004 

        n_actions = 52 # one for each card
        # 52 to encode cards in play, 52 to encode cards in hand, 52 to encode what cards have been previously played
        # 4 to encode player order, 4 to encode suit of current hand, 4 to encode which players have won hearts
        n_observations = 168

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.parameters())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(100) # this was suggested in a paper
        self.steps_done = 0

    # TODO: write action_from_vec, find way to get card from name
    def get_action(self, state, legal_actions):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # roll RNG
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                action_tsr = self.policy_net(state).max(1)[1].view(1, 1)
                action = action_from_vec(action_tsr) # WRITE THIS
                return action, action_tsr
        else:
            action = random.choice(legal_actions)
            action_tsr = [0] * 52
            action_tsr[card_to_index(action.name)] = 1
            return action, torch.tensor(action_tsr, device=device, dtype = torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
    
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def train(self):
        num_epochs = 0
        if torch.cuda.is_available():
            num_epochs = 1000 # will stop if we converge
        else:
            num_epochs = 50 
        
        for i in range(num_epochs):
            curr_trick = Trick(NUM_PLAYERS)
            tricks = []
            hearts_broken = False
            state, players = get_starting_state()
            legal_actions = players[AGENT_INDEX].get_legal_moves(curr_trick, hearts_broken)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            while True: # play a whole game
                if state is None: break
                action, action_tsr = self.get_action(state, legal_actions)
                curr_trick, tricks, players, hearts_broken, next_state, reward = \
                    simulate_transition(curr_trick, tricks, players, hearts_broken, action) # play the trick to the end and get the next state

                if next_state != None:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                
                self.memory.push(state, action_tsr, next_state, reward)
                state = next_state
                self.optimize_model()

                # soft update target net weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

# TODO: write code for the agent to play a game (probably need to load trained weights somewhere)
