"""
File: classes.py
Last update: 3/17/23 by Michelle

Plays the game.

Follows these Black Lady rules:
    - Player at pos 0 always leads the first trick.
    - You may not lead with a heart until one has been played
        - You may lead with the Black Lady anytime.
    - You may "break hearts" as early as the first trick.
    - Collecting all hearts and the Black Lady is "shooting the moon,"
      i.e. -25 for you and 25 for everyone else.
    - Collecting all tricks is "shooting the sun," i.e. -50 for you and
      50 for everyone else.
    - A full game ends when one person reaches END_THRESHOLD.

Author: Peyton Lee
"""

import sys
from classes import Card, Trick, ConsolePlayer, Player, StateRecord
from baseline_agents import BaselineAgent, GreedyBaseline
# uncomment this to play against DQN
# from deep_q import deepQAgent, DQN 
import torch

from random import shuffle

NUM_PLAYERS = 4
END_THRESHOLD = 100

# 13 * (14 ** 0 + 14 ** 1 + 14 ** 2 + 14 ** 3) 
FINAL_STATE = 38415

# Returns a full deck of cards.
def generate_deck() -> 'list[Card]':
    deck = []
    counter = 0
    for s in ['s', 'd', 'h', 'c']:
        for r in range(2, 15):
            deck.append(Card(r, s, counter))
            counter += 1
    return deck

# Play one trick and return the new trick starter and whether hearts are broken.
def play_trick(players: 'list[Player]', tricks: 'list[Trick]', trick_starter: int, hearts_broken: bool, console_game: bool,
               state_record: StateRecord):
    trick = Trick(NUM_PLAYERS)
    curr_player = trick_starter
    # Each player plays a card
    for i in range(NUM_PLAYERS):
        player = players[curr_player]
        state = player.get_state(trick, tricks, player.get_legal_moves(trick, hearts_broken))
        curr_score = player.compute_score()
        if state_record is not None and player.prev_state is not None:
            reward = player.prev_score - curr_score
            state_record.record.append([player.pos, player.prev_state, player.prev_action, reward, state])
        player.prev_state = state
        player.prev_score = curr_score
        
        card = player.take_turn(trick, tricks, players, hearts_broken)
        trick.add_card(curr_player, card)
        player.prev_action = card.id
        curr_player = (curr_player + 1) % NUM_PLAYERS
    
    # Winner of trick is determined and the cards go to the winner.
    # Next trick's starter is set.
    winner = trick.determine_winner()
    players[winner].won_tricks += trick.cards.values()
    if console_game:
        print("Player " + str(winner) + " won the trick!")
        print([str(player) + ": " + card.name for player, card in trick.cards.items()])
        print("------------------")
    hearts_broken = hearts_broken or any(c.suit == 'h' for c in trick.cards.values())
    tricks.append(trick)
    return winner, hearts_broken
        
# Given a list of players, runs a full game of hearts.
# Returns a dictionary of players (by pos) and their scores.
def run_game(players: 'list[Player]', console_game: bool, state_record: StateRecord) -> dict:
    deck = generate_deck()  # For toy example (two cards per hand), modify this line.
    shuffle(deck)
    curr_player = 0
    for card in deck:
        players[curr_player].add_card_to_hand(card)
        curr_player = (curr_player + 1) % NUM_PLAYERS
    
    players = sorted(players, key=lambda x: x.pos)
    trick_starter = 0
    hearts_broken = False
    tricks = []
    while players[0].hand:
        trick_starter, hearts_broken = play_trick(players, tricks, trick_starter, hearts_broken, console_game, state_record)
        
    game_scores = [player.compute_score() for player in players]
    if any(score < 0 for score in game_scores):
        try:
            shooter = game_scores.index(-25)
        except:
            shooter = game_scores.index(-50)
        for i in range(NUM_PLAYERS):
            game_scores[i] = -game_scores[shooter] if i != shooter else game_scores[i]
        
    for i in range(NUM_PLAYERS):
        players[i].total_score += game_scores[i]
    
    return {i:game_scores[i] for i in range(NUM_PLAYERS)}
    

# Runs the game until one player hits END_THRESHOLD points, then declares
# that player the loser.
def play(players: 'list[Player]', end_threshold: int, console_game: bool, state_record: StateRecord):
    while True:
        scores = run_game(players, console_game, state_record)
        if console_game:
            print("Results: ")
            for player, score in scores.items():
                print("Player " + str(player) + ": " + str(score))
        for player in players:
            state = FINAL_STATE
            curr_score = scores[player.pos]
            if state_record is not None:
                reward = player.prev_score - curr_score
                state_record.record.append([player.pos, player.prev_state, player.prev_action, reward, state])
            
            player.won_tricks.clear()
            player.prev_state = None
            player.prev_action = None
            player.prev_score = 0
        if any(player.total_score >= end_threshold for player in players):
            break
        
    labeled_final_scores = {player.pos:player.total_score for player in players}
    final_scores = [player.total_score for player in players]
    loser = final_scores.index(max(final_scores))
    if console_game:
        print("Final results: ")
        for p, score in labeled_final_scores.items():
            print("Player " + str(p) + ": " + str(score))
        print("Player " + str(loser) + " lost!")
    for player in players:
        player.total_score = 0
    return labeled_final_scores

if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    cpu_type = sys.argv[1]
    if cpu_type not in ['baseline', 'greedy']:
        print("Invalid player type. Should be one of: ", ['baseline', 'greedy'])
    else:
        players = []
        # for human analysis of deep q agent performance
        # policy_net = torch.load('deepq-policy.pt')
        # q_agent = deepQAgent(0, policy_net)
        # players.append(q_agent)
        players.append(ConsolePlayer(0))
        if cpu_type == 'baseline':
            players.append(BaselineAgent(1))
            players.append(BaselineAgent(2))
            players.append(BaselineAgent(3))
        elif cpu_type == 'greedy':
            players.append(GreedyBaseline(1))
            players.append(GreedyBaseline(2))
            players.append(GreedyBaseline(3))
        play(players, END_THRESHOLD, True, None)
        