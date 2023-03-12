"""
File: simulate_transition.py
Last update: 03/09 by Michelle

Simulates a transition. Given an input state and an action, return a next state and a reward.
"""
from random import shuffle
from classes import Player, Trick
from baseline_agents import BaselineAgent
from play import generate_deck
from utils import state_to_vec

NUM_PLAYERS = 4
AGENT_INDEX = 0

def get_starting_state():
    players = [Player(0), BaselineAgent(1), BaselineAgent(2), BaselineAgent(3)] # we might test against greedy baseline later?
    deck = generate_deck()  
    shuffle(deck)
    curr_player = 0
    for card in deck:
        players[curr_player].add_card_to_hand(card)
        curr_player = (curr_player + 1) % NUM_PLAYERS
    # need to return info for other players so we can play the game
    state = state_to_vec(players, Trick(NUM_PLAYERS))
    return state, players

def simulate_transition(curr_trick: Trick, tricks: 'list[Trick]', players: 'list[Player]', hearts_broken, action):
    curr_player = AGENT_INDEX # each transition starts with the agent taking an action

    # finish the current trick
    while len(curr_trick.cards) < 4:
        player = players[curr_player]
        if curr_player == AGENT_INDEX:
            curr_trick.add_card(curr_player, action)
            player.prev_actions.append(action)
        else:
            card = player.take_turn(curr_trick, tricks, hearts_broken)
            curr_trick.add_card(curr_player, card)
            player.prev_actions.append(card)

        curr_player = (curr_player + 1) % NUM_PLAYERS
    
    winner = curr_trick.determine_winner()
    players[winner].won_tricks += curr_trick.cards.values()
    players[winner].won_hearts = players[winner].won_hearts or any(c.suit == 'h' for c in curr_trick.cards.values())
    hearts_broken = hearts_broken or any(c.suit == 'h' for c in curr_trick.cards.values())
    tricks.append(curr_trick)

    curr_score = players[AGENT_INDEX].compute_score()
    reward = players[AGENT_INDEX].prev_score - curr_score
    players[AGENT_INDEX].prev_score = curr_score

    if len(players[AGENT_INDEX].hand) == 0:
        terminated = True
        next_trick = None
    else:
        terminated = False
        # start the next trick
        trick_starter = winner
        next_trick = Trick(NUM_PLAYERS)
        curr_player = trick_starter
        our_pos = 0
        for i in range(NUM_PLAYERS):
            if curr_player == AGENT_INDEX: 
                our_pos = i
                break # this signals that we've transitioned to our new state

            player = players[curr_player]
            card = player.take_turn(curr_trick, tricks, hearts_broken)
            curr_trick.add_card(curr_player, card)
            player.prev_actions.append(card)
            curr_player = (curr_player + 1) % NUM_PLAYERS
    
    state_vec = None if terminated else state_to_vec(players, next_trick) 
    return next_trick, tricks, players, hearts_broken, state_vec, reward # this is disgusting
