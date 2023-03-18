"""
File: utils.py
Last update: 03/17/23 by Michelle

Utility functions.
Author: Michelle Fu
"""

CARD_TO_IND = {
    'AS': 0, '2S': 1, '3S': 2, '4S': 3, '5S': 4, '6S': 5, '7S': 6, '8S': 7, '9S': 8, '10S': 9, 'JS': 10, 'QS': 11, 'KS': 12,
    'AD': 13, '2D': 14, '3D': 15, '4D': 16, '5D': 17, '6D': 18, '7D': 19, '8D': 20, '9D': 21, '10D': 22, 'JD': 23, 'QD': 24, 'KD': 25,
    'AH': 26, '2H': 27, '3H': 28, '4H': 29, '5H': 30, '6H': 31, '7H': 32, '8H': 33, '9H': 34, '10H': 35, 'JH': 36, 'QH': 37, 'KH': 38,
    'AC': 39, '2C': 40, '3C': 41, '4C': 42, '5C': 43, '6C': 44, '7C': 45, '8C': 46, '9C': 47, '10C': 48, 'JC': 49, 'QC': 50, 'KC': 51
}
def int_rank_to_name(rank: int) -> str:
    if rank <= 10:
        return str(rank)
    if rank == 11:
        return 'J'
    if rank == 12:
        return 'Q'
    if rank == 13:
        return 'K'
    if rank == 14:
        return 'A'
    
# Helper for parsing input
def name_to_info(card_name: str):
    rank_name = card_name[:-1].lower()
    if rank_name == 'j':
        rank = 11
    elif rank_name == 'q':
        rank = 12
    elif rank_name == 'k':
        rank = 13
    elif rank_name == 'a':
        rank = 14
    else:
        try:
            rank = int(rank_name)
        except:
            return None, 'e'
    suit = card_name[-1].lower()
    return rank, suit if rank >= 2 and rank <= 14 and suit in ['s', 'd', 'h', 'c'] else 'e'

def suit_to_index(suit):
    suit_num = 0
    if suit == 'S':
        suit_num = 0
    elif suit == 'D':
        suit_num = 1
    elif suit == 'H':
        suit_num = 2
    elif suit == 'C':
        suit_num = 3
    return suit_num

def index_to_suit(index):
    suit = ''
    if index == 0:
        suit = 's'
    elif index == 1:
        suit = 'd'
    elif index == 2:
        suit = 'h'
    else:
        suit = 'c'
    return suit

def action_from_tsr(action_tsr):
    ind = action_tsr.item()
    id = ind // 13
    suit = index_to_suit(id)

    rank_num = ind % 13 + 1
    if rank_num == 1:
        rank_num = 14
    # print(suit, rank_num, id)
    return suit, rank_num, id

# 52 to encode cards in play, 52 to encode cards in hand, 52 to encode what cards have been previously played
# 4 to encode player order, 4 to encode suit of current hand, 4 to encode which players have won hearts
def state_to_vec(players, trick):
    in_play = [0] * 52
    in_hand = [0] * 52
    previous = [0] * 52
    our_pos = [0] * 4
    curr_suit = [0] * 4
    has_hearts = [0] * 4

    if trick.suit is not None:
        curr_suit[suit_to_index(trick.suit)] = 1
    our_pos[len(trick.cards)] = 1
    for i in range(4):
        player = players[i]
        for card in player.prev_actions:
            previous[CARD_TO_IND[card.name]] = 1
        if player.won_hearts:
            has_hearts[i] = 1
    for card in players[0].hand:
        in_hand[CARD_TO_IND[card.name]] = 1
    for card in trick.cards.values():
        in_play[CARD_TO_IND[card.name]] = 1   

    state_vec = in_play + in_hand + previous + our_pos + curr_suit + has_hearts
    return state_vec