"""
File: utils.py
Last update: 03/11 by Michelle

Utility functions.
"""
CARD_TO_IND = {
    'As': 0, '2s': 1, '3s': 2, '4s': 3, '5s': 4, '6s': 5, '7s': 6, '8s': 7, '9s': 8, '10s': 9, 'Js': 10, 'Qs': 11, 'Ks': 12,
    'Ad': 13, '2d': 14, '3d': 15, '4d': 16, '5d': 17, '6d': 18, '7d': 19, '8d': 20, '9d': 21, '10d': 22, 'Jd': 23, 'Qd': 24, 'Kd': 25,
    'Ah': 26, '2h': 27, '3h': 28, '4h': 29, '5h': 30, '6h': 31, '7h': 32, '8h': 33, '9h': 34, '10h': 35, 'Jh': 36, 'Qh': 37, 'Kh': 38,
    'Ac': 39, '2c': 40, '3c': 41, '4c': 42, '5c': 43, '6c': 44, '7c': 45, '8c': 46, '9c': 47, '10c': 48, 'Jc': 49, 'Qc': 50, 'Kc': 51
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
    if suit == 's':
        suit_num = 0
    elif suit == 'd':
        suit_num = 1
    elif suit == 'h':
        suit_num = 2
    elif suit == 'c':
        suit_num = 3
    return suit_num

# 52 to encode cards in play, 52 to encode cards in hand, 52 to encode what cards have been previously played
# 4 to encode player order, 4 to encode suit of current hand, 4 to encode which players have won hearts
def state_to_vec(players, trick):
    in_play = [0] * 52
    in_hand = [0] * 52
    previous = [0] * 52
    our_pos = [0] * 4
    curr_suit = [0] * 4
    has_hearts = [0] * 4

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
        in_play[CARD_TO_IND[card]] = 1   

    return in_play + in_hand + our_pos + curr_suit + has_hearts