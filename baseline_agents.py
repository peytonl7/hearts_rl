"""
File: baseline_agent.py 
A simple computer Hearts agent. Given a state, it takes a random legal action.

Author: Michelle Fu
"""

from classes import Player, Trick, Card
import random

DEBUGGING = False

class BaselineAgent(Player):
    def __init__(self, pos: int):
        super().__init__(pos)
    
    # this baseline agent just plays a random legal move
    def take_turn(self, trick: Trick, tricks: list[Trick], hearts_broken: bool) -> Card:
        legal_moves = []
        has_suit = any(c.suit == trick.suit for c in self.hand)
        if not has_suit:
            for card in self.hand: 
                if not trick.suit and card.suit == 'h' and not hearts_broken and any(c.suit != 'h' for c in self.hand):
                    continue
                else:
                    legal_moves.append(card)
        else:
            for card in self.hand:
                if card.suit == trick.suit:
                    legal_moves.append(card)
        move = random.choice(legal_moves)
        
        if DEBUGGING:
            hand = sorted([card.name for card in self.hand], key=lambda x: x[-1])
            choices = sorted([card.name for card in legal_moves], key=lambda x: x[-1])
            print("Hand: ", hand)
            print("From ", choices, ", picked ", move.name)
        self.hand.remove(move)
        return move


class GreedyBaseline(Player):
    def __init__(self, pos: int):
        super().__init__(pos)

    # Follows the following heuristic policy:
    # - Play the highest legal card you can without winning a heart, i.e. if 
    #   a heart is already played, the highest card below the current trick winner; 
    #   else the highest card of any legal suit
    #   - EXCEPT if it's the queen, king, or ace of spades
    #       - EXCEPT if the queen of spades has already been played this game
    # - Play the queen of spades if it's safe to get rid of it, i.e. you will not 
    #   collect it based on the current state of the trick
    def take_turn(self, trick: Trick, tricks: list[Trick], hearts_broken: bool) -> Card:
        legal_moves = []
        has_suit = any(c.suit == trick.suit for c in self.hand)
        if not has_suit:
            for card in self.hand: 
                if not trick.suit and card.suit == 'h' and not hearts_broken and any(c.suit != 'h' for c in self.hand):
                    continue
                else:
                    legal_moves.append(card)
        else:
            for card in self.hand:
                if card.suit == trick.suit:
                    legal_moves.append(card)
        
        # Checks state for if the current trick has points, what the current winning
        # card of the trick is, if the queen of spades has been played, and if it's
        # safe to play the queen of spades
        has_points = any(card.suit == 'h' or card.name == 'QS' for card in trick.cards.values())
        winning_card = None
        if trick.suit:
            for card in trick.cards.values():
                if not winning_card or (card.suit == trick.suit and card.rank > winning_card.rank):
                    winning_card = card 
        black_lady_played = any(any(card.name == 'QS' for card in t.cards.values()) for t in tricks)
        safe_to_play_QS = trick.suit and trick.suit != 's' or any(card.name in ['AS', 'KS'] for card in trick.cards.values())
        
        # Decides on move based on heuristic
        move = None
        # Case: Black Lady safe to play. Play it.
        if safe_to_play_QS and any(card.name == 'QS' for card in legal_moves):
            for card in legal_moves:
                if card.name == 'QS':
                    move = card
        # Case: Play the highest legal card, either because:
        # - No points yet
        # - Don't have suit, so won't win trick
        # - Setting suit of trick
        # - Only have cards that will become the winning card
        elif not has_points or not has_suit or not winning_card or all(card.rank > winning_card.rank for card in legal_moves):
            for card in legal_moves:
                if not move or card.rank > move.rank:
                    if black_lady_played or card.name not in ["QS", "KS", "AS"] or all(c.name in ["QS", "KS", "AS"] for c in legal_moves):
                        move = card
        # Case: Play the highest legal card below the current winning card, so as not to win points.
        else:
            for card in legal_moves:
                if (not move or card.rank > move.rank) and card.rank < winning_card.rank:
                    move = card
          
        if DEBUGGING:          
            hand = sorted([card.name for card in self.hand], key=lambda x: x[-1])
            choices = sorted([card.name for card in legal_moves], key=lambda x: x[-1])
            print("Hand: ", hand)
            print("From ", choices, ", picked ", move.name)
        self.hand.remove(move)
        return move
