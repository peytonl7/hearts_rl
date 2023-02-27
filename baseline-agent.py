"""
File: baseline-agent.py 
A simple computer Hearts agent. Given a state, it takes a random legal action.

Author: Michelle Fu
"""

from classes import Player, Trick, Card
import random

class Baseline_Agent(Player):
    def __init__(self, pos: int):
        super().__init__(pos)
    
    # this baseline agent just plays a random legal move
    def take_turn(self, trick: Trick, hearts_broken: bool) -> Card:
        legal_moves = []
        has_suit = any(c.suit == trick.suit for c in self.hand)
        if not has_suit:
            for card in self.hand: 
                if card.suit == 'h' and not hearts_broken:
                    continue
                else:
                    legal_moves.append(card)
        else:
            for card in self.hand:
                if card.suit == trick.suit:
                    legal_moves.append(card)
        move = random.choice(legal_moves)
        self.hand.remove(move)
        return move


