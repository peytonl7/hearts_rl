"""
File: classes.py
Last update: 2/28 by Peyton

Contains relevant classes for modeling the game.
"""

"""-----------MAIN CLASSES-----------"""

# Model of a card
class Card():
    def __init__(self, rank: int, suit: str):
        self.rank = rank    # int
        self.suit = suit    # Should be enforced as one of [s, d, h, c]
        self.name = int_rank_to_name(rank) + suit.upper()   # For display
        
# Trick used during gameplay. 13 instantiations in a normal round
class Trick():
    def __init__(self, num_players: int):
        self.cards = {}                 # list[Card]
        self.suit = None                # suit of the trick, i.e. first card played
        self.last_added = None          # int pos of last player to play a card in the trick
        self.num_players = num_players  # number of players
       
    # Adds a card and sets the suit if necessary         
    def add_card(self, player: int, card: Card):
        self.cards[player] = card
        if len(self.cards) == 1:
            self.suit = card.suit
        self.last_added = player
        
    # Determines winner based on original suit played
    def determine_winner(self):
        winner = 0
        curr_high = 0
        for player, card in self.cards.items():
            if card.suit == self.suit and card.rank > curr_high:
                winner = player
                curr_high = card.rank
        return winner        

# Generic player class. To make specific decision maker, inherit this.
class Player():
    def __init__(self, pos: int):
        self.pos = pos      # Should be enforced as one of [0, 1, 2, 3]
        self.hand = []
        self.won_tricks = []
        self.total_score = 0
    
    # Self-explanatory
    def add_card_to_hand(self, card: Card):
        self.hand.append(card)
    
    # Take state information and return a card to play. Should remove the card
    # from the player's hand. 
    # 
    # Should also verify that the card is a legal play. For the agent, this probably
    # means limiting the action space based on the state.
    def take_turn(self, trick: Trick, tricks: list[Trick], hearts_broken: bool) -> Card:
        # TODO: Defined by user input, heuristic policy, or reinforcement learning agent
        pass
    
    # After game is done, computes score based on won tricks. 
    # Returns the score for that game.
    def compute_score(self) -> int:
        game_score = 0
        for card in self.won_tricks:
            if card.suit == 'h':
                game_score += 1
            if card.suit == 's' and card.rank == 12:
                game_score += 13
        # Shooting the moon
        if game_score == 26:
            game_score = -25
        # Shooting the sun
        if len(self.won_tricks) == 52:
            game_score = -50
        return game_score
    
# A player who makes decisions based on console input.
class ConsolePlayer(Player):
    def __init__(self, pos: int):
        super().__init__(pos)
        
    def take_turn(self, trick: Trick, tricks: list[Trick], hearts_broken: bool) -> Card:
        for player, card in trick.cards.items():
            print("Player " + str(player) + " played " + card.name)
        card_names = sorted([card.name for card in self.hand], key=lambda x: x[-1])
        print("Your cards: ", card_names)
        
        # Input and error handling
        request = input("What will you play, player " + str(self.pos) + "? Enter here (format: 4H, 10d, AS): ")
        rank, suit = name_to_info(request)
        has_suit = any(c.suit == trick.suit for c in self.hand)
        invalid_input = suit == 'e'
        hearts_unbroken = not trick.suit and not hearts_broken and suit == 'h'
        invalid_suit = has_suit and suit != trick.suit
        card_not_in_hand = all(c.suit != suit or c.rank != rank for c in self.hand)
        while invalid_input or hearts_unbroken or invalid_suit or card_not_in_hand:
            if invalid_input: print("Invalid input.")
            elif hearts_unbroken: print("Hearts not broken! Pick a non-heart card.")
            elif invalid_suit: print("You must play a card of the trick's suit, since you have one.")
            elif card_not_in_hand: print("Card not in hand.")
            request = input("What will you play, player " + str(self.pos) + "? Enter here (format: 4H, 10d, AS): ")
            rank, suit = name_to_info(request)
            invalid_input = suit == 'e'
            hearts_unbroken = not trick.suit and not hearts_broken and suit == 'h'
            invalid_suit = has_suit and suit != trick.suit
            card_not_in_hand = all(c.suit != suit or c.rank != rank for c in self.hand)
        
        # Valid input: find and remove card from hand. Return card.
        for c in self.hand:
            if c.suit == suit and c.rank == rank:
                self.hand.remove(c)
                return c
    
"""-----------HELPERS-----------"""
    
# Helper for printing
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