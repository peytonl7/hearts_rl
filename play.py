"""
File: classes.py
Last update: 2/23 by Peyton

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
"""

from classes import Card, Trick, ConsolePlayer, Player

from random import shuffle

NUM_PLAYERS = 4
END_THRESHOLD = 0
CONSOLE_GAME = True

# Returns a full deck of cards.
def generate_deck() -> list[Card]:
    deck = []
    for r in range(2, 15):
        for s in ['s', 'd', 'h', 'c']:
            deck.append(Card(r, s))
    return deck

# Play one trick and return the new trick starter and whether hearts are broken.
def play_trick(players: list[Player], trick_starter: int, hearts_broken: bool):
    trick = Trick(NUM_PLAYERS)
    curr_player = trick_starter
    # Each player plays a card
    for i in range(NUM_PLAYERS):
        card = players[curr_player].take_turn(trick, hearts_broken)
        trick.add_card(curr_player, card)
        curr_player = (curr_player + 1) % NUM_PLAYERS
    
    # Winner of trick is determined and the cards go to the winner.
    # Next trick's starter is set.
    winner = trick.determine_winner()
    players[winner].won_tricks += trick.cards.values()
    if CONSOLE_GAME:
        print("Player " + str(winner) + " won the trick!")
    hearts_broken = hearts_broken or any(c.suit == 'h' for c in trick.cards.values())
    return winner, hearts_broken
        
# Given a list of players, runs a full game of hearts.
# Returns a dictionary of players (by pos) and their scores.
def run_game(players: list[Player]) -> dict:
    deck = generate_deck()  # For toy example (two cards per hand), modify this line.
    shuffle(deck)
    curr_player = 0
    for card in deck:
        players[curr_player].add_card_to_hand(card)
        curr_player = (curr_player + 1) % NUM_PLAYERS
    
    players = sorted(players, key=lambda x: x.pos)
    trick_starter = 0
    hearts_broken = False
    while players[0].hand:
        trick_starter, hearts_broken = play_trick(players, trick_starter, hearts_broken)
        
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
def main():
    players = []
    # Instantiates players. Modify to change strategies.
    for i in range(NUM_PLAYERS):
        players.append(ConsolePlayer(i))
    
    while True:
        scores = run_game(players)
        if CONSOLE_GAME:
            print("Results: ")
            for player, score in scores.items():
                print("Player " + str(player) + ": " + str(score))
        if any(player.total_score >= END_THRESHOLD for player in players):
            break
        
    final_scores = [player.total_score for player in players]
    loser = final_scores.index(max(final_scores))
    print("Player " + str(loser) + " lost!")

if __name__ == "__main__":
    main()