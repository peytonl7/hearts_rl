"""
File: evaluate.py
Last update: 2/28 by Peyton
"""

from play import play
from classes import Player
from baseline_agents import BaselineAgent, GreedyBaseline

from tqdm import tqdm

# Evaluates the player configuration with the given end-game threshold, testing 
# for num_eval iterations.
def evaluate(players: 'list[Player]', end_threshold: int, num_evals: int):
    wins, losses = [0, 0, 0, 0], [0, 0, 0, 0]
    for i in tqdm(range(num_evals)):
        final_scores = play(players, end_threshold, False, None)
        for player, score in final_scores.items():
            if score == max(final_scores.values()):
                losses[player] += 1
            if score == min(final_scores.values()):
                wins[player] += 1
    
    return [n * 100 / num_evals for n in wins], [n * 100 / num_evals for n in losses]

def main():
    # Instantiate players. Change to test models.
    players = []
    players.append(GreedyBaseline(0))
    players.append(BaselineAgent(1))
    players.append(BaselineAgent(2))
    players.append(BaselineAgent(3))
    """
    Peyton 2/28: With these players (one Greedy vs. three Baseline) and END_THRESHOLD = 100,
    the GreedyBaseline basically never loses (less than 1% of rounds).
    
    That said, it does lose individual rounds, usually because it's forced
    to play AS or KS, or because a BaselineAgent randomly drops the QS on it.
    Still, it wins about half of individual rounds, losing just 12%.
    """
    one_round_wins, one_round_losses = evaluate(players, end_threshold=0, num_evals=100000)
    print("One round win percentages: ", one_round_wins)
    print("One round loss percentages: ", one_round_losses)
    full_game_wins, full_game_losses = evaluate(players, end_threshold=100, num_evals=10000)
    print("Full game win percentages: ", full_game_wins)
    print("Full game loss percentages: ", full_game_losses)
    
if __name__ == "__main__":
    main()
