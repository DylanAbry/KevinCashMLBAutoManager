"""Run expectancy (RE24) and a simple win-probability estimate.

The RE24 values are approximate modern-MLB averages (expected runs from this state to the end of the
half-inning). They can later be recomputed from your own simulator or real play-by-play data.
"""
from math import erf, sqrt

from .game_state import GameState

# key: (base_index, outs); base_index = 1*(runner on 1B) + 2*(2B) + 4*(3B)
RE24 = {
    (0, 0): 0.50, (0, 1): 0.27, (0, 2): 0.10,
    (1, 0): 0.90, (1, 1): 0.54, (1, 2): 0.22,
    (2, 0): 1.13, (2, 1): 0.69, (2, 2): 0.33,
    (3, 0): 1.47, (3, 1): 0.93, (3, 2): 0.46,
    (4, 0): 1.36, (4, 1): 0.95, (4, 2): 0.37,
    (5, 0): 1.79, (5, 1): 1.17, (5, 2): 0.50,
    (6, 0): 2.00, (6, 1): 1.38, (6, 2): 0.58,
    (7, 0): 2.28, (7, 1): 1.53, (7, 2): 0.75,
}
RUNS_PER_HALF_INNING = 0.50
VAR_PER_HALF_INNING = 1.10


def run_expectancy(state: GameState) -> float:
    return RE24[(state.base_index, min(state.outs, 2))]


def win_prob_home(s: GameState) -> float:
    """Normal approximation of the home team's win probability from the current state."""
    future_top = max(9 - s.inning, 0)
    future_bot = max(9 - s.inning, 0) + (1 if s.top else 0)
    mean = (s.home_score - s.away_score) + (future_bot - future_top) * RUNS_PER_HALF_INNING
    mean += run_expectancy(s) * (-1 if s.top else 1)
    sd = sqrt(VAR_PER_HALF_INNING * (future_top + future_bot + 1))
    return 0.5 * (1 + erf(mean / (sd * sqrt(2))))