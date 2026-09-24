"""Exact (no random noise) expected runs for a batting order, via a base-out-batter Markov chain,
plus a swap-based search for the best order.

State = (outs, bases, batter due up) = 216 states. From every half-inning start state we get the expected
runs and the distribution of who leads off the next inning, then chain nine innings together.
Uses the same baserunning rules and RISP effects as the simulator; ignores steals (they need runner identity).
"""
from itertools import combinations

import numpy as np

from sim.baserunning import summary
from sim.pa_model import outcome_probs
from sim.types import BatterModel

N_STATES = 3 * 8 * 9


def _half_inning_tables(batters: list[BatterModel], vs_starter: bool):
    Q = np.zeros((N_STATES, N_STATES))
    R = np.zeros((N_STATES, 9))
    r = np.zeros(N_STATES)
    for outs in range(3):
        for mask in range(8):
            for b in range(9):
                i = (outs * 8 + mask) * 9 + b
                bm = batters[b]
                target = bm.woba(vs_starter) + (bm.risp_delta if mask & 6 else 0.0)
                nxt = (b + 1) % 9
                for outcome, p in enumerate(outcome_probs(bm.profile, target)):
                    if p <= 0:
                        continue
                    for q, new_outs, new_mask, runs in summary(mask, outs, outcome):
                        pp = p * q
                        r[i] += pp * runs
                        if new_outs >= 3:
                            R[i, nxt] += pp
                        else:
                            Q[i, (new_outs * 8 + new_mask) * 9 + nxt] += pp
    sol = np.linalg.solve(np.eye(N_STATES) - Q, np.column_stack([r, R]))
    return sol[:9, 0], sol[:9, 1:]          # start rows = (0 outs, bases empty, batter b)


def expected_runs(batters: list[BatterModel], starter_innings: int = 6) -> float:
    """Expected runs over 9 innings against the opposing starter (then bullpen)."""
    Es, Ts = _half_inning_tables(batters, True)
    Ep, Tp = (_half_inning_tables(batters, False) if starter_innings < 9 else (Es, Ts))
    v = np.zeros(9)
    v[0] = 1.0
    total = 0.0
    for inning in range(1, 10):
        E, T = (Es, Ts) if inning <= starter_innings else (Ep, Tp)
        total += float(v @ E)
        v = v @ T
    return total


def optimize_order(batters: list[BatterModel], pins: dict[int, int] | None = None, starter_innings: int = 6):
    """batters are given in a sensible starting order. pins maps batter index -> batting slot (0-8).
    Returns (order, expected_runs) where order[k] is the batter index hitting in slot k."""
    pins = pins or {}
    order: list[int | None] = [None] * 9
    for bi, slot in pins.items():
        order[slot] = bi
    rest = iter(i for i in range(9) if i not in pins)
    order = [o if o is not None else next(rest) for o in order]
    free = [k for k in range(9) if k not in {s for s in pins.values()}]

    def score(o):
        return expected_runs([batters[i] for i in o], starter_innings)

    best = score(order)
    for _ in range(20):
        move, move_score = None, best + 1e-7
        for a, b in combinations(free, 2):
            cand = order[:]
            cand[a], cand[b] = cand[b], cand[a]
            sc = score(cand)
            if sc > move_score:
                move, move_score = cand, sc
        if move is None:
            break
        order, best = move, move_score
    return order, best
