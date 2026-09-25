"""Lineup + defensive alignment optimizer.

For every (hitter, position) pair we estimate the runs per game he adds:

    value = offense (expected wOBA vs today's starter)  +  defense at that position (DRS/OAA)

then solve the whole thing as one assignment problem (Hungarian algorithm), so the nine starters
and their positions are chosen together. Because DH has no defensive value, the DH slot naturally
goes to the hitter who gains the most from not fielding: a strong, reliable bat (talent is shrunk
by sample size) with the weakest glove. Batting order follows the "The Book" heuristic.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.names import find_by_name
from data.schema import Hitter, Pitcher
from models.matchup import LEAGUE_WOBA, PA_PER_BATTER_GAME, WOBA_SCALE, expected_woba

POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
FIELD_POSITIONS = POSITIONS[:-1]
RANK_TO_SLOT = [2, 1, 4, 3, 5, 6, 7, 8, 9]       # best hitters in slots 1, 2, 4 ("The Book")

DEFENSE_PRIOR_GAMES = 40      # shrinks small-sample defensive numbers toward 0
PA_PER_GAME_STARTER = 4.1     # to estimate games played from PA
INELIGIBLE_PENALTY = 0.10     # runs/game (~16 per 162) for playing someone out of position
INELIGIBLE_PENALTY_C = 0.30   # catching without being a catcher is far worse
PRIMARY_POS_BONUS = 0.002     # runs/game


@dataclass
class Spot:
    slot: int
    hitter: Hitter
    field_pos: str
    woba: float          # expected wOBA vs the opposing starter
    off_rpg: float       # offense, runs above average per game
    def_rpg: float       # defense, runs above average per game (0 at DH)


@dataclass
class Lineup:
    spots: list[Spot]
    def_rpg: float       # total runs per game the defense saves vs average
    off_rpg: float
    dh_note: str


def offense_rpg(woba: float) -> float:
    return (woba - LEAGUE_WOBA) / WOBA_SCALE * PA_PER_BATTER_GAME


def eligible(h: Hitter, pos: str) -> bool:
    if pos == "DH" or pos == h.pos or pos in h.def_runs:
        return True
    return pos in ("LF", "CF", "RF") and h.pos == "OF"


def defense_rpg(h: Hitter, pos: str) -> float:
    if pos == "DH":
        return 0.0
    runs = h.def_runs.get(pos)
    if runs is None:
        return 0.0
    games = h.pa / PA_PER_GAME_STARTER
    return runs / (games + DEFENSE_PRIOR_GAMES)


def _best_field_spot(h: Hitter) -> tuple[float, str] | None:
    options = [(defense_rpg(h, p), p) for p in FIELD_POSITIONS if p in h.def_runs or p == h.pos]
    return max(options) if options else None


FORCED_BONUS = 5.0            # runs/game added so a forced starter is always in the nine (same at every position)


def build_lineup(hitters: list[Hitter], pitcher: Pitcher, defense_weight: float = 1.0,
                 forced: frozenset | set = frozenset()) -> Lineup:
    """defense_weight=0 gives an offense-first baseline (still respects position eligibility).
    forced = hitter ids that must be in the starting nine (positions are still chosen by the optimizer)."""
    if len(hitters) < 9:
        raise ValueError(f"Need at least 9 hitters, got {len(hitters)}")
    woba = {h.id: expected_woba(h, pitcher) for h in hitters}
    value = np.zeros((len(POSITIONS), len(hitters)))
    for j, h in enumerate(hitters):
        for i, pos in enumerate(POSITIONS):
            v = offense_rpg(woba[h.id]) + defense_weight * defense_rpg(h, pos)
            if not eligible(h, pos):
                v -= INELIGIBLE_PENALTY_C if pos == "C" else INELIGIBLE_PENALTY
            elif pos == h.pos:
                v += PRIMARY_POS_BONUS      # tiny tie-breaker: prefer a player's natural position
            if h.id in forced:
                v += FORCED_BONUS
            value[i, j] = v
    rows, cols = linear_sum_assignment(-value)

    chosen = {hitters[j].id: (hitters[j], POSITIONS[i]) for i, j in zip(rows, cols)}
    ranked = sorted(chosen, key=lambda hid: woba[hid], reverse=True)
    spots = []
    for rank, hid in enumerate(ranked):
        h, pos = chosen[hid]
        spots.append(Spot(RANK_TO_SLOT[rank], h, pos, woba[hid], offense_rpg(woba[hid]), defense_rpg(h, pos)))
    spots.sort(key=lambda s: s.slot)

    dh = next(s for s in spots if s.field_pos == "DH")
    best = _best_field_spot(dh.hitter)
    if best:
        note = (f"DH: {dh.hitter.name} - expected wOBA {dh.woba:.3f} over {dh.hitter.pa} PA; "
                f"in the field his best spot ({best[1]}) is worth {best[0] * 162:+.1f} runs/162")
    else:
        note = f"DH: {dh.hitter.name} - expected wOBA {dh.woba:.3f} over {dh.hitter.pa} PA (no fielding data)"
    return Lineup(spots, sum(s.def_rpg for s in spots), sum(s.off_rpg for s in spots), note)



def draft_lineup(hitters: list[Hitter], names: list[str], pitcher: Pitcher) -> Lineup:
    """A user-drafted lineup: exactly nine names in batting order. Positions/defense are assigned optimally
    among those nine; the batting order is kept as given."""
    if len(names) != 9:
        raise SystemExit(f"--opp-lineup needs exactly 9 names separated by commas (got {len(names)})")
    chosen = [find_by_name(hitters, n, "the opposing active roster") for n in names]
    if len({h.id for h in chosen}) != 9:
        raise SystemExit("--opp-lineup lists the same player twice")
    order = {h.id: k for k, h in enumerate(chosen)}
    lu = build_lineup(chosen, pitcher)
    spots = sorted(lu.spots, key=lambda s: order[s.hitter.id])
    return replace(lu, spots=[replace(s, slot=k + 1) for k, s in enumerate(spots)])