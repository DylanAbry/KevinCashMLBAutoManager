"""Plate-appearance-level game simulator built on GameState.

What it models: player-specific outcome profiles, matchup-adjusted wOBA (starter vs bullpen), RISP hitting,
stolen-base attempts by the actual runner, reached-on-error, home-field advantage, and extra innings with
the automatic runner (the batter who made the last out of the previous inning).

Not modeled yet: pitchouts, wild pitches/passed balls, hit-and-run, bunts, intentional walks, injuries.
Run `python -m sim.engine` for a calibration check against real-MLB benchmarks.
"""
import random
from bisect import bisect
from dataclasses import dataclass

from models.game_state import GameState
from sim.baserunning import transition_table
from sim.pa_model import sample_outcome
from sim.types import TeamSide, league_batter

HOME_FIELD_WOBA_EDGE = 0.006      # added to home hitters' wOBA and subtracted from road hitters' (~54% home win%)
MAX_INNINGS = 25


def _half_inning(s: GameState, bat: TeamSide, fld: TeamSide, rng, counters: list) -> None:
    s.outs = 0
    idx = s.home_batter if s.batting_home else s.away_batter
    run = [None, None, None]                       # batting-order index of the runner on 1B / 2B / 3B
    if s.inning >= 10:
        run[1] = (idx - 1) % 9                     # automatic runner = last batter to make an out
    s.bases = tuple(r is not None for r in run)
    vs_starter = s.inning <= fld.starter_innings
    edge = HOME_FIELD_WOBA_EDGE if s.batting_home else -HOME_FIELD_WOBA_EDGE
    batters = bat.batters
    while s.outs < 3:
        idx = s.home_batter if s.batting_home else s.away_batter
        # stolen-base attempt (runner on 1st, 2nd base open)
        r1 = run[0]
        if r1 is not None and run[1] is None:
            rb = batters[r1]
            if rng.random() < rb.steal_attempt:
                if rng.random() < rb.steal_success:
                    run[1], run[0] = r1, None
                    counters[0] += 1
                else:
                    run[0] = None
                    s.outs += 1
                    counters[1] += 1
                    if s.outs >= 3:
                        break
        b = batters[idx]
        risp = run[1] is not None or run[2] is not None
        target = b.woba(vs_starter) + edge + (b.risp_delta if risp else 0.0)
        mask = (run[0] is not None) + 2 * (run[1] is not None) + 4 * (run[2] is not None)
        outcome = sample_outcome(b.profile, target, rng)
        cum, dlist = transition_table(mask, s.outs, outcome)
        dests = dlist[bisect(cum, rng.random())]
        who = (idx, run[0], run[1], run[2])
        new_run = [None, None, None]
        runs = 0
        for k in range(4):
            d = dests[k]
            if d < 0:
                continue
            if d == 0:
                s.outs += 1
            elif d == 4:
                runs += 1
            else:
                new_run[d - 1] = who[k]
        run = new_run
        s.bases = (run[0] is not None, run[1] is not None, run[2] is not None)
        if s.batting_home:
            s.home_score += runs
            s.home_batter = (idx + 1) % 9
            if s.inning >= 9 and s.home_score > s.away_score:
                return                             # walk-off
        else:
            s.away_score += runs
            s.away_batter = (idx + 1) % 9


def simulate_game(home: TeamSide, away: TeamSide, rng, counters: list | None = None) -> GameState:
    counters = counters if counters is not None else [0, 0]
    s = GameState()
    while s.inning <= MAX_INNINGS:
        s.top = True
        _half_inning(s, away, home, rng, counters)
        if s.inning >= 9 and s.home_score > s.away_score:
            break
        s.top = False
        _half_inning(s, home, away, rng, counters)
        if s.inning >= 9 and s.home_score != s.away_score:
            break
        s.inning += 1
    return s


@dataclass
class SimResult:
    games: int
    win_pct: float
    runs_for: float
    runs_against: float
    extras_pct: float            # share of games that went past 9 innings
    steal_attempts_pg: float     # attempts per team per game
    steal_success_pct: float


def simulate(ours: TeamSide, theirs: TeamSide, we_are_home: bool, n: int = 5000, seed: int = 1) -> SimResult:
    rng = random.Random(seed)
    home, away = (ours, theirs) if we_are_home else (theirs, ours)
    counters = [0, 0]
    wins = rf = ra = extras = 0
    for _ in range(n):
        s = simulate_game(home, away, rng, counters)
        our, opp = (s.home_score, s.away_score) if we_are_home else (s.away_score, s.home_score)
        wins += our > opp
        rf += our
        ra += opp
        extras += s.inning > 9
    sb, cs = counters
    att = sb + cs
    return SimResult(n, wins / n, rf / n, ra / n, extras / n, att / (2 * n), sb / att if att else 0.0)


if __name__ == "__main__":
    avg = TeamSide("League avg", [league_batter()] * 9)
    r = simulate(avg, avg, True, n=20000)
    print(f"League-average vs league-average, {r.games:,} games (real MLB: ~4.4-4.5 runs/team, home win% ~.535,"
          f" ~9% extras, ~0.9 steal attempts/team, ~78% steal success):")
    print(f"  runs/game home {r.runs_for:.2f}  away {r.runs_against:.2f}  home win% {r.win_pct:.3f}  "
          f"extras {r.extras_pct:.1%}  steal att/team-game {r.steal_attempts_pg:.2f}  success {r.steal_success_pct:.0%}")
