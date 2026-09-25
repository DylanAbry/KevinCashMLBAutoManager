"""Plate-appearance-level game simulator built on GameState.

What it models: player-specific outcome profiles, matchup-adjusted wOBA, RISP hitting, stolen-base attempts by
the actual runner, reached-on-error, home-field advantage, extra innings with the automatic runner, and
PITCHING CHANGES: each side has a staff of pitchers (starter + relievers) with per-hitter matchup values,
times-through-the-order and fatigue penalties, rest-based tiredness, and a policy that decides when to change
pitchers (subject to the three-batter minimum).

Not modeled yet: pinch hitters, defensive replacements, pitchouts, wild pitches, bunts, intentional walks.
Run `python -m sim.engine` for a calibration check against real-MLB benchmarks.
"""
import random
from bisect import bisect
from dataclasses import dataclass, field

from models.game_state import GameState
from sim.baserunning import transition_table
from sim.pa_model import sample_outcome
from sim.types import NO_TTO, Ctx, InningsPolicy, PitcherModel, TeamSide, league_batter

HOME_FIELD_WOBA_EDGE = 0.006      # added to home hitters' wOBA and subtracted from road hitters' (~54% home win%)
MAX_INNINGS = 25
# average pitches thrown by PA outcome: K, OUT, BB, 1B, 2B, 3B, HR, ROE
PITCHES = (4.8, 3.4, 5.6, 3.7, 3.7, 3.7, 3.6, 3.6)


class _Fld:
    """Per-game pitching state for one team."""
    __slots__ = ("cur", "pc", "bf", "used", "faced", "runs_app", "runs_game", "outs_by")

    def __init__(self, staff):
        self.cur, self.pc, self.bf = 0, 0.0, 0
        self.used = {0}
        self.faced = [[0] * 9 for _ in staff]
        self.runs_app = self.runs_game = 0
        self.outs_by = [0] * len(staff)


def _legacy_staff(bat: TeamSide) -> list:
    """Two-pitcher staff (starter, generic bullpen) built from the batting side's legacy wOBA fields."""
    return [
        PitcherModel("Starter", "R", True, [b.woba_starter for b in bat.batters], tto=NO_TTO, fatigue_per_pitch=0.0),
        PitcherModel("Bullpen", "R", False, [b.woba_pen for b in bat.batters], tto=NO_TTO, fatigue_per_pitch=0.0),
    ]


def _half_inning(s: GameState, bat: TeamSide, staff: list, policy, ffs: _Fld, rng, counters: list) -> None:
    s.outs = 0
    idx = s.home_batter if s.batting_home else s.away_batter
    run = [None, None, None]                       # batting-order index of the runner on 1B / 2B / 3B
    if s.inning >= 10:
        run[1] = (idx - 1) % 9                     # automatic runner = last batter to make an out
    s.bases = tuple(r is not None for r in run)
    fld_home = s.top                               # the home team is in the field while the visitors bat
    edge = HOME_FIELD_WOBA_EDGE if s.batting_home else -HOME_FIELD_WOBA_EDGE
    batters = bat.batters
    first = True
    while s.outs < 3:
        idx = s.home_batter if s.batting_home else s.away_batter
        mask = (run[0] is not None) + 2 * (run[1] is not None) + 4 * (run[2] is not None)
        # pitching change? (three-batter minimum: only at the start of a half-inning or after 3 batters faced)
        if first or ffs.bf >= 3:
            lead = (s.home_score - s.away_score) if fld_home else (s.away_score - s.home_score)
            new = policy.choose(Ctx(s.inning, s.outs, mask, lead, idx, first, ffs.cur, ffs.pc, ffs.bf,
                                    ffs.runs_app, ffs.runs_game, ffs.faced[ffs.cur], staff, ffs.used, fld_home))
            if new is not None and new not in ffs.used:
                ffs.cur, ffs.pc, ffs.bf, ffs.runs_app = new, 0.0, 0, 0
                ffs.used.add(new)
        first = False
        j = ffs.cur
        o0 = s.outs
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
                        ffs.outs_by[j] += s.outs - o0
                        break
        b = batters[idx]
        risp = run[1] is not None or run[2] is not None
        mask = (run[0] is not None) + 2 * (run[1] is not None) + 4 * (run[2] is not None)
        p = staff[j]
        target = (p.woba_vs[idx] + p.penalty(ffs.faced[j][idx], ffs.pc) + edge
                  + (b.risp_delta if risp else 0.0))
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
        ffs.faced[j][idx] += 1
        ffs.bf += 1
        ffs.pc += PITCHES[outcome]
        ffs.runs_app += runs
        ffs.runs_game += runs
        ffs.outs_by[j] += s.outs - o0
        if s.batting_home:
            s.home_score += runs
            s.home_batter = (idx + 1) % 9
            if s.inning >= 9 and s.home_score > s.away_score:
                return                             # walk-off
        else:
            s.away_score += runs
            s.away_batter = (idx + 1) % 9


def simulate_game(home: TeamSide, away: TeamSide, rng, counters: list | None = None):
    """Returns (final GameState, home pitching state, away pitching state)."""
    counters = counters if counters is not None else [0, 0]
    h_staff = home.staff or _legacy_staff(away)
    a_staff = away.staff or _legacy_staff(home)
    h_pol = home.policy or InningsPolicy(home.starter_innings)
    a_pol = away.policy or InningsPolicy(away.starter_innings)
    hf, af = _Fld(h_staff), _Fld(a_staff)
    s = GameState()
    while s.inning <= MAX_INNINGS:
        s.top = True
        _half_inning(s, away, h_staff, h_pol, hf, rng, counters)
        if s.inning >= 9 and s.home_score > s.away_score:
            break
        s.top = False
        _half_inning(s, home, a_staff, a_pol, af, rng, counters)
        if s.inning >= 9 and s.home_score != s.away_score:
            break
        s.inning += 1
    return s, hf, af


@dataclass
class SimResult:
    games: int
    win_pct: float
    runs_for: float
    runs_against: float
    extras_pct: float            # share of games that went past 9 innings
    steal_attempts_pg: float     # attempts per team per game
    steal_success_pct: float
    starter_outs: float = 0.0    # average outs recorded by OUR starter
    relievers_used: float = 0.0  # average relievers used by OUR side
    usage: list = field(default_factory=list)   # (name, % of games used, avg outs when used) for OUR staff


def simulate(ours: TeamSide, theirs: TeamSide, we_are_home: bool, n: int = 5000, seed: int = 1) -> SimResult:
    rng = random.Random(seed)
    home, away = (ours, theirs) if we_are_home else (theirs, ours)
    counters = [0, 0]
    wins = rf = ra = extras = 0
    starter_outs = relievers = 0
    k = len(ours.staff) if ours.staff else 0
    games_used, outs_used = [0] * k, [0] * k
    for _ in range(n):
        s, hf, af = simulate_game(home, away, rng, counters)
        our, opp = (s.home_score, s.away_score) if we_are_home else (s.away_score, s.home_score)
        wins += our > opp
        rf += our
        ra += opp
        extras += s.inning > 9
        f = hf if we_are_home else af
        starter_outs += f.outs_by[0]
        relievers += len(f.used) - 1
        for j in f.used:
            if j < k:
                games_used[j] += 1
                outs_used[j] += f.outs_by[j]
    sb, cs = counters
    att = sb + cs
    usage = [(ours.staff[j].name, games_used[j] / n, outs_used[j] / games_used[j] if games_used[j] else 0.0)
             for j in range(k)]
    return SimResult(n, wins / n, rf / n, ra / n, extras / n, att / (2 * n), sb / att if att else 0.0,
                     starter_outs / n, relievers / n, usage)


if __name__ == "__main__":
    avg = TeamSide("League avg", [league_batter()] * 9)
    r = simulate(avg, avg, True, n=20000)
    print(f"League-average vs league-average, {r.games:,} games (real MLB: ~4.4-4.5 runs/team, home win% ~.535,"
          f" ~9% extras, ~0.9 steal attempts/team, ~78% steal success):")
    print(f"  runs/game home {r.runs_for:.2f}  away {r.runs_against:.2f}  home win% {r.win_pct:.3f}  "
          f"extras {r.extras_pct:.1%}  steal att/team-game {r.steal_attempts_pg:.2f}  success {r.steal_success_pct:.0%}")