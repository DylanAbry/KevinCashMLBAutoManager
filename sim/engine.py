"""Plate-appearance-level game simulator built on GameState.

What it models: player-specific outcome profiles, matchup-adjusted wOBA, RISP hitting, stolen-base attempts,
reached-on-error, home-field advantage, extra innings with the automatic runner, ballpark factors, pitching
changes (times-through-the-order, fatigue, rest, three-batter minimum), intentional walks, sacrifice bunts,
pinch hitters, and defensive replacements.

Not modeled yet: hit-and-run, pitchouts, wild pitches/passed balls, injuries.
Run `python -m sim.engine` for a calibration check against real-MLB benchmarks.
"""
import random
from bisect import bisect
from dataclasses import dataclass, field

from decisions.strategy import choose_defensive_sub, choose_pinch_hitter, should_bunt, should_ibb
from models.game_state import GameState
from sim.baserunning import bunt_transition_table, transition_table
from sim.pa_model import BB, sample_outcome
from sim.types import NO_TTO, Ctx, InningsPolicy, PitcherModel, TeamSide, league_batter

HOME_FIELD_WOBA_EDGE = 0.006      # added to home hitters' wOBA and subtracted from road hitters' (~54% home win%)
MAX_INNINGS = 25
PITCHES_IBB = 1.0
PITCHES_BUNT = 2.6
# average pitches thrown by PA outcome: K, OUT, BB, 1B, 2B, 3B, HR, ROE
PITCHES = (4.8, 3.4, 5.6, 3.7, 3.7, 3.7, 3.6, 3.6)


class _Side:
    """One team's mutable state for a single game: batters/bench can change (pinch hit, defensive sub);
    staff/pitching state tracks who's on the mound and how used up the bullpen is."""

    def __init__(self, team: TeamSide):
        self.team = team
        self.batters = list(team.batters)                  # per-game copy: substitutions must not leak between games
        self.bench = list(team.bench) if team.bench else []
        self.bench_used: set[int] = set()
        self.style = team.style
        self.def_bonus = 0.0                                # wOBA taken off opponents while this team is fielding
        self.staff = team.staff or _legacy_staff(team)
        self.policy = team.policy or InningsPolicy(team.starter_innings)
        self.cur, self.pc, self.bf = 0, 0.0, 0
        self.used_pitchers = {0}
        self.faced = [[0] * 9 for _ in self.staff]
        self.runs_app = self.runs_game = 0
        self.outs_by = [0] * len(self.staff)


def _legacy_staff(team: TeamSide) -> list:
    """Two-pitcher staff (starter, generic bullpen) built from a side's legacy wOBA fields."""
    return [
        PitcherModel("Starter", "R", True, [b.woba_starter for b in team.batters], tto=NO_TTO, fatigue_per_pitch=0.0),
        PitcherModel("Bullpen", "R", False, [b.woba_pen for b in team.batters], tto=NO_TTO, fatigue_per_pitch=0.0),
    ]


def _maybe_defensive_sub(fld: _Side, inning: int, lead: int, rng, counters: list) -> None:
    pick = choose_defensive_sub(fld.style, fld.bench, fld.bench_used, 0, inning, lead)
    if pick is None:
        return
    i, gain162 = pick
    sub = fld.bench[i]
    for k, b in enumerate(fld.batters):
        if sub.positions and b.field_pos in sub.positions and gain162 > b.def_rpg162:
            fld.bench_used.add(i)
            fld.batters[k] = sub.as_batter()
            fld.def_bonus += (gain162 - b.def_rpg162) / 162 * 1.20 / 4.3     # -> wOBA points/PA, same scale as elsewhere
            counters[5] += 1
            break


def _maybe_pinch_hit(bat: _Side, idx: int, pitcher_throws: str, inning: int, li: float, lead: int, counters: list) -> None:
    pick = choose_pinch_hitter(bat.style, bat.bench, bat.bench_used, idx, bat.batters, pitcher_throws, inning, li, lead)
    if pick is not None:
        i, _ = pick
        bat.bench_used.add(i)
        bat.batters[idx] = bat.bench[i].as_batter()
        counters[4] += 1


def _half_inning(s: GameState, bat: _Side, fld: _Side, rng, counters: list) -> None:
    s.outs = 0
    idx = s.home_batter if s.batting_home else s.away_batter
    run = [None, None, None]
    if s.inning >= 10:
        run[1] = (idx - 1) % 9
    s.bases = tuple(r is not None for r in run)
    fld_home = s.top
    edge = HOME_FIELD_WOBA_EDGE if s.batting_home else -HOME_FIELD_WOBA_EDGE
    first = True
    _maybe_defensive_sub(fld, s.inning, (s.home_score - s.away_score) if fld_home else (s.away_score - s.home_score),
                         rng, counters)
    while s.outs < 3:
        idx = s.home_batter if s.batting_home else s.away_batter
        mask = (run[0] is not None) + 2 * (run[1] is not None) + 4 * (run[2] is not None)
        lead = (s.home_score - s.away_score) if fld_home else (s.away_score - s.home_score)
        if first or fld.bf >= 3:
            new = fld.policy.choose(Ctx(s.inning, s.outs, mask, lead, idx, first, fld.cur, fld.pc, fld.bf,
                                        fld.runs_app, fld.runs_game, fld.faced[fld.cur], fld.staff, fld.used_pitchers,
                                        fld_home))
            if new is not None and new not in fld.used_pitchers:
                fld.cur, fld.pc, fld.bf, fld.runs_app = new, 0.0, 0, 0
                fld.used_pitchers.add(new)
        li = None
        if not first:
            from decisions.bullpen import leverage
            li = leverage(s.inning, lead, mask, s.outs, fld_home)
            _maybe_pinch_hit(bat, idx, fld.staff[fld.cur].throws, s.inning, li, -lead, counters)
        first = False
        j = fld.cur
        p = fld.staff[j]
        o0 = s.outs
        r1 = run[0]
        if r1 is not None and run[1] is None:
            rb = bat.batters[r1]
            if rng.random() < rb.steal_attempt:
                if rng.random() < rb.steal_success:
                    run[1], run[0] = r1, None
                    counters[0] += 1
                else:
                    run[0] = None
                    s.outs += 1
                    counters[1] += 1
                    if s.outs >= 3:
                        fld.outs_by[j] += s.outs - o0
                        break
        mask = (run[0] is not None) + 2 * (run[1] is not None) + 4 * (run[2] is not None)
        b = bat.batters[idx]
        target = p.woba_vs[idx] + p.penalty(fld.faced[j][idx], fld.pc) + edge - fld.def_bonus
        nxt = bat.batters[(idx + 1) % 9]
        nxt_target = p.woba_vs[(idx + 1) % 9] + p.penalty(fld.faced[j][(idx + 1) % 9], fld.pc) + edge - fld.def_bonus
        if should_ibb(fld.style, s.inning, s.outs, mask, lead, target, nxt_target):
            cum, dlist = transition_table(mask, s.outs, BB)
            dests, pitches, outcome_for_faced = dlist[bisect(cum, rng.random())], PITCHES_IBB, BB
            counters[3] += 1
        elif should_bunt(bat.style, s.inning, s.outs, mask, -lead, target + (b.risp_delta if mask & 6 else 0.0)):
            cum, dlist = bunt_transition_table(mask, s.outs)
            dests, pitches, outcome_for_faced = dlist[bisect(cum, rng.random())], PITCHES_BUNT, None
            counters[2] += 1
        else:
            risp = run[1] is not None or run[2] is not None
            target += b.risp_delta if risp else 0.0
            outcome = sample_outcome(b.profile, target, rng)
            cum, dlist = transition_table(mask, s.outs, outcome)
            dests, pitches, outcome_for_faced = dlist[bisect(cum, rng.random())], PITCHES[outcome], outcome
        who = (idx, run[0], run[1], run[2])
        new_run, runs = [None, None, None], 0
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
        if outcome_for_faced is not None:
            fld.faced[j][idx] += 1
        fld.bf += 1
        fld.pc += pitches
        fld.runs_app += runs
        fld.runs_game += runs
        fld.outs_by[j] += s.outs - o0
        if s.batting_home:
            s.home_score += runs
            s.home_batter = (idx + 1) % 9
            if s.inning >= 9 and s.home_score > s.away_score:
                return
        else:
            s.away_score += runs
            s.away_batter = (idx + 1) % 9


def simulate_game(home: TeamSide, away: TeamSide, rng, counters: list | None = None):
    """Returns (final GameState, home _Side, away _Side)."""
    counters = counters if counters is not None else [0, 0, 0, 0, 0, 0]
    hs, aws = _Side(home), _Side(away)
    s = GameState()
    while s.inning <= MAX_INNINGS:
        s.top = True
        _half_inning(s, aws, hs, rng, counters)
        if s.inning >= 9 and s.home_score > s.away_score:
            break
        s.top = False
        _half_inning(s, hs, aws, rng, counters)
        if s.inning >= 9 and s.home_score != s.away_score:
            break
        s.inning += 1
    return s, hs, aws


@dataclass
class SimResult:
    games: int
    win_pct: float
    runs_for: float
    runs_against: float
    extras_pct: float
    steal_attempts_pg: float
    steal_success_pct: float
    starter_outs: float = 0.0
    relievers_used: float = 0.0
    usage: list = field(default_factory=list)
    bunts_pg: float = 0.0
    ibbs_pg: float = 0.0
    pinch_hits_pg: float = 0.0
    def_subs_pg: float = 0.0


def simulate(ours: TeamSide, theirs: TeamSide, we_are_home: bool, n: int = 5000, seed: int = 1) -> SimResult:
    rng = random.Random(seed)
    home, away = (ours, theirs) if we_are_home else (theirs, ours)
    counters = [0, 0, 0, 0, 0, 0]
    wins = rf = ra = extras = 0
    starter_outs = relievers = 0
    k = len(ours.staff) if ours.staff else 0
    games_used, outs_used = [0] * k, [0] * k
    for _ in range(n):
        s, hs, aws = simulate_game(home, away, rng, counters)
        our_side, opp_side = (hs, aws) if we_are_home else (aws, hs)
        our, opp = (s.home_score, s.away_score) if we_are_home else (s.away_score, s.home_score)
        wins += our > opp
        rf += our
        ra += opp
        extras += s.inning > 9
        starter_outs += our_side.outs_by[0]
        relievers += len(our_side.used_pitchers) - 1
        for j in our_side.used_pitchers:
            if j < k:
                games_used[j] += 1
                outs_used[j] += our_side.outs_by[j]
    sb, cs, bunts, ibbs, pinch, defsub = counters
    att = sb + cs
    usage = [(ours.staff[j].name, games_used[j] / n, outs_used[j] / games_used[j] if games_used[j] else 0.0)
             for j in range(k)]
    return SimResult(n, wins / n, rf / n, ra / n, extras / n, att / (2 * n), sb / att if att else 0.0,
                     starter_outs / n, relievers / n, usage, bunts / (2 * n), ibbs / (2 * n),
                     pinch / n, defsub / n)


if __name__ == "__main__":
    avg = TeamSide("League avg", [league_batter()] * 9)
    r = simulate(avg, avg, True, n=20000)
    print(f"League-average vs league-average, {r.games:,} games (real MLB: ~4.4-4.5 runs/team, home win% ~.535,"
          f" ~9% extras, ~0.9 steal attempts/team, ~78% steal success):")
    print(f"  runs/game home {r.runs_for:.2f}  away {r.runs_against:.2f}  home win% {r.win_pct:.3f}  "
          f"extras {r.extras_pct:.1%}  steal att/team-game {r.steal_attempts_pg:.2f}  success {r.steal_success_pct:.0%}")