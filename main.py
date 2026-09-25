import argparse
import sys
from dataclasses import dataclass, field
from datetime import date

from data.names import find_by_name, norm
from decisions.batting_order import order_lineup
from decisions.bullpen import AutoPolicy, TypicalPolicy
from decisions.lineup import build_lineup, draft_lineup
from models.matchup import neutral_pitcher
from sim.engine import simulate
from sim.setup import make_side, make_staff
from sim.types import Ctx


@dataclass
class Inputs:
    ours: list                       # our hitters (active roster)
    theirs: list | None              # their hitters
    opp_starter: object              # their probable starter (Pitcher)
    our_starter: object | None       # our starter (Pitcher) or None
    our_pen: list = field(default_factory=list)
    opp_pen: list = field(default_factory=list)
    we_home: bool = True


def _is_reliever(p) -> bool:
    g, gs = p.season.get("g", 0), p.season.get("gs", 0)
    return not (g and gs / g >= 0.5)


def _pick_pitcher(pitchers, name, season):
    from data import mlb_api
    q = norm(name)
    hits = [p for p in pitchers if q in norm(p.name)]
    if len(hits) == 1:
        return hits[0]
    return mlb_api.get_pitcher(mlb_api.find_player_id(name), season)     # not on the active roster


def load_live(args) -> Inputs:
    from data import mlb_api
    team_id = mlb_api.get_team_id(args.team, args.season)
    opp_id = mlb_api.get_team_id(args.opponent, args.season)
    info = mlb_api.find_matchup_starters(team_id, opp_id)
    opp_name = args.pitcher or info["opp_pitcher"]
    our_name = args.our_pitcher or info["our_pitcher"]
    if not opp_name:
        sys.exit('Could not find the opponent\'s probable starter - pass one with --pitcher "Full Name"')
    ours = mlb_api.get_hitters(team_id, args.season)
    if args.no_sim:
        opp_p = mlb_api.get_pitcher(mlb_api.find_player_id(opp_name), args.season)
        return Inputs(ours, None, opp_p, None, we_home=info["home"])
    theirs = mlb_api.get_hitters(opp_id, args.season)
    opp_arms = mlb_api.get_pitchers(opp_id, args.season)
    our_arms = mlb_api.get_pitchers(team_id, args.season)
    opp_starter = _pick_pitcher(opp_arms, opp_name, args.season)
    our_starter = _pick_pitcher(our_arms, our_name, args.season) if our_name else None
    if not our_starter:
        print(f'[warn] no starter found for {args.team}; using a league-average starter. Pass --our-pitcher "Name".',
              file=sys.stderr)
    return Inputs(
        ours, theirs, opp_starter, our_starter,
        [p for p in our_arms if _is_reliever(p) and (not our_starter or p.id != our_starter.id)],
        [p for p in opp_arms if _is_reliever(p) and p.id != opp_starter.id],
        info["home"])


def load_demo() -> Inputs:
    from data.demo import demo_hitters, demo_pitcher, demo_staff
    our_sp, our_pen = demo_staff(2, "Home", "Our Righty", "R")
    opp_sp, opp_pen = demo_staff(3, "Rival", "Demo Lefty", "L")
    return Inputs(demo_hitters(7, "Home"), demo_hitters(11, "Rival"), opp_sp, our_sp, our_pen, opp_pen, True)


def print_roster(hitters) -> None:
    print(f"\n{'Roster data':<24}{'Pos':<5}{'PA':>5}{'wOBA':>7}{'xwOBA':>7}{'RISP OPS':>9}{'SB-CS':>7}  Defense (runs, season)")
    for h in sorted(hitters, key=lambda x: -x.pa):
        w = f"{h.woba:.3f}" if h.woba is not None else "  n/a"
        x = f"{h.xwoba:.3f}" if h.xwoba is not None else "  n/a"
        r = f"{h.risp[0]:.3f}" if h.risp else "  n/a"
        sb = f"{int(h.season.get('sb', 0))}-{int(h.season.get('cs', 0))}" if h.season else "n/a"
        d = ", ".join(f"{p} {v:+.1f}" for p, v in h.def_runs.items()) or "none loaded"
        print(f"{h.name:<24}{h.pos:<5}{h.pa:>5}{w:>7}{x:>7}{r:>9}{sb:>7}  {d}")


def print_lineup(title: str, lineup, footnote: bool = True) -> None:
    print(f"\n{title}\n")
    print(f"{'#':<3}{'Player':<24}{'Pos':<5}{'Bats':<5}{'xwOBA*':>7}{'PA':>5}{'Def/162':>9}")
    for s in lineup.spots:
        print(f"{s.slot:<3}{s.hitter.name:<24}{s.field_pos:<5}{s.hitter.bat_side:<5}"
              f"{s.woba:>7.3f}{s.hitter.pa:>5}{s.def_rpg * 162:>+9.1f}")
    if footnote:
        print("\n* expected wOBA vs this starter (talent blend of wOBA/xwOBA, shrunk by PA, plus platoon/pitcher effects)")
    print(f"Lineup value: offense {lineup.off_rpg:+.2f} runs/game, defense {lineup.def_rpg:+.2f} runs/game")
    print(lineup.dh_note)


def print_staff(title: str, staff) -> None:
    print(f"\n{title}\n")
    print(f"{'Pitcher':<26}{'Thr':<5}{'Role':<6}{'wOBA allowed':>13}  Status")
    for m in staff:
        status = "starter" if m.is_starter else ("available - " + m.note if m.available else "UNAVAILABLE - " + m.note)
        print(f"{m.name:<26}{m.throws:<5}{m.role:<6}{m.quality():>13.3f}  {status}")


def print_usage(res, staff) -> None:
    print("\nProjected bullpen usage under the AutoManager plan (simulated):")
    print(f"  starter averages {res.starter_outs / 3:.1f} innings; {res.relievers_used:.1f} relievers used per game")
    print(f"  {'Pitcher':<26}{'Role':<6}{'Used in':>9}{'Avg outs':>10}")
    for m, (name, pct, outs) in zip(staff, res.usage):
        if pct >= 0.01:
            print(f"  {name:<26}{m.role:<6}{pct * 100:>8.0f}%{outs:>10.1f}")


def parse_situation(text: str) -> dict:
    d = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        d[k.strip().lower()] = v.strip()
    return d


def advise(text: str, staff, we_home: bool) -> None:
    """Explain what the AutoManager would do in a specific game situation."""
    sit = parse_situation(text)
    runners = sit.get("runners", "0")
    mask = sum({"1": 1, "2": 2, "3": 4}.get(ch, 0) for ch in runners if ch in "123")
    cur = 0
    if sit.get("current"):
        cur = staff.index(find_by_name(staff, sit["current"], "our pitching staff"))
    used = {cur, 0}
    for nm in filter(None, (x.strip() for x in sit.get("used", "").split(";"))):
        used.add(staff.index(find_by_name(staff, nm, "our pitching staff")))
    outs = int(sit.get("outs", 0))
    bf = int(sit.get("bf", 9 if cur == 0 else 0))
    first = sit.get("newinning", "1" if outs == 0 and mask == 0 else "0") == "1"
    ctx = Ctx(int(sit.get("inning", 7)), outs, mask, int(sit.get("lead", 0)), (int(sit.get("batter", 1)) - 1) % 9, first,
              cur, float(sit.get("pitches", 0)), bf, int(sit.get("runs", 0)), int(sit.get("gameruns", 0)),
              [int(sit.get("tto", 1))] * 9, staff, used, we_home)
    pol = AutoPolicy()
    r = pol.evaluate(ctx)
    cur_m = staff[cur]
    print(f"\nSituation: inning {ctx.inning}, {outs} out, runners {runners or '-'}, our lead {ctx.lead:+d}, "
          f"{cur_m.name} at {ctx.pc:.0f} pitches, hitter #{ctx.batter + 1} due up")
    print(f"Leverage {r['li']:.2f}; {cur_m.name} is expected to allow {r['w_cur']:.3f} wOBA to the next three hitters")
    if not (first or bf >= 3):
        print(f"Three-batter minimum: {cur_m.name} must face {3 - bf} more hitter(s) unless the inning ends.")
    print(f"\n{'Option':<26}{'Role':<6}{'wOBA next 3':>12}{'Needs LI':>10}  Notes")
    for o in r["options"]:
        m = staff[o["j"]]
        note = "eligible" if o["eligible"] else "held back for higher leverage"
        print(f"{m.name:<26}{m.role:<6}{o['look']:>12.3f}{o['thr']:>10.2f}  {note}")
    if r["action"] == "pull" and (first or bf >= 3):
        print(f"\nRECOMMENDATION: bring in {staff[r['pick']].name} - {r['reason']}.")
    else:
        why = r["reason"] or "no better option"
        print(f"\nRECOMMENDATION: stay with {cur_m.name} - {why}.")


def main():
    ap = argparse.ArgumentParser(description="MLB AutoManager")
    ap.add_argument("--team", help="Your team, e.g. NYM")
    ap.add_argument("--opponent", help="Opposing team, e.g. PIT")
    ap.add_argument("--pitcher", "--opp-pitcher", dest="pitcher",
                    help="Opposing starter's full name (auto-detected if omitted)")
    ap.add_argument("--our-pitcher", dest="our_pitcher", help="Your starter's full name (auto-detected if omitted)")
    ap.add_argument("--opp-lineup", dest="opp_lineup",
                    help='Opposing lineup, 9 names in batting order, comma-separated (default: projected optimal lineup)')
    ap.add_argument("--start", action="append", default=[], help='Force a hitter into your starting nine (repeatable)')
    ap.add_argument("--bench", action="append", default=[], help='Keep a hitter out of your lineup (repeatable)')
    ap.add_argument("--pin", action="append", default=[],
                    help='Force a hitter into a batting slot, e.g. --pin "Bichette:2" (also makes him start)')
    ap.add_argument("--unavailable", default="", help='Your relievers who cannot pitch, comma-separated')
    ap.add_argument("--opp-unavailable", dest="opp_unavailable", default="", help="Their relievers who cannot pitch")
    ap.add_argument("--situation", help='Ask what the AutoManager would do mid-game, e.g. '
                    '"inning=7,outs=1,runners=13,lead=1,pitches=88,batter=4,tto=2"')
    ap.add_argument("--venue", choices=["home", "away"], help="Override home/away (default: from the schedule, else home)")
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--sims", type=int, default=6000, help="Games to simulate per scenario")
    ap.add_argument("--no-sim", action="store_true", help="Lineup only: skip the opponent, bullpen and simulation")
    ap.add_argument("--show-data", action="store_true", help="Print the data loaded for every hitter on your roster")
    ap.add_argument("--demo", action="store_true", help="Use synthetic data (no network)")
    args = ap.parse_args()

    if args.demo:
        inp = load_demo()
    else:
        if not (args.team and args.opponent):
            ap.error("--team and --opponent are required unless --demo is used")
        inp = load_live(args)
    if args.no_sim:
        inp.theirs = None
    we_home = (args.venue == "home") if args.venue else inp.we_home
    if args.show_data:
        print_roster(inp.ours)

    # --- who is available, who must start -------------------------------------------------------------
    roster_label = "your active roster (26-man)"
    benched = {find_by_name(inp.ours, n, roster_label).id for n in args.bench}
    forced_names = args.start + [p.rpartition(":")[0] or p for p in args.pin]
    forced = {find_by_name(inp.ours, n, roster_label).id for n in forced_names}
    if forced & benched:
        sys.exit("A hitter can't be both benched and forced to start")
    ours = [h for h in inp.ours if h.id not in benched]

    our_starter = inp.our_starter or neutral_pitcher("R")
    opp_lineup = None
    if inp.theirs:
        names = [n.strip() for n in args.opp_lineup.split(",") if n.strip()] if args.opp_lineup else None
        opp_lineup = draft_lineup(inp.theirs, names, our_starter) if names else build_lineup(inp.theirs, our_starter)
    opp_def = opp_lineup.def_rpg if opp_lineup else 0.0

    picked = build_lineup(ours, inp.opp_starter, forced=forced)            # who plays where
    best, info = order_lineup(picked, inp.opp_starter, opp_def, args.pin)  # who bats where
    print_lineup(f"Recommended lineup vs {inp.opp_starter.name} ({inp.opp_starter.throws}HP)", best)
    print(f"\nBatting order: exact expected runs over 9 innings = {info['runs']:.3f} "
          f"({(info['runs'] - info['heuristic_runs']) * 162:+.1f} runs/162 vs a simple best-hitters-first order)")
    if args.pin:
        cost = (info["free_runs"] - info["runs"]) * 162
        print(f"Your pinned slots cost about {cost:.1f} runs per 162 games versus the unconstrained best order "
              f"({cost / 10:.1f} wins).")
    if opp_lineup is None:
        return

    opp_label = "Opposing lineup (drafted by you)" if args.opp_lineup else "Opposing lineup (projected optimal)"
    print_lineup(f"{opp_label} vs {our_starter.name}", opp_lineup, footnote=False)

    def staffs(lu):
        ours_s = make_staff(inp.our_starter, inp.our_pen, [s.hitter for s in opp_lineup.spots], lu.def_rpg,
                            args.unavailable.split(","))
        theirs_s = make_staff(inp.opp_starter, inp.opp_pen, [s.hitter for s in lu.spots], opp_lineup.def_rpg,
                              args.opp_unavailable.split(","))
        return ours_s, theirs_s

    our_staff, _ = staffs(best)
    print_staff(f"Our pitching staff vs the opposing lineup (starter: {our_starter.name})", our_staff)
    if args.situation:
        advise(args.situation, our_staff, we_home)
        return

    baseline = build_lineup(ours, inp.opp_starter, defense_weight=0.0, forced=forced)   # offense-first, ignores gloves
    print(f"\nSimulating {args.sims:,} games per scenario ({'home' if we_home else 'away'}) ...")
    scenarios = [("Lineup + AutoManager bullpen", best, AutoPolicy()),
                 ("Lineup + typical bullpen use", best, TypicalPolicy()),
                 ("Offense-first lineup + typical", baseline, TypicalPolicy())]
    rows = []
    for label, lu, policy in scenarios:
        ours_s, theirs_s = staffs(lu)
        us = make_side("Us", lu.spots, inp.opp_starter, opp_lineup.def_rpg, staff=ours_s, policy=policy)
        them = make_side("Them", opp_lineup.spots, our_starter, lu.def_rpg, staff=theirs_s, policy=TypicalPolicy())
        rows.append((label, simulate(us, them, we_home, args.sims, seed=42)))
    print(f"\n{'':<34}{'Runs/G':>8}{'Opp R/G':>9}{'Win %':>8}{'SP IP':>7}{'Relievers':>10}{'Extras':>8}")
    for label, r in rows:
        print(f"{label:<34}{r.runs_for:>8.2f}{r.runs_against:>9.2f}{r.win_pct * 100:>7.1f}%"
              f"{r.starter_outs / 3:>7.1f}{r.relievers_used:>10.1f}{r.extras_pct * 100:>7.1f}%")
    print_usage(rows[0][1], our_staff)
    print("\nSame random seed in every scenario. The opponent always uses typical bullpen management. The simulation"
          " includes home-field advantage, steals, errors, RISP, extra innings, times-through-the-order, fatigue and"
          " reliever rest. Win% differences under ~1 point are within simulation noise.")


if __name__ == "__main__":
    main()