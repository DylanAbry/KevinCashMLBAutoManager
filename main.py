import argparse
import sys
from datetime import date

from decisions.batting_order import order_lineup
from decisions.lineup import build_lineup
from models.matchup import neutral_pitcher
from sim.engine import simulate
from sim.setup import make_side


def load_live(args):
    from data import mlb_api
    team_id = mlb_api.get_team_id(args.team, args.season)
    opp_id = mlb_api.get_team_id(args.opponent, args.season)
    info = mlb_api.find_matchup_starters(team_id, opp_id)
    opp_name = args.pitcher or info["opp_pitcher"]
    our_name = args.our_pitcher or info["our_pitcher"]
    if not opp_name:
        sys.exit('Could not find the opponent\'s probable starter - pass one with --pitcher "Full Name"')
    ours = mlb_api.get_hitters(team_id, args.season)
    theirs = None if args.no_sim else mlb_api.get_hitters(opp_id, args.season)
    opp_p = mlb_api.get_pitcher(mlb_api.find_player_id(opp_name), args.season)
    our_p = mlb_api.get_pitcher(mlb_api.find_player_id(our_name), args.season) if our_name and not args.no_sim else None
    return ours, theirs, opp_p, our_p, info["home"]


def load_demo():
    from data.demo import demo_hitters, demo_pitcher
    return (demo_hitters(7, "Home"), demo_hitters(11, "Rival"), demo_pitcher("Demo Lefty", "L"),
            demo_pitcher("Our Righty", "R"), True)


def print_roster(hitters) -> None:
    print(f"\n{'Roster data':<24}{'Pos':<5}{'PA':>5}{'wOBA':>7}{'xwOBA':>7}{'RISP OPS':>9}{'SB-CS':>7}  Defense (runs, season)")
    for h in sorted(hitters, key=lambda x: -x.pa):
        w = f"{h.woba:.3f}" if h.woba is not None else "  n/a"
        x = f"{h.xwoba:.3f}" if h.xwoba is not None else "  n/a"
        r = f"{h.risp[0]:.3f}" if h.risp else "  n/a"
        sb = f"{int(h.season.get('sb', 0))}-{int(h.season.get('cs', 0))}" if h.season else "n/a"
        d = ", ".join(f"{p} {v:+.1f}" for p, v in h.def_runs.items()) or "none loaded"
        print(f"{h.name:<24}{h.pos:<5}{h.pa:>5}{w:>7}{x:>7}{r:>9}{sb:>7}  {d}")


def print_lineup(title: str, lineup) -> None:
    print(f"\n{title}\n")
    print(f"{'#':<3}{'Player':<24}{'Pos':<5}{'Bats':<5}{'xwOBA*':>7}{'PA':>5}{'Def/162':>9}")
    for s in lineup.spots:
        print(f"{s.slot:<3}{s.hitter.name:<24}{s.field_pos:<5}{s.hitter.bat_side:<5}"
              f"{s.woba:>7.3f}{s.hitter.pa:>5}{s.def_rpg * 162:>+9.1f}")
    print("\n* expected wOBA vs this starter (talent blend of wOBA/xwOBA, shrunk by PA, plus platoon/pitcher effects)")
    print(f"Lineup value: offense {lineup.off_rpg:+.2f} runs/game, defense {lineup.def_rpg:+.2f} runs/game")
    print(lineup.dh_note)


def main():
    ap = argparse.ArgumentParser(description="MLB AutoManager")
    ap.add_argument("--team", help="Your team, e.g. NYM")
    ap.add_argument("--opponent", help="Opposing team, e.g. PIT")
    ap.add_argument("--pitcher", help="Opposing starter's full name (auto-detected if omitted)")
    ap.add_argument("--our-pitcher", dest="our_pitcher", help="Your starter's full name (auto-detected if omitted)")
    ap.add_argument("--venue", choices=["home", "away"], help="Override home/away (default: from the schedule, else home)")
    ap.add_argument("--pin", action="append", default=[],
                    help='Force a starter into a batting slot, e.g. --pin "Bichette:2" (repeatable)')
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--sims", type=int, default=10000, help="Games to simulate per lineup")
    ap.add_argument("--no-sim", action="store_true", help="Skip the simulation")
    ap.add_argument("--show-data", action="store_true", help="Print the data loaded for every hitter on your roster")
    ap.add_argument("--demo", action="store_true", help="Use synthetic data (no network)")
    args = ap.parse_args()

    if args.demo:
        ours, theirs, opp_p, our_p, we_home = load_demo()
    else:
        if not (args.team and args.opponent):
            ap.error("--team and --opponent are required unless --demo is used")
        ours, theirs, opp_p, our_p, we_home = load_live(args)
    if args.venue:
        we_home = args.venue == "home"
    if args.show_data:
        print_roster(ours)

    our_starter = our_p or neutral_pitcher("R")
    opp_lineup = build_lineup(theirs, our_starter) if theirs else None
    opp_def = opp_lineup.def_rpg if opp_lineup else 0.0

    picked = build_lineup(ours, opp_p)                                   # who plays where
    best, info = order_lineup(picked, opp_p, opp_def, args.pin)          # who bats where
    print_lineup(f"Recommended lineup vs {opp_p.name} ({opp_p.throws}HP)", best)

    print(f"\nBatting order: exact expected runs over 9 innings = {info['runs']:.3f} "
          f"({(info['runs'] - info['heuristic_runs']) * 162:+.1f} runs/162 vs a simple best-hitters-first order)")
    if args.pin:
        cost = (info["free_runs"] - info["runs"]) * 162
        print(f"Your pinned slots cost about {cost:.1f} runs per 162 games versus the unconstrained best order "
              f"({cost / 10:.1f} wins).")

    if args.no_sim or theirs is None:
        return
    baseline = build_lineup(ours, opp_p, defense_weight=0.0)             # offense-first, ignores glove quality
    print(f"\nSimulating {args.sims:,} games each ({'home' if we_home else 'away'}) ...")
    rows = []
    for label, lu in (("Recommended lineup", best), ("Offense-first baseline", baseline)):
        us = make_side("Us", lu.spots, opp_p, opp_lineup.def_rpg)
        them = make_side("Them", opp_lineup.spots, our_starter, lu.def_rpg)
        rows.append((label, simulate(us, them, we_home, args.sims, seed=42)))
    print(f"\n{'':<24}{'Runs/G':>8}{'Opp R/G':>9}{'Win %':>8}{'Extras':>8}{'SB att/G':>10}")
    for label, r in rows:
        print(f"{label:<24}{r.runs_for:>8.2f}{r.runs_against:>9.2f}{r.win_pct * 100:>7.1f}%"
              f"{r.extras_pct * 100:>7.1f}%{r.steal_attempts_pg:>10.2f}")
    print("\nSame random seed for both runs; the simulation includes home-field advantage, steals, errors, RISP"
          " and extra innings. Win% differences under ~1 point are within simulation noise.")


if __name__ == "__main__":
    main()
