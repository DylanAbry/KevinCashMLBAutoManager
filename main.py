# ============================================================================================
# MLB AUTOMANAGER - COMMAND REFERENCE
# ============================================================================================
# Base command:
#   python main.py --team <CODE> --opponent <CODE> [options below]
#   python main.py --demo                            (no network - synthetic roster/pitchers)
#
# --------------------------------------------------------------------------------------------
# ONE-SHOT FLAGS (set before running - these shape a single report, then the program exits)
# --------------------------------------------------------------------------------------------
#   --team CODE                 Your team, e.g. NYM                          [required unless --demo]
#   --opponent CODE              Opposing team, e.g. PIT                       [required unless --demo]
#   --pitcher "Name"             Opposing starter (auto-detected if omitted)   [alias: --opp-pitcher]
#   --our-pitcher "Name"         Your starter (auto-detected if omitted)
#   --opp-lineup "A,B,C,...,I"    Opposing lineup, 9 names in batting order (default: projected optimal)
#   --start "Name"               Force a hitter into your starting nine        [repeatable]
#   --bench "Name"                Keep a hitter OUT of your lineup              [repeatable]
#   --pin "Name:Slot"             Force a hitter into a batting slot (1-9), e.g. --pin "Bichette:2"
#                                  (also makes him start)                       [repeatable]
#   --unavailable "A,B,C"         Your relievers who can't pitch today
#   --opp-unavailable "A,B,C"     Their relievers who can't pitch today
#   --park CODE                  Ballpark for park factors, e.g. COL (default: the home team's park)
#   --style typical|auto|never    Your tactics in the two comparison scenarios (AutoManager scenario
#                                  always uses "auto" regardless of this flag)
#   --venue home|away             Override home/away (default: from the schedule, else home)
#   --season YYYY                 Season to pull stats for (default: current year)
#   --sims N                      Games to simulate per scenario (default: 6000)
#   --no-sim                      Lineup only - skip opponent/bullpen/simulation entirely
#   --show-data                   Print the full loaded data line for every hitter on your roster
#   --demo                        Use synthetic data - no network calls, good for testing
#
# --------------------------------------------------------------------------------------------
# ONE-SHOT MID-GAME QUESTION (still exits after printing the answer)
# --------------------------------------------------------------------------------------------
#   --situation "inning=7,outs=1,runners=13,lead=1,pitches=88,batter=4,tto=2"
#       Asks what the AutoManager would do RIGHT NOW, without starting a persistent session.
#       Recognized keys: inning, outs, runners (digits 1/2/3 = occupied, e.g. "13"), lead,
#       batter (1-9), pitches, bf, runs, gameruns, tto, current (pitcher name), used (name;name),
#       newinning (1/0)
#
# --------------------------------------------------------------------------------------------
# INTERACTIVE / AGENTIC MODE  --  python main.py --team X --opponent Y --live
# --------------------------------------------------------------------------------------------
#   Starts a persistent session that remembers who's been used (bullpen, bench) and updates its
#   recommendations as you report what actually happened. Enter these at the "> " prompt:
#
#   sit inning=7,outs=1,runners=13,lead=1,batter=4,field=1[,first=1]
#       -> get a recommendation for this exact situation.
#          runners: which bases are occupied (e.g. "13" = 1st & 3rd). field=1 if WE are fielding
#          (pitching change / IBB / defensive sub apply); field=0 if WE are batting (bunt / pinch
#          hit apply). first=1 marks the first batter of the half-inning.
#   change <pitcher name>             -> record that this pitcher is now in the game
#   pinch <slot 1-9> <bench name>     -> record a pinch hitter entering at that batting slot
#   defsub <slot 1-9> <bench name>    -> record a defensive substitution at that batting slot
#   pc <pitches>                      -> update the current pitcher's pitch count
#   log                               -> show every situation asked about so far, with its advice
#   help                              -> show this command list
#   quit                              -> end the session
# ============================================================================================

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date

from data.ballparks import park_factor
from data.names import find_by_name, norm
from decisions.batting_order import order_lineup
from decisions.bullpen import AutoPolicy, TypicalPolicy
from decisions.lineup import build_lineup, draft_lineup
from models.matchup import neutral_pitcher, park_woba_shift
from sim.engine import simulate
from sim.setup import make_bench_player, make_side, make_staff
from sim.types import Ctx
from agent.session import Session


@dataclass
class Inputs:
    ours: list                       # our hitters (active roster)
    theirs: list | None              # their hitters
    opp_starter: object              # their probable starter (Pitcher)
    our_starter: object | None       # our starter (Pitcher) or None
    our_pen: list = field(default_factory=list)
    opp_pen: list = field(default_factory=list)
    we_home: bool = True
    team_id: int | None = None       # our MLB team id (for weather/fatigue lookups; None in --demo)
    opp_id: int | None = None        # their MLB team id


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
        return Inputs(ours, None, opp_p, None, we_home=info["home"], team_id=team_id, opp_id=opp_id)
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
        info["home"], team_id=team_id, opp_id=opp_id)


def load_demo() -> Inputs:
    from data.demo import demo_hitters, demo_pitcher, demo_staff
    our_sp, our_pen = demo_staff(2, "Home", "Our Righty", "R")
    opp_sp, opp_pen = demo_staff(3, "Rival", "Demo Lefty", "L")
    return Inputs(demo_hitters(7, "Home"), demo_hitters(11, "Rival"), opp_sp, our_sp, our_pen, opp_pen, True)


def print_roster(hitters) -> None:
    from models.matchup import gidp_rate
    print(f"\n{'Roster data':<24}{'Pos':<5}{'PA':>5}{'wOBA':>7}{'xwOBA':>7}{'RISP OPS':>9}{'SB-CS':>7}{'GIDP%':>7}"
          f"  Defense (runs, season - C is framing)")
    for h in sorted(hitters, key=lambda x: -x.pa):
        w = f"{h.woba:.3f}" if h.woba is not None else "  n/a"
        x = f"{h.xwoba:.3f}" if h.xwoba is not None else "  n/a"
        r = f"{h.risp[0]:.3f}" if h.risp else "  n/a"
        sb = f"{int(h.season.get('sb', 0))}-{int(h.season.get('cs', 0))}" if h.season else "n/a"
        g = f"{gidp_rate(h) * 100:.0f}%"      # the simulator's per-PA-opportunity double-play rate, not a raw GIDP count
        d = ", ".join(f"{p} {v:+.1f}" for p, v in h.def_runs.items()) or "none loaded"
        print(f"{h.name:<24}{h.pos:<5}{h.pa:>5}{w:>7}{x:>7}{r:>9}{sb:>7}{g:>7}  {d}")


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


def print_bench(title: str, bench) -> None:
    if not bench:
        return
    print(f"\n{title}\n")
    print(f"{'Player':<24}{'Bats':<5}{'Pos':<10}{'xwOBA':>7}{'Def/162':>9}")
    for b in bench:
        print(f"{b.name:<24}{b.bat_side:<5}{','.join(b.positions):<10}{b.woba_starter:>7.3f}{b.def_runs:>+9.1f}")


def print_usage(res, staff) -> None:
    print("\nProjected bullpen usage under the AutoManager plan (simulated):")
    print(f"  starter averages {res.starter_outs / 3:.1f} innings; {res.relievers_used:.1f} relievers used per game")
    print(f"  {'Pitcher':<26}{'Role':<6}{'Used in':>9}{'Avg outs':>10}")
    for m, (name, pct, outs) in zip(staff, res.usage):
        if pct >= 0.01:
            print(f"  {name:<26}{m.role:<6}{pct * 100:>8.0f}%{outs:>10.1f}")
    print(f"\nPer team-game: {res.bunts_pg:.2f} sac bunts, {res.ibbs_pg:.2f} intentional walks, "
          f"{res.pinch_hits_pg:.2f} pinch hits, {res.def_subs_pg:.2f} defensive substitutions.")


def parse_situation(text: str) -> dict:
    d = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        d[k.strip().lower()] = v.strip()
    return d


def advise(text: str, staff, our_batters, our_bench, we_home: bool) -> None:
    """Explain what the AutoManager would do in a specific game situation. field=1 (the default, for
    backward compatibility with scripts that never pass it) means WE are fielding - evaluates a pitching
    change, exactly as before. field=0 means WE are batting - evaluates a sacrifice bunt / pinch hitter
    instead, using the same should_bunt()/choose_pinch_hitter() logic --live's "sit" command uses. This
    is a one-shot query (unlike --live's persistent Session): no bench players or relievers are assumed
    used yet unless you say so with "used=" (pitching side only - there's no bench equivalent here; use
    --live if you need to track who's already been used for both pitching AND the bench)."""
    from decisions.bullpen import leverage
    from decisions.strategy import choose_pinch_hitter, should_bunt
    sit = parse_situation(text)
    runners = sit.get("runners", "0")
    mask = sum({"1": 1, "2": 2, "3": 4}.get(ch, 0) for ch in runners if ch in "123")
    inning = int(sit.get("inning", 7))
    outs = int(sit.get("outs", 0))
    lead = int(sit.get("lead", 0))
    idx = (int(sit.get("batter", 1)) - 1) % 9
    fielding = sit.get("field", "1") == "1"

    if not fielding:
        li = leverage(inning, -lead, mask, outs, False)
        print(f"\nSituation: inning {inning}, {outs} out, runners {runners or '-'}, our lead {lead:+d}, "
              f"hitter #{idx + 1} due up (we're batting)")
        print(f"Leverage {li:.2f}")
        batter_woba = our_batters[idx].woba_starter
        bunt = should_bunt("auto", inning, outs, mask, lead, batter_woba)
        pick = choose_pinch_hitter("auto", our_bench, set(), idx, our_batters, "R", inning, li, lead)
        if bunt:
            print("\nRECOMMENDATION: call for a sacrifice bunt.")
        elif pick:
            name, gain = our_bench[pick[0]].name, pick[1]
            print(f"\nRECOMMENDATION: pinch hit with {name} (+{gain:.3f} wOBA edge).")
        else:
            print("\nRECOMMENDATION: no move - let the current batter hit.")
        return

    cur = 0
    if sit.get("current"):
        cur = staff.index(find_by_name(staff, sit["current"], "our pitching staff"))
    used = {cur, 0}
    for nm in filter(None, (x.strip() for x in sit.get("used", "").split(";"))):
        used.add(staff.index(find_by_name(staff, nm, "our pitching staff")))
    bf = int(sit.get("bf", 9 if cur == 0 else 0))
    first = sit.get("newinning", "1" if outs == 0 and mask == 0 else "0") == "1"
    ctx = Ctx(inning, outs, mask, lead, idx, first,
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


def _fmt_rec(rec: dict) -> str:
    lines = [f"Inning {rec['inning']}, {rec['outs']} out, runners {rec['runners'] or '-'}, lead {rec['lead']:+d}, "
             f"batter #{rec['batter']} due up (leverage {rec['leverage']:.2f})"]
    if "pitching" in rec:
        p = rec["pitching"]
        if p["action"] == "pull":
            lines.append(f"  PITCHING: bring in staff option — {p['reason']}")
        else:
            lines.append(f"  PITCHING: stay — {p['reason']}")
        if rec.get("ibb"):
            lines.append("  IBB: intentionally walk the batter due up")
        if rec.get("defensive_sub"):
            name, gain = rec["defensive_sub"]
            lines.append(f"  DEFENSE: send in {name} (+{gain:.1f} runs/162 at his position)")
    else:
        if rec.get("bunt"):
            lines.append("  BUNT: call for a sacrifice")
        if rec.get("pinch_hit"):
            name, gain = rec["pinch_hit"]
            lines.append(f"  PINCH HIT: send up {name} (+{gain:.3f} wOBA edge)")
        if not rec.get("bunt") and not rec.get("pinch_hit"):
            lines.append("  OFFENSE: no move recommended - let the current batter hit")
    return "\n".join(lines)


def run_live(session: Session) -> None:
    print("\nLive AutoManager session. Report each situation, get a recommendation, then tell it what actually")
    print("happened so the next call reflects reality. Type 'help' for commands, 'quit' to end.\n")
    help_text = (
        'Commands:\n'
        '  sit inning=7,outs=1,runners=13,lead=1,batter=4,field=1[,first=1]   -> get a recommendation\n'
        '      runners: digits 1/2/3 present = occupied (e.g. "13" = 1st and 3rd). field=1 if WE are fielding.\n'
        '  change <pitcher name>            -> record that this pitcher is now in\n'
        '  pinch <slot 1-9> <bench name>    -> record a pinch hitter\n'
        '  defsub <slot 1-9> <bench name>   -> record a defensive substitution\n'
        '  pc <pitches>                     -> update the current pitcher\'s pitch count\n'
        '  log                              -> show every situation asked about so far\n'
        '  quit                              -> end the session'
    )
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        if cmd == "quit":
            break
        if cmd == "help":
            print(help_text)
        elif cmd == "log":
            for rec in session.log:
                print(_fmt_rec(rec)); print()
        elif cmd == "sit":
            d = parse_situation(rest)
            mask_runners = d.get("runners", "0")
            try:
                rec = session.recommend(int(d.get("inning", 1)), int(d.get("outs", 0)), mask_runners,
                                        int(d.get("lead", 0)), int(d.get("batter", 1)),
                                        d.get("field", "0") == "1", d.get("first", "0") == "1")
            except SystemExit as e:
                print(e); continue
            print(_fmt_rec(rec))
        elif cmd == "change":
            print(session.change_pitcher(rest))
        elif cmd == "pinch":
            slot, _, name = rest.partition(" ")
            try:
                print(session.pinch_hit(int(slot), name))
            except (ValueError, SystemExit) as e:
                print(f"Usage: pinch <slot> <name> - {e}")
        elif cmd == "defsub":
            slot, _, name = rest.partition(" ")
            try:
                print(session.defensive_sub(int(slot), name))
            except (ValueError, SystemExit) as e:
                print(f"Usage: defsub <slot> <name> - {e}")
        elif cmd == "pc":
            try:
                session.report_pitch_count(float(rest))
                print(f"Pitch count updated: {rest}")
            except ValueError:
                print("Usage: pc <number>")
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for the list.")


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
    ap.add_argument("--park", help="Ballpark team code for park factors, e.g. COL (default: the home team's park)")
    ap.add_argument("--style", choices=["typical", "auto", "never"], default="typical",
                    help="Your tactics (bunts, IBBs, pinch hitters, defensive subs) in the comparison scenarios; "
                         "the AutoManager scenario always uses 'auto'")
    ap.add_argument("--situation", help='Ask what the AutoManager would do mid-game, e.g. '
                    '"inning=7,outs=1,runners=13,lead=1,pitches=88,batter=4,tto=2,field=1". '
                    'field=1 (default) means WE are fielding - asks about a pitching change. field=0 means '
                    'WE are batting - asks about a bunt/pinch-hit instead.')
    ap.add_argument("--live", action="store_true",
                    help="Start an interactive session that remembers pitching/bench usage as you report events")
    ap.add_argument("--venue", choices=["home", "away"], help="Override home/away (default: from the schedule, else home)")
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--sims", type=int, default=6000, help="Games to simulate per scenario")
    ap.add_argument("--no-sim", action="store_true", help="Lineup only: skip the opponent, bullpen and simulation")
    ap.add_argument("--show-data", action="store_true", help="Print the data loaded for every hitter on your roster")
    ap.add_argument("--demo", action="store_true", help="Use synthetic data (no network)")
    ap.add_argument("--no-weather", action="store_true", help="Skip today's weather adjustment to the park factor")
    ap.add_argument("--no-fatigue", action="store_true",
                    help="Skip the day-game-after-night-game fatigue check for both teams")
    ap.add_argument("--date", help="Game date for weather/fatigue checks, YYYY-MM-DD (default: today - set this "
                    "to a real date from the season if the season has ended, since 'today' won't have games)")
    args = ap.parse_args()

    game_date = None
    if args.date:
        try:
            game_date = date.fromisoformat(args.date)
        except ValueError:
            ap.error(f"--date '{args.date}' isn't a valid date; use YYYY-MM-DD, e.g. --date 2026-09-14")

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

    park_code = args.park or (args.team if we_home else args.opponent)
    weather_note = fatigue_note_us = fatigue_note_opp = None
    if park_code:
        weather_mult = 1.0
        if not args.demo and not args.no_weather:
            from data.weather import weather_factor
            weather_mult, weather_note = weather_factor(park_code, game_date)
        park_shift = park_woba_shift(park_factor(park_code) * weather_mult)
    else:
        park_shift = 0.0
    # Fatigue is a property of the BATTING team, not the park, so it's tracked separately per side and
    # combined with park_shift at each point below rather than folded into one shared "shift" - see the
    # make_staff() note a few lines down for why the two sides get swapped there.
    our_fatigue = opp_fatigue = 0.0
    if not args.demo and not args.no_fatigue and inp.team_id and inp.opp_id:
        from data.schedule import fatigue_shift
        our_fatigue, fatigue_note_us = fatigue_shift(inp.team_id, args.season, game_date)
        opp_fatigue, fatigue_note_opp = fatigue_shift(inp.opp_id, args.season, game_date)
    our_shift, opp_shift = park_shift + our_fatigue, park_shift + opp_fatigue

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

    our_bench_hitters = [h for h in ours if h.id not in {s.hitter.id for s in best.spots}]
    opp_bench_hitters = [h for h in inp.theirs if h.id not in {s.hitter.id for s in opp_lineup.spots}] if inp.theirs else []
    our_bench = [make_bench_player(h, inp.opp_starter, opp_def, our_shift) for h in our_bench_hitters]
    opp_bench = [make_bench_player(h, our_starter, best.def_rpg, opp_shift) for h in opp_bench_hitters]
    if args.show_data:
        print_bench("Our bench", our_bench)

    def staffs(lu):
        # NOTE the swap: make_staff()'s own "shift" is added to the wOBA of the opposing hitters it's
        # handed (that's who it's computing woba_vs for), so OUR staff takes opp_shift (it's being
        # evaluated against THEIR hitters) and THEIR staff takes our_shift (it's being evaluated
        # against OUR hitters). park_shift alone wouldn't need this since it's the same for both sides;
        # fatigue is team-specific, so getting this backwards would apply each team's tiredness to the
        # wrong lineup.
        ours_s = make_staff(inp.our_starter, inp.our_pen, [s.hitter for s in opp_lineup.spots], lu.def_rpg,
                            args.unavailable.split(","), opp_shift)
        theirs_s = make_staff(inp.opp_starter, inp.opp_pen, [s.hitter for s in lu.spots], opp_lineup.def_rpg,
                              args.opp_unavailable.split(","), our_shift)
        return ours_s, theirs_s

    our_staff, _ = staffs(best)
    print_staff(f"Our pitching staff vs the opposing lineup (starter: {our_starter.name})", our_staff)
    if args.situation or args.live:
        from sim.setup import make_batter
        our_batters_live = [make_batter(s.hitter, inp.opp_starter, opp_lineup.def_rpg if opp_lineup else 0.0, our_shift,
                                        s.field_pos, s.def_rpg * 162) for s in best.spots]
        if args.situation:
            advise(args.situation, our_staff, our_batters_live, list(our_bench), we_home)
            return
        session = Session(our_batters_live, list(our_bench), list(our_staff))
        run_live(session)
        return

    baseline = build_lineup(ours, inp.opp_starter, defense_weight=0.0, forced=forced)   # offense-first, ignores gloves
    print(f"\nSimulating {args.sims:,} games per scenario ({'home' if we_home else 'away'}) ...")
    scenarios = [("Lineup + AutoManager (bullpen+tactics)", best, AutoPolicy(), "auto"),
                 ("Lineup + typical bullpen/tactics", best, TypicalPolicy(), args.style),
                 ("Offense-first lineup + typical", baseline, TypicalPolicy(), args.style)]
    rows = []
    for label, lu, policy, style in scenarios:
        ours_s, theirs_s = staffs(lu)
        us = make_side("Us", lu.spots, inp.opp_starter, opp_lineup.def_rpg, staff=ours_s, policy=policy,
                      park_shift=our_shift, bench=our_bench, style=style)
        them = make_side("Them", opp_lineup.spots, our_starter, lu.def_rpg, staff=theirs_s, policy=TypicalPolicy(),
                         park_shift=opp_shift, bench=opp_bench, style="typical")
        rows.append((label, simulate(us, them, we_home, args.sims, seed=42)))
    print(f"\n{'':<38}{'Runs/G':>8}{'Opp R/G':>9}{'Win %':>8}{'SP IP':>7}{'Relievers':>10}{'Extras':>8}")
    for label, r in rows:
        print(f"{label:<38}{r.runs_for:>8.2f}{r.runs_against:>9.2f}{r.win_pct * 100:>7.1f}%"
              f"{r.starter_outs / 3:>7.1f}{r.relievers_used:>10.1f}{r.extras_pct * 100:>7.1f}%")
    print_usage(rows[0][1], our_staff)
    if park_code:
        line = f"\nBallpark: {park_code} (static park factor {park_factor(park_code):.2f}; 1.00 = neutral)"
        if weather_note:
            line += f", today's conditions: {weather_note}"
        print(line + ".")
    if fatigue_note_us and our_fatigue:
        print(f"Fatigue: {args.team} - {fatigue_note_us}.")
    if fatigue_note_opp and opp_fatigue:
        print(f"Fatigue: {args.opponent} - {fatigue_note_opp}.")
    print("\nSame random seed in every scenario. The opponent always fields with typical bullpen/tactics management."
          " The simulation includes home-field advantage, steals, errors, RISP, extra innings, times-through-the-order,"
          " fatigue (both pitcher workload and day-after-night travel), ballpark factors plus today's weather,"
          " GIDP-aware baserunning, pitch-mix matchups, intentional walks, sacrifice bunts, pinch hitters and"
          " defensive substitutions. Win% differences under ~1 point are within simulation noise.")


if __name__ == "__main__":
    main()