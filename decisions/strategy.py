"""In-game tactical decisions that aren't pitching changes: intentional walks, sacrifice bunts, pinch
hitters, and defensive replacements. "typical" mimics a conventional manager (bunts more, IBBs the classic
"open base, first to third" spot); "auto" follows modern run-expectancy analysis (bunts and IBBs rarely,
since both usually cost expected runs); "never" turns a tactic off entirely.

These are plain functions, not classes, so the CLI advisor and the simulator call the exact same logic.
"""
from models.run_expectancy import RE24

# Sacrifice bunts lose expected runs in almost all textbook spots; only use them at the extremes.
BUNT_WOBA_CEILING_AUTO = 0.230       # "auto" only bunts a genuinely weak hitter (e.g. an NL pitcher's spot)
BUNT_WOBA_CEILING_TYPICAL = 0.300    # "typical" bunts a lot more often

IBB_WOBA_GAP_AUTO = 0.140            # "auto" needs a huge gap between this hitter and the next to consider it
IBB_WOBA_GAP_TYPICAL = 0.075

PINCH_HIT_MIN_GAIN = 0.030           # wOBA gain needed to burn a bench bat (auto)
PINCH_HIT_LEVERAGE_MIN = 0.8         # don't bother pinch hitting in a low-leverage spot
DEF_SUB_MIN_RUNS162 = 4.0            # runs/162 the substitute must add on defense (auto)
DEF_SUB_LEAD_MAX = 4                 # only protect leads of about this size or smaller


def should_bunt(style: str, inning: int, outs: int, mask: int, lead: int, batter_woba: float) -> bool:
    if style == "never" or outs >= 2 or mask not in (1, 3):        # only makes sense: runner on 1st, maybe 2nd too
        return False
    if abs(lead) >= 4 or inning >= 9:
        return False
    ceiling = BUNT_WOBA_CEILING_AUTO if style == "auto" else BUNT_WOBA_CEILING_TYPICAL
    return batter_woba <= ceiling


def should_ibb(style: str, inning: int, outs: int, mask: int, lead: int,
              batter_woba: float, on_deck_woba: float) -> bool:
    if style == "never" or mask & 1 or outs >= 2:                  # first base must be open
        return False
    if inning < 7 or lead < -1 or lead > 2:
        return False
    gap = IBB_WOBA_GAP_AUTO if style == "auto" else IBB_WOBA_GAP_TYPICAL
    return batter_woba - on_deck_woba >= gap


def choose_pinch_hitter(style: str, bench: list, used: set, idx: int, batters: list,
                        pitcher_throws: str, inning: int, leverage: float, lead: int):
    """Returns (bench_index, gain) or None. Swaps for a same-side or better platoon bat with a real edge."""
    if style == "never" or style == "typical" or not bench or inning < 6 or leverage < PINCH_HIT_LEVERAGE_MIN:
        return None
    if abs(lead) > 5:
        return None
    cur = batters[idx].woba_starter
    best = None
    for i, b in enumerate(bench):
        if i in used or (pitcher_throws in ("L",) and b.bat_side == "L"):
            continue
        gain = b.woba_starter - cur
        if gain >= PINCH_HIT_MIN_GAIN and (best is None or gain > best[1]):
            best = (i, gain)
    return best


def choose_defensive_sub(style: str, bench: list, used: set, idx: int, inning: int, lead: int):
    """Returns (bench_index, def_runs162) or None. Protects a lead late by upgrading the glove at some cost to offense."""
    if style != "auto" or not bench or inning < 7 or not (0 < lead <= DEF_SUB_LEAD_MAX):
        return None
    best = None
    for i, b in enumerate(bench):
        if i in used or b.def_runs < DEF_SUB_MIN_RUNS162:
            continue
        if best is None or b.def_runs > best[1]:
            best = (i, b.def_runs)
    return best