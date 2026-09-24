"""Matchup model v3: expected wOBA for a hitter against a specific pitcher, plus RISP and running skills.

  expected_wOBA = talent  +  platoon adjustment  +  pitcher adjustment

talent    : blend of season wOBA (40%) and Statcast xwOBA (60%), shrunk toward the league
            average by plate appearances, so small samples get pulled back (= "reliability").
platoon   : how much better/worse the hitter is vs this pitcher's hand than his own overall
            OPS, heavily shrunk because platoon splits are noisy.
pitcher   : how much OPS the pitcher allows to this batter's side vs the league average, shrunk.
RISP      : how much better/worse he hits with runners in scoring position than overall, shrunk
            VERY hard toward zero (K_RISP), because RISP performance carries over little year to year.
"""
from data.schema import Hitter, Pitcher

LEAGUE_WOBA = 0.315
LEAGUE_OPS = 0.720
WOBA_SCALE = 1.20            # wOBA points -> runs per PA
OPS_TO_WOBA = 0.43
PA_PER_BATTER_GAME = 4.3
TEAM_PA_PER_GAME = 38.5

K_TALENT = 200               # PA of league-average "prior" added to a hitter's season line
K_PLATOON = 400
K_PITCHER = 300
K_RISP = 800                 # raise this to trust RISP splits more, lower it to trust them less
BULLPEN_RIGHTY_SHARE = 0.70

LEAGUE_STEAL_ATTEMPT = 0.09  # per plate appearance with a runner on 1st and 2nd base open
LEAGUE_STEAL_SUCCESS = 0.78
K_ATTEMPT = 60               # times-on-first of prior
K_SUCCESS = 20               # attempts of prior
PA_WITH_RUNNER_PER_TIME_ON_FIRST = 1.4


def _shrink(value: float, n: float, prior: float, k: float) -> float:
    return (value * n + prior * k) / (n + k)


def talent_woba(h: Hitter) -> float:
    parts = [(w, x) for w, x in ((0.4, h.woba), (0.6, h.xwoba)) if x is not None]
    if not parts:
        return LEAGUE_WOBA
    raw = sum(w * x for w, x in parts) / sum(w for w, _ in parts)
    return _shrink(raw, h.pa, LEAGUE_WOBA, K_TALENT)


def overall_ops(h: Hitter) -> float:
    if h.season.get("ops"):
        return h.season["ops"]
    n = sum(v[1] for v in h.vs.values())
    return sum(v[0] * v[1] for v in h.vs.values()) / n if n else LEAGUE_OPS


def platoon_delta_ops(h: Hitter, pitcher_hand: str) -> float:
    other = "L" if pitcher_hand == "R" else "R"
    o_h, n_h = h.vs.get(pitcher_hand, (LEAGUE_OPS, 0))
    o_o, n_o = h.vs.get(other, (LEAGUE_OPS, 0))
    if n_h + n_o == 0 or n_h == 0:
        return 0.0
    overall = (o_h * n_h + o_o * n_o) / (n_h + n_o)
    return _shrink(o_h, n_h, overall, K_PLATOON) - overall


def expected_woba(h: Hitter, p: Pitcher) -> float:
    side = h.bat_side if h.bat_side != "S" else ("L" if p.throws == "R" else "R")
    p_ops, p_n = p.vs.get(side, (LEAGUE_OPS, 0))
    pitcher_delta = _shrink(p_ops, p_n, LEAGUE_OPS, K_PITCHER) - LEAGUE_OPS
    return talent_woba(h) + OPS_TO_WOBA * (platoon_delta_ops(h, p.throws) + pitcher_delta)


def risp_delta_woba(h: Hitter) -> float:
    """wOBA change when runners are in scoring position (heavily regressed)."""
    if not h.risp or h.risp[1] <= 0:
        return 0.0
    ops, n = h.risp
    base = overall_ops(h)
    return OPS_TO_WOBA * (_shrink(ops, n, base, K_RISP) - base)


def steal_rates(h: Hitter) -> tuple[float, float]:
    """(attempt probability per PA with a runner on 1st and 2nd open, success probability)."""
    s = h.season
    sb, cs = s.get("sb", 0.0), s.get("cs", 0.0)
    times_on_first = s.get("s1", 0.0) + s.get("bb", 0.0)
    raw = (sb + cs) / times_on_first / PA_WITH_RUNNER_PER_TIME_ON_FIRST if times_on_first else LEAGUE_STEAL_ATTEMPT
    attempt = _shrink(raw, times_on_first, LEAGUE_STEAL_ATTEMPT, K_ATTEMPT)
    raw_succ = sb / (sb + cs) if sb + cs else LEAGUE_STEAL_SUCCESS
    return attempt, _shrink(raw_succ, sb + cs, LEAGUE_STEAL_SUCCESS, K_SUCCESS)


def neutral_pitcher(throws: str = "R") -> Pitcher:
    return Pitcher(id=0, name="League-average pitcher", throws=throws, vs={})


def bullpen_woba(h: Hitter) -> float:
    """Expected wOBA vs a league-average reliever (mix of righties and lefties)."""
    r = expected_woba(h, neutral_pitcher("R"))
    l = expected_woba(h, neutral_pitcher("L"))
    return BULLPEN_RIGHTY_SHARE * r + (1 - BULLPEN_RIGHTY_SHARE) * l
