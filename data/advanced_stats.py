"""Advanced stats from Baseball Savant / FanGraphs via pybaseball (imported lazily).

- xwOBA / wOBA : Savant expected-stats leaderboard (ids are MLBAM ids, same as the MLB Stats API)
- OAA          : Savant Outs Above Average by position (converted to runs)
- DRS          : FanGraphs, matched by player name. FanGraphs sometimes blocks scraping,
                 so DRS is best-effort; if it fails we fall back to OAA only and say so.
Anything that fails prints a [warn] line and the program carries on with what it has.
"""
import json
import re
import sys
import unicodedata

from .cache import cached
from .schema import Hitter

POS_CODES = {"1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}
OAA_TO_RUNS = 0.9      # roughly 0.9 runs per out above average
MIN_ATTEMPTS = 5       # ignore emergency cameos at a position

_MEMO: dict[int, dict] = {}


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def fetch_xstats(season: int) -> dict[int, dict]:
    try:
        from pybaseball import statcast_batter_expected_stats
        rows = cached(f"savant_xstats_{season}",
                      lambda: _records(statcast_batter_expected_stats(season, minPA=1)))
        return {int(r["player_id"]): {"pa": int(r.get("pa") or 0), "woba": r.get("woba"), "xwoba": r.get("est_woba")}
                for r in rows if r.get("player_id") is not None}
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load Statcast wOBA/xwOBA ({type(e).__name__}: {e}); using platoon OPS only")
        return {}


def fetch_oaa(season: int) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    try:
        from pybaseball import statcast_outs_above_average
    except Exception as e:  # noqa: BLE001
        _warn(f"pybaseball not available for OAA ({e})")
        return out
    for pos, code in POS_CODES.items():
        try:
            rows = cached(f"savant_oaa_{season}_{pos}",
                          lambda: _records(statcast_outs_above_average(season, code, min_att=1)))
        except Exception as e:  # noqa: BLE001
            _warn(f"couldn't load OAA for {pos} ({type(e).__name__}: {e})")
            continue
        for r in rows:
            pid, att = r.get("player_id"), r.get("attempts")
            if pid is None or (att is not None and att < MIN_ATTEMPTS):
                continue
            runs = r.get("fielding_runs_prevented")
            if runs is None and r.get("outs_above_average") is not None:
                runs = r["outs_above_average"] * OAA_TO_RUNS
            if runs is not None:
                out.setdefault(int(pid), {})[pos] = float(runs)
    if not out:
        _warn("no OAA data loaded - defense will be treated as league average")
    return out


def fetch_drs(season: int) -> dict[str, float]:
    """name -> total DRS (best effort; FanGraphs may block automated requests)."""
    try:
        from pybaseball import fielding_stats
        rows = cached(f"fg_drs_{season}", lambda: _records(fielding_stats(season, qual=1)[["Name", "DRS"]]))
        out: dict[str, float] = {}
        for r in rows:
            if r.get("DRS") is not None:
                out[_norm(r["Name"])] = out.get(_norm(r["Name"]), 0.0) + float(r["DRS"])
        return out
    except Exception as e:  # noqa: BLE001
        _warn(f"DRS unavailable ({type(e).__name__}: {e}); using OAA only for defense")
        return {}


def load_advanced(season: int) -> dict:
    if season not in _MEMO:
        _MEMO[season] = {"x": fetch_xstats(season), "oaa": fetch_oaa(season), "drs": fetch_drs(season)}
    return _MEMO[season]


def enrich_hitters(hitters: list[Hitter], season: int) -> None:
    adv = load_advanced(season)
    for h in hitters:
        x = adv["x"].get(h.id)
        if x:
            h.pa, h.woba, h.xwoba = x["pa"], x["woba"], x["xwoba"]
        h.def_runs = dict(adv["oaa"].get(h.id, {}))
        drs = adv["drs"].get(_norm(h.name))
        if drs is not None and h.pos in POS_CODES:      # DRS is a season total; apply at primary position
            oaa = h.def_runs.get(h.pos)
            h.def_runs[h.pos] = drs if oaa is None else (oaa + drs) / 2