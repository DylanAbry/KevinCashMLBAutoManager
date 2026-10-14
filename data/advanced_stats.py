"""Advanced stats from Baseball Savant / FanGraphs via pybaseball (imported lazily).

- xwOBA / wOBA      : Savant expected-stats leaderboard (ids are MLBAM ids, same as the MLB Stats API)
- OAA               : Savant Outs Above Average by position (converted to runs)
- DRS               : FanGraphs, matched by player name. FanGraphs sometimes blocks scraping,
                      so DRS is best-effort; if it fails we fall back to OAA only and say so.
- catcher framing   : Savant's catcher-framing leaderboard (strike-zone runs added/lost), stored as
                      def_runs["C"] - a catcher's defensive value is almost entirely framing/blocking,
                      not range, so this is kept separate from the OAA/DRS fielding-runs path above
                      (which never covers catchers).
- pitch-mix matchup : Savant's pitch-arsenal leaderboards, one for how each hitter performs against
                      each pitch TYPE league-wide, one for how often each pitcher throws each type.
                      Combined in models/matchup.pitch_mix_delta_woba() - see that function's docstring
                      for why this is the more reliable stand-in for "this hitter sees that pitcher well"
                      than a literal (and far smaller-sample) batter-vs-this-exact-pitcher history.
Anything that fails prints a [warn] line and the program carries on with what it has (league-average
defense, no pitch-mix adjustment, etc.) - this file's data sources are the least stable part of the
project (a scraped leaderboard, or a pybaseball function that may not exist in every installed version),
so none of them are allowed to take the rest of the run down with them.
"""
import json
import re
import sys
import unicodedata

from .cache import cached
from .schema import Hitter, Pitcher

POS_CODES = {"1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}
OAA_TO_RUNS = 0.9      # roughly 0.9 runs per out above average
MIN_ATTEMPTS = 5       # ignore emergency cameos at a position
MIN_FRAMING_INNINGS = 150     # ignore cameo catching appearances
MIN_PITCH_COUNT = 20           # ignore a pitch type thrown/faced too rarely to mean anything

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


def fetch_catcher_framing(season: int) -> dict[int, float]:
    """player_id -> framing runs above average, from Savant's catcher-framing leaderboard. The exact
    pybaseball helper and column names for this leaderboard have moved around between pybaseball
    versions, so this tries the current name and a couple of plausible fallbacks and gives up quietly
    (defense just treated as league-average for every catcher) if none of them work - framing is a
    nice-to-have refinement, not something worth crashing the whole run over."""
    try:
        import pybaseball
    except Exception as e:  # noqa: BLE001
        _warn(f"pybaseball not available for catcher framing ({e})")
        return {}
    fetch = getattr(pybaseball, "statcast_catcher_framing", None) or getattr(pybaseball, "statcast_framing", None)
    if fetch is None:
        _warn("this version of pybaseball has no catcher-framing leaderboard helper; catchers will be "
              "treated as league-average on defense")
        return {}
    try:
        rows = cached(f"savant_framing_{season}", lambda: _records(fetch(season)))
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load catcher framing ({type(e).__name__}: {e}); catchers will be league-average on defense")
        return {}
    out: dict[int, float] = {}
    for r in rows:
        pid = r.get("player_id") or r.get("playerid") or r.get("mlbam_id")
        innings = r.get("n_called_pitches") or r.get("innings") or r.get("inning") or MIN_FRAMING_INNINGS
        runs = r.get("runs_extra_strikes") or r.get("framing_runs") or r.get("strike_rate_runs") or r.get("runs")
        if pid is None or runs is None:
            continue
        try:
            if float(innings) < MIN_FRAMING_INNINGS:
                continue
        except (TypeError, ValueError):
            pass
        out[int(pid)] = float(runs)
    if not out:
        _warn("catcher-framing leaderboard returned no usable rows; catchers will be league-average on defense")
    return out


def fetch_batter_pitch_woba(season: int) -> dict[int, dict[str, tuple[float, int]]]:
    """player_id -> {pitch_type_code: (wOBA, pitches_seen)}, from Savant's pitch-arsenal-stats leaderboard
    (batter side). Same best-effort/give-up-quietly approach as fetch_catcher_framing, for the same
    reason: this leaderboard's exact pybaseball name and column layout isn't as stable as the core
    xwOBA/OAA ones this file has used for a while."""
    try:
        import pybaseball
    except Exception as e:  # noqa: BLE001
        _warn(f"pybaseball not available for pitch-type matchup data ({e})")
        return {}
    fetch = getattr(pybaseball, "statcast_batter_pitch_arsenal", None) or getattr(pybaseball, "statcast_batter_arsenal_stats", None)
    if fetch is None:
        _warn("this version of pybaseball has no batter pitch-arsenal helper; pitch-mix matchups will be skipped")
        return {}
    try:
        rows = cached(f"savant_batter_arsenal_{season}", lambda: _records(fetch(season, minPA=1)))
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load batter pitch-type data ({type(e).__name__}: {e}); pitch-mix matchups will be skipped")
        return {}
    out: dict[int, dict[str, tuple[float, int]]] = {}
    for r in rows:
        pid = r.get("player_id") or r.get("batter") or r.get("mlbam_id")
        pt = r.get("pitch_type") or r.get("pitch_name")
        n = r.get("pitches") or r.get("n") or r.get("pitch_count")
        woba = r.get("woba") or r.get("est_woba") or r.get("whiff_woba")
        if pid is None or not pt or woba is None or not n:
            continue
        if float(n) < MIN_PITCH_COUNT:
            continue
        out.setdefault(int(pid), {})[str(pt).upper()] = (float(woba), int(n))
    if not out:
        _warn("batter pitch-arsenal leaderboard returned no usable rows; pitch-mix matchups will be skipped")
    return out


def fetch_pitcher_pitch_mix(season: int) -> dict[int, dict[str, float]]:
    """player_id -> {pitch_type_code: usage share summing to ~1.0}, from Savant's pitch-arsenal-stats
    leaderboard (pitcher side)."""
    try:
        import pybaseball
    except Exception as e:  # noqa: BLE001
        _warn(f"pybaseball not available for pitch-mix data ({e})")
        return {}
    fetch = getattr(pybaseball, "statcast_pitcher_pitch_arsenal", None) or getattr(pybaseball, "statcast_pitcher_arsenal_stats", None)
    if fetch is None:
        _warn("this version of pybaseball has no pitcher pitch-arsenal helper; pitch-mix matchups will be skipped")
        return {}
    try:
        rows = cached(f"savant_pitcher_arsenal_{season}", lambda: _records(fetch(season, minP=1)))
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load pitcher pitch-mix data ({type(e).__name__}: {e}); pitch-mix matchups will be skipped")
        return {}
    out: dict[int, dict[str, float]] = {}
    for r in rows:
        pid = r.get("player_id") or r.get("pitcher") or r.get("mlbam_id")
        pt = r.get("pitch_type") or r.get("pitch_name")
        usage = r.get("pitch_usage") or r.get("usage") or r.get("n_pitches_formatted")
        if pid is None or not pt or usage is None:
            continue
        usage = float(usage)
        if usage > 1.5:          # some leaderboards report usage as a 0-100 percentage, not a 0-1 share
            usage /= 100.0
        out.setdefault(int(pid), {})[str(pt).upper()] = usage
    if not out:
        _warn("pitcher pitch-arsenal leaderboard returned no usable rows; pitch-mix matchups will be skipped")
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
        _MEMO[season] = {
            "x": fetch_xstats(season), "oaa": fetch_oaa(season), "drs": fetch_drs(season),
            "framing": fetch_catcher_framing(season), "batter_pitch": fetch_batter_pitch_woba(season),
            "pitcher_mix": fetch_pitcher_pitch_mix(season),
        }
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
        framing = adv["framing"].get(h.id)
        if framing is not None and h.pos == "C":
            h.def_runs["C"] = framing
        h.pitch_woba = dict(adv["batter_pitch"].get(h.id, {}))


def enrich_pitchers(pitchers: list[Pitcher], season: int) -> None:
    adv = load_advanced(season)
    for p in pitchers:
        p.pitch_mix = dict(adv["pitcher_mix"].get(p.id, {}))