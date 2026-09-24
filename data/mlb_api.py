"""Rosters, platoon splits, season lines, RISP splits and probable starters from the free MLB Stats API
(cached to data/cache). `statsapi` is imported lazily so --demo works without network or the package."""
from datetime import date, timedelta

from .advanced_stats import _warn as warn
from .advanced_stats import enrich_hitters
from .cache import cached
from .schema import Hitter, Pitcher


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def get_team_id(term: str, season: int) -> int:
    import statsapi
    matches = statsapi.lookup_team(term, season=season, sportIds=1)
    if not matches:
        raise ValueError(f"No MLB team found for '{term}'")
    t = term.lower()
    for m in matches:      # prefer an exact abbreviation/name hit over a substring hit
        if t in (str(m.get(k, "")).lower() for k in ("fileCode", "teamCode", "abbreviation", "teamName", "name")):
            return m["id"]
    return matches[0]["id"]


def _people(ids: list[int], hydrate: str, label: str, season: int) -> list[dict]:
    import statsapi
    key = f"people_{label}_{season}_" + "-".join(map(str, sorted(ids)))[:80]
    return cached(key, lambda: statsapi.get(
        "people", {"personIds": ",".join(map(str, ids)), "hydrate": hydrate})["people"])


def _people_with_splits(ids: list[int], group: str, season: int) -> list[dict]:
    hydrate = f"stats(group=[{group}],type=[statSplits],sitCodes=[vl,vr],season={season})"
    return _people(ids, hydrate, group, season)


def _parse_splits(person: dict, pa_key: str, wanted=("vl", "vr")) -> dict:
    out = {}
    for block in person.get("stats", []):
        for s in block.get("splits", []):
            code = s.get("split", {}).get("code")
            if code in wanted:
                st = s["stat"]
                out[code.upper() if code == "risp" else code[-1].upper()] = (_num(st.get("ops")), int(_num(st.get(pa_key))))
    return out


def _parse_season(person: dict) -> dict:
    for block in person.get("stats", []):
        for s in block.get("splits", []):
            st = s.get("stat", {})
            pa = int(_num(st.get("plateAppearances")))
            if not pa:
                continue
            hits, d, t, hr = (_num(st.get(k)) for k in ("hits", "doubles", "triples", "homeRuns"))
            return {"pa": pa, "k": _num(st.get("strikeOuts")), "bb": _num(st.get("baseOnBalls")) + _num(st.get("hitByPitch")),
                    "s1": hits - d - t - hr, "s2": d, "s3": t, "hr": hr,
                    "sb": _num(st.get("stolenBases")), "cs": _num(st.get("caughtStealing")), "ops": _num(st.get("ops"))}
    return {}


def _attach_extras(hitters: list[Hitter], season: int) -> None:
    """Season counting stats (for outcome profiles and steals) and RISP splits. Each is best-effort."""
    ids = [h.id for h in hitters]
    by_id = {h.id: h for h in hitters}
    try:
        for p in _people(ids, f"stats(group=[hitting],type=[season],season={season})", "season", season):
            if p["id"] in by_id:
                by_id[p["id"]].season = _parse_season(p)
    except Exception as e:  # noqa: BLE001
        warn(f"couldn't load season stat lines ({type(e).__name__}: {e}); using league-average outcome profiles")
    try:
        hyd = f"stats(group=[hitting],type=[statSplits],sitCodes=[risp],season={season})"
        for p in _people(ids, hyd, "risp", season):
            r = _parse_splits(p, "plateAppearances", wanted=("risp",)).get("RISP")
            if p["id"] in by_id and r and r[1] > 0:
                by_id[p["id"]].risp = r
    except Exception as e:  # noqa: BLE001
        warn(f"couldn't load RISP splits ({type(e).__name__}: {e}); RISP will be ignored")


def get_hitters(team_id: int, season: int) -> list[Hitter]:
    import statsapi
    roster = cached(f"roster_{team_id}", lambda: statsapi.get(
        "team_roster", {"teamId": team_id, "rosterType": "active"})["roster"])
    bats = [r for r in roster if r["position"]["type"] != "Pitcher"]
    people = _people_with_splits([r["person"]["id"] for r in bats], "hitting", season)
    pos_by_id = {r["person"]["id"]: r["position"]["abbreviation"] for r in bats}
    hitters = [
        Hitter(id=p["id"], name=p["fullName"], pos=pos_by_id.get(p["id"], "DH"),
               bat_side=p.get("batSide", {}).get("code", "R"), vs=_parse_splits(p, "plateAppearances"))
        for p in people
    ]
    enrich_hitters(hitters, season)
    _attach_extras(hitters, season)
    return hitters


def get_pitcher(pitcher_id: int, season: int) -> Pitcher:
    p = _people_with_splits([pitcher_id], "pitching", season)[0]
    return Pitcher(id=p["id"], name=p["fullName"], throws=p.get("pitchHand", {}).get("code", "R"),
                   vs=_parse_splits(p, "battersFaced"))


def find_player_id(name: str) -> int:
    import statsapi
    hits = statsapi.lookup_player(name)
    if not hits:
        raise ValueError(f"No player found for '{name}'")
    return hits[0]["id"]


def find_matchup_starters(team_id: int, opp_id: int) -> dict:
    """Next scheduled game between the two teams: {'home': bool, 'our_pitcher': str|None, 'opp_pitcher': str|None}."""
    import statsapi
    try:
        games = statsapi.schedule(
            team=team_id, opponent=opp_id,
            start_date=date.today().strftime("%m/%d/%Y"),
            end_date=(date.today() + timedelta(days=30)).strftime("%m/%d/%Y"))
    except Exception:  # noqa: BLE001
        games = []
    for g in games:
        we_home = g.get("home_id") == team_id
        ours = g.get("home_probable_pitcher") if we_home else g.get("away_probable_pitcher")
        theirs = g.get("away_probable_pitcher") if we_home else g.get("home_probable_pitcher")
        if ours or theirs:
            return {"home": we_home, "our_pitcher": ours or None, "opp_pitcher": theirs or None}
    return {"home": True, "our_pitcher": None, "opp_pitcher": None}
