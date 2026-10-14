"""Day-game-after-night-game fatigue, from the MLB Stats API schedule.

A team that played a night game yesterday and is turning around for a day game today is running on
less rest than usual - the classic "getaway day"/businessman's-special fatigue effect broadcasters
mention constantly. This is a narrower, simpler thing than full travel-distance/time-zone modeling
(which would need the previous city's and today's city's locations, time-zone deltas, etc.) - just the
single, well-known "night into day" turnaround, applied as a small flat wOBA shift to that team's
hitters for this game, the same way park_woba_shift() applies the park factor.
"""
import sys
from datetime import date, timedelta

from .cache import cached

DAY_AFTER_NIGHT_WOBA_PENALTY = 0.006   # modest - similar order of magnitude to the reliever workload
                                       # penalties in models/pitching.py, not a dramatic swing


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _is_day_game(game: dict) -> bool | None:
    dn = (game.get("day_night") or "").lower()
    if dn in ("day", "night"):
        return dn == "day"
    return None


def fatigue_shift(team_id: int, season: int, when: date | None = None) -> tuple[float, str]:
    """(wOBA shift to apply to this team's hitters, human-readable note), for the game date `when`
    (default: today). 0.0 / "rested" if the team didn't play the day before `when`, played a day game
    that day, isn't playing a day game on `when`, or the schedule couldn't be loaded - best-effort,
    like every other live-data fetch in this project. Pass `when` explicitly (main.py's --date) to
    check a date from a season that's already over, since "today" won't have any real games then."""
    try:
        import statsapi
    except Exception as e:  # noqa: BLE001
        _warn(f"statsapi not available for schedule/fatigue check ({e})")
        return 0.0, "unavailable"
    today = when or date.today()
    yesterday = today - timedelta(days=1)

    def _fetch():
        games = statsapi.schedule(team=team_id, start_date=yesterday.strftime("%m/%d/%Y"),
                                  end_date=today.strftime("%m/%d/%Y"))
        return [g for g in games if g.get("status") not in ("Cancelled", "Postponed")]

    try:
        games = cached(f"schedule_{team_id}_{yesterday.isoformat()}", _fetch, ttl_hours=6)
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load schedule for fatigue check ({type(e).__name__}: {e}); assuming normal rest")
        return 0.0, "unavailable"
    by_date = {g.get("game_date"): g for g in games}
    today_g, yday_g = by_date.get(today.isoformat()), by_date.get(yesterday.isoformat())
    if not today_g or not yday_g:
        return 0.0, "no game yesterday"
    today_day, yday_day = _is_day_game(today_g), _is_day_game(yday_g)
    if today_day is True and yday_day is False:
        return -DAY_AFTER_NIGHT_WOBA_PENALTY, "day game after a night game - short turnaround"
    return 0.0, "normal rest"