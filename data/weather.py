"""Game-day weather, layered on top of the static park factor (data/ballparks.py).

A park's SEASON-long park factor already bakes in its usual weather pattern (Coors is thin-air hitter
friendly on average; Oracle Park is cool and offense-suppressing on average). What it can't capture is
TODAY's specific conditions - a hot, calm afternoon plays hitter-friendlier than that park's average
day, a cold one plays pitcher-friendlier. That's what this file adds: a same-day temperature read from
a free weather API, turned into a small multiplicative nudge on top of the static factor.

Deliberately NOT modeled: wind direction. The classic "wind blowing out at Wrigley" effect is real and
large, but modeling it correctly needs each park's home-plate-to-center-field compass orientation, and
there's no single reliable, verified source for all 30 (and the retractable-roof parks would need to
know whether the roof is even open that day, which isn't something a forecast API can tell you). Rather
than fake an orientation table and get the direction of the effect wrong half the time, wind speed is
reported for context in the printed ballpark line but doesn't change the number. If you want to add
this properly later, the model.weather_factor() function below is the one place it would plug in.

Indoor parks (true fixed domes, and every retractable-roof park - since there's no reliable way to know
from here whether the roof happens to be open today) are excluded entirely: outdoor weather just
doesn't apply to them, and guessing wrong would be a worse error than skipping it.
"""
import sys
from datetime import date

from .cache import cached

# Approximate stadium coordinates (city-block precision - plenty for a weather lookup).
BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "COL": (39.7559, -104.9942), "BOS": (42.3467, -71.0972), "MIN": (44.9817, -93.2776),
    "ARI": (33.4455, -112.0667), "PHI": (39.9061, -75.1665), "WSH": (38.8730, -77.0074),
    "BAL": (39.2838, -76.6217), "KC": (39.0517, -94.4803), "PIT": (40.4469, -80.0057),
    "CIN": (39.0974, -84.5068), "TOR": (43.6414, -79.3894), "LAD": (34.0739, -118.2400),
    "NYY": (40.8296, -73.9262), "MIA": (25.7781, -80.2197), "HOU": (29.7572, -95.3552),
    "DET": (42.3390, -83.0485), "ATL": (33.8908, -84.4678), "CWS": (41.8299, -87.6338),
    "LAA": (33.8003, -117.8827), "CLE": (41.4962, -81.6852), "MIL": (43.0280, -87.9712),
    "CHC": (41.9484, -87.6553), "NYM": (40.7571, -73.8458), "SD": (32.7073, -117.1566),
    "SF": (37.7786, -122.3893), "STL": (38.6226, -90.1928), "TB": (27.7683, -82.6534),
    "TEX": (32.7473, -97.0945), "SEA": (47.5914, -122.3325), "ATH": (38.5805, -121.5195),
}

# Excluded from weather adjustment entirely - see module docstring. TB is a true fixed dome; the rest
# have a retractable roof and there's no forecast-API field that says whether it's open today.
DOME_OR_UNKNOWN_ROOF = {"TB", "ARI", "HOU", "MIA", "MIL", "SEA", "TOR", "TEX"}

LEAGUE_AVG_TEMP_F = 70.0
# Rough, deliberately gentle rule of thumb: Alan Nathan's physics work on batted-ball carry puts the
# effect at roughly an extra foot of fly-ball distance per ~2.5 degF of warmer air. That's only enough
# to flip a handful of balls a year from outs into homers, so this translates to a SMALL run-scoring
# nudge, not a big one - +/-0.04 at the extremes (40 degF below/above league-average), clamped so a
# freak heat wave or cold snap can't swing the park factor wildly.
TEMP_WOBA_PER_DEGREE = 0.0010
MAX_TEMP_SHIFT = 0.04


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def fetch_forecast(team_code: str, when: date | None = None) -> dict | None:
    """{'temp_f': float, 'wind_mph': float} for the afternoon/evening of `when` (default: today) at this
    park, via Open-Meteo (free, no API key). Best-effort: returns None on any failure so callers just
    skip the adjustment, same philosophy as the rest of data/advanced_stats.py."""
    coords = BALLPARK_COORDS.get(team_code.upper())
    if coords is None:
        return None
    when = when or date.today()
    lat, lon = coords

    def _fetch():
        import urllib.request
        import json as _json
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph"
               f"&start_date={when.isoformat()}&end_date={when.isoformat()}&timezone=auto")
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        hourly = data.get("hourly", {})
        temps, winds = hourly.get("temperature_2m", []), hourly.get("wind_speed_10m", [])
        if not temps:
            return None
        # most MLB games are played in the evening - use the hour closest to 7pm local as a rough stand-in
        # for first pitch (good enough for a small nudge; pass a real first-pitch hour in if you have one)
        i = min(19, len(temps) - 1)
        return {"temp_f": temps[i], "wind_mph": winds[i] if i < len(winds) else None}

    try:
        return cached(f"weather_{team_code.upper()}_{when.isoformat()}", _fetch, ttl_hours=3)
    except Exception as e:  # noqa: BLE001
        _warn(f"couldn't load weather for {team_code} ({type(e).__name__}: {e}); ignoring today's conditions")
        return None


def weather_factor(team_code: str, when: date | None = None) -> tuple[float, str]:
    """(multiplicative factor to apply on top of the static park factor, human-readable note). 1.0 /
    "indoor or no data" when the park is excluded or the forecast couldn't be loaded."""
    code = team_code.upper()
    if code in DOME_OR_UNKNOWN_ROOF:
        return 1.0, "roof/dome - outdoor weather not applied"
    fc = fetch_forecast(code, when)
    if not fc or fc.get("temp_f") is None:
        return 1.0, "forecast unavailable - no adjustment"
    temp = fc["temp_f"]
    shift = max(-MAX_TEMP_SHIFT, min(MAX_TEMP_SHIFT, (temp - LEAGUE_AVG_TEMP_F) * TEMP_WOBA_PER_DEGREE))
    factor = 1.0 + shift
    wind_note = f", wind {fc['wind_mph']:.0f} mph (not modeled - see module docstring)" if fc.get("wind_mph") is not None else ""
    return factor, f"{temp:.0f}°F{wind_note}"