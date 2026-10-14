"""Park factors sourced from Baseball Savant's "Statcast Park Factors" leaderboard
(https://baseballsavant.mlb.com/leaderboard/statcast-park-factors), filtered to
FACTOR=Year, SEASON=2026, BAT SIDE=Both, CONDITION=All, ROLLING YEARS=3, PARKS=MLB
(i.e. the 2024-2026 three-year blend, overall wOBA-based "Park Factor" column),
pulled 2026-10-09. Savant reports these on a 100 = neutral scale; they're divided
by 100 below to keep this file's existing 1.00 = neutral convention, since the
rest of the project (models/matchup.py's park_woba_shift) expects that scale.

The Athletics are the one exception: they play at Sutter Health Park, a Triple-A
facility in West Sacramento, which Savant's leaderboard doesn't track at all (it
only covers MLB venues), so there's no 3-year Statcast blend to pull for it.
Public trackers that did compute a 2025 single-season run park factor for it
(e.g. RotoWire's "Z-Files") put it around 117 (100 = neutral), and the small,
short-dimensioned AAA field is generally understood to play as one of the more
hitter-friendly parks in the majors - second only to Coors among these 30. ATH
is set to 1.08 on that basis: clearly hitter-friendly, but shrunk somewhat from
the raw single-season read of ~1.17 since it isn't backed by the same 3-year
sample everyone else here gets. Revisit if more big-league seasons of data (or
a future Savant listing) give a firmer number.
"""

PARK_FACTOR = {
    "COL": 1.12, "BOS": 1.03, "MIN": 1.03, "ARI": 1.03, "PHI": 1.02, "WSH": 1.02,
    "BAL": 1.02, "KC": 1.02, "PIT": 1.01, "CIN": 1.01, "TOR": 1.01, "LAD": 1.01,
    "NYY": 1.01, "MIA": 1.00, "HOU": 1.00, "DET": 1.00, "ATL": 1.00, "CWS": 0.99,
    "LAA": 0.99, "CLE": 0.99, "MIL": 0.98, "CHC": 0.98, "NYM": 0.98, "SD": 0.97,
    "SF": 0.97, "STL": 0.97, "TB": 0.97, "TEX": 0.94, "SEA": 0.92,
    "ATH": 1.08,   # not in Savant's leaderboard - see module docstring
}
DEFAULT_FACTOR = 1.00


def park_factor(team_code: str) -> float:
    return PARK_FACTOR.get(team_code.upper(), DEFAULT_FACTOR)