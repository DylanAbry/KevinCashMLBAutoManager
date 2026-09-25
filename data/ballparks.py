"""Approximate park factors (3-year blends, rough public estimates - not official Statcast/FanGraphs numbers).
1.00 = neutral. Values are directional for tuning, not a precise research-grade dataset."""

PARK_FACTOR = {
    "COL": 1.13, "CIN": 1.05, "BAL": 1.04, "TEX": 1.04, "PHI": 1.03, "BOS": 1.03,
    "CWS": 1.02, "ARI": 1.02, "TOR": 1.01, "MIN": 1.01, "HOU": 1.01, "ATL": 1.00,
    "MIL": 1.00, "WSH": 1.00, "LAA": 1.00, "NYY": 1.00, "CHC": 0.99, "STL": 0.99,
    "KC": 0.98, "TB": 0.98, "LAD": 0.97, "SD": 0.96, "CLE": 0.96, "DET": 0.96,
    "SEA": 0.95, "NYM": 0.95, "PIT": 0.95, "MIA": 0.94, "OAK": 0.94, "SF": 0.90,
}
DEFAULT_FACTOR = 1.00


def park_factor(team_code: str) -> float:
    return PARK_FACTOR.get(team_code.upper(), DEFAULT_FACTOR)