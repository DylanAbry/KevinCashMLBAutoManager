from models.matchup import (TEAM_PA_PER_GAME, WOBA_SCALE, bullpen_woba, expected_woba, risp_delta_woba,
                            steal_rates)
from sim.pa_model import shrunk_profile
from sim.types import BatterModel, TeamSide


def make_batter(h, opp_starter, opp_defense_rpg: float = 0.0) -> BatterModel:
    """opp_defense_rpg: runs per game the OPPOSING defense saves; it lowers this hitter's wOBA per PA."""
    shift = -opp_defense_rpg * WOBA_SCALE / TEAM_PA_PER_GAME
    attempt, success = steal_rates(h)
    return BatterModel(
        name=h.name,
        profile=shrunk_profile(h.season),
        woba_starter=expected_woba(h, opp_starter) + shift,
        woba_pen=bullpen_woba(h) + shift,
        risp_delta=risp_delta_woba(h),
        steal_attempt=attempt,
        steal_success=success,
    )


def make_side(name, spots, opp_starter, opp_defense_rpg: float, starter_innings: int = 6) -> TeamSide:
    return TeamSide(name, [make_batter(s.hitter, opp_starter, opp_defense_rpg) for s in spots], starter_innings)
