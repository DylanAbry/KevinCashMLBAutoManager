from data.names import norm
from data.schema import Pitcher
from models.matchup import (TEAM_PA_PER_GAME, WOBA_SCALE, bullpen_woba, expected_woba, neutral_pitcher,
                            risp_delta_woba, steal_rates)
from models.pitching import rest_status
from sim.pa_model import shrunk_profile
from sim.types import TTO_STARTER, BatterModel, PitcherModel, TeamSide

GENERIC_PEN_THROWS = "RRRLRLR"     # used only if no bullpen data could be loaded


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


def make_side(name, spots, opp_starter, opp_defense_rpg: float, starter_innings: int = 6,
              staff=None, policy=None) -> TeamSide:
    return TeamSide(name, [make_batter(s.hitter, opp_starter, opp_defense_rpg) for s in spots],
                    starter_innings, staff, policy)


def _assign_roles(models: list[PitcherModel], pitchers: list[Pitcher]) -> None:
    idx = list(range(1, len(models)))
    if not idx:
        return

    def usage(j):
        s = pitchers[j].season
        return s.get("sv", 0) + 0.5 * s.get("hld", 0) + 0.2 * s.get("gf", 0)

    order = sorted(idx, key=lambda j: -usage(j)) if any(usage(j) for j in idx) else sorted(idx, key=lambda j: models[j].quality())
    for rank, j in enumerate(order):
        models[j].role_rank = rank
        models[j].role = "CL" if rank == 0 else "SU" if rank <= 2 else "MR"
    rest = [j for j in order if models[j].role == "MR"]
    if rest:
        lr = max(rest, key=lambda j: pitchers[j].season.get("ip", 0) / max(pitchers[j].season.get("g", 1), 1))
        models[lr].role = "LR"


def make_staff(starter, relievers, opp_hitters, own_defense_rpg: float = 0.0, unavailable=()) -> list[PitcherModel]:
    """Turn real pitchers into simulator PitcherModels against a specific opposing lineup (in batting order).
    own_defense_rpg is the runs per game OUR defense saves; it lowers what the pitchers allow."""
    shift = -own_defense_rpg * WOBA_SCALE / TEAM_PA_PER_GAME
    starter = starter or Pitcher(0, "League-average starter", "R", {})
    if not relievers:
        relievers = [Pitcher(0, f"Reliever {i + 1}", t, {}) for i, t in enumerate(GENERIC_PEN_THROWS)]
    blocked = [norm(u) for u in unavailable if u.strip()]

    def build(p: Pitcher, is_starter: bool) -> PitcherModel:
        m = PitcherModel(p.name, p.throws, is_starter, [expected_woba(h, p) + shift for h in opp_hitters])
        if is_starter:
            m.tto, m.fatigue_start, m.fatigue_per_pitch, m.role, m.role_rank = TTO_STARTER, 85.0, 0.0008, "SP", -1
        else:
            m.available, m.tired, m.note = rest_status(p.recent)
            if any(b in norm(p.name) for b in blocked):
                m.available, m.note = False, "marked unavailable"
        return m

    pitchers = [starter, *relievers]
    models = [build(p, i == 0) for i, p in enumerate(pitchers)]
    _assign_roles(models, pitchers)
    return models