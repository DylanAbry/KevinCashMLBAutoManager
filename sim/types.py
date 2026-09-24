from dataclasses import dataclass

from sim.pa_model import LEAGUE_RATES


@dataclass
class BatterModel:
    name: str
    profile: tuple                    # (K, BB, 1B, 2B, 3B, HR) rates per PA
    woba_starter: float               # expected wOBA vs the opposing starter (defense adjusted)
    woba_pen: float                   # expected wOBA vs the opposing bullpen
    risp_delta: float = 0.0           # wOBA change with runners in scoring position
    steal_attempt: float = 0.0        # P(attempt steal of 2B) per PA with a runner on 1st and 2nd open
    steal_success: float = 0.78

    def woba(self, vs_starter: bool) -> float:
        return self.woba_starter if vs_starter else self.woba_pen


@dataclass
class TeamSide:
    name: str
    batters: list[BatterModel]        # in batting order
    starter_innings: int = 6          # innings THIS team's starter works before its bullpen takes over


def league_batter(woba: float = 0.315, name: str = "League avg", steals: bool = True) -> BatterModel:
    return BatterModel(name, LEAGUE_RATES, woba, woba, 0.0, 0.09 if steals else 0.0, 0.78)
