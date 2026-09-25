from dataclasses import dataclass

from sim.pa_model import LEAGUE_RATES

# Times-through-the-order penalty (wOBA points) by how many times this hitter has already faced the pitcher.
TTO_STARTER = (0.0, 0.010, 0.025, 0.035)
TTO_RELIEVER = (0.0, 0.006, 0.012, 0.012)
NO_TTO = (0.0, 0.0, 0.0, 0.0)


@dataclass
class BatterModel:
    name: str
    profile: tuple                    # (K, BB, 1B, 2B, 3B, HR) rates per PA
    woba_starter: float               # expected wOBA vs the opposing starter (defense adjusted)
    woba_pen: float                   # expected wOBA vs a generic opposing bullpen
    risp_delta: float = 0.0           # wOBA change with runners in scoring position
    steal_attempt: float = 0.0        # P(attempt steal of 2B) per PA with a runner on 1st and 2nd open
    steal_success: float = 0.78

    def woba(self, vs_starter: bool) -> float:
        return self.woba_starter if vs_starter else self.woba_pen


@dataclass
class PitcherModel:
    name: str
    throws: str
    is_starter: bool
    woba_vs: list[float]              # expected wOBA allowed to each opposing lineup slot (fresh, incl. own defense)
    tto: tuple = TTO_RELIEVER
    fatigue_start: float = 22.0       # pitches before he starts to tire
    fatigue_per_pitch: float = 0.0012
    tired: float = 0.0                # penalty from recent workload
    available: bool = True
    role: str = "MR"                  # SP, CL, SU, MR, LR
    role_rank: int = 9                # 0 = closer, higher = lower in the pecking order
    note: str = ""

    def penalty(self, times_faced: int, pitches: float) -> float:
        p = self.tto[times_faced if times_faced < 3 else 3] + self.tired
        if pitches > self.fatigue_start:
            p += (pitches - self.fatigue_start) * self.fatigue_per_pitch
        return p

    def quality(self) -> float:
        """Average wOBA allowed to this lineup (lower is better)."""
        return sum(self.woba_vs) / len(self.woba_vs) + self.tired


class Ctx:
    """Snapshot handed to a pitching policy before each plate appearance."""
    __slots__ = ("inning", "outs", "mask", "lead", "batter", "first", "cur", "pc", "bf",
                 "runs_app", "runs_game", "faced", "staff", "used", "fld_home")

    def __init__(self, inning, outs, mask, lead, batter, first, cur, pc, bf, runs_app, runs_game,
                 faced, staff, used, fld_home):
        self.inning, self.outs, self.mask, self.lead, self.batter, self.first = inning, outs, mask, lead, batter, first
        self.cur, self.pc, self.bf, self.runs_app, self.runs_game = cur, pc, bf, runs_app, runs_game
        self.faced, self.staff, self.used, self.fld_home = faced, staff, used, fld_home


class InningsPolicy:
    """Legacy policy: the starter (staff[0]) goes N innings, then staff[1] finishes."""

    def __init__(self, starter_innings: int = 6):
        self.n = starter_innings

    def choose(self, c: Ctx):
        if c.cur == 0 and c.first and c.inning > self.n and len(c.staff) > 1:
            return 1
        return None


@dataclass
class TeamSide:
    name: str
    batters: list[BatterModel]        # in batting order
    starter_innings: int = 6          # legacy: innings THIS team's starter works before its bullpen takes over
    staff: list | None = None         # this team's pitchers (PitcherModel); staff[0] starts. None = legacy 2-pitcher staff
    policy: object | None = None      # decides pitching changes; None = InningsPolicy(starter_innings)


def league_batter(woba: float = 0.315, name: str = "League avg", steals: bool = True) -> BatterModel:
    return BatterModel(name, LEAGUE_RATES, woba, woba, 0.0, 0.09 if steals else 0.0, 0.78)