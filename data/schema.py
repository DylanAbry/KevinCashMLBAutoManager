from dataclasses import dataclass, field

# A split is (OPS, sample_size). Keys are the hand of the opposing player: "L" or "R".
Split = tuple[float, int]


@dataclass
class Hitter:
    id: int
    name: str
    pos: str                 # primary roster position, e.g. "SS", "CF", "DH"
    bat_side: str            # "L", "R", or "S" (switch)
    vs: dict[str, Split] = field(default_factory=dict)        # OPS + PA vs LHP / vs RHP
    pa: int = 0                                                # season plate appearances (Statcast)
    woba: float | None = None                                  # season wOBA
    xwoba: float | None = None                                 # season expected wOBA (Statcast)
    def_runs: dict[str, float] = field(default_factory=dict)  # season fielding runs above avg, by position
    # season counting stats: pa, k, bb (incl. HBP), s1, s2, s3, hr, sb, cs, ops
    season: dict[str, float] = field(default_factory=dict)
    risp: tuple[float, int] | None = None                      # (OPS, PA) with runners in scoring position


@dataclass
class Pitcher:
    id: int
    name: str
    throws: str              # "L" or "R"
    vs: dict[str, Split] = field(default_factory=dict)   # OPS allowed + batters faced vs LHB / vs RHB
