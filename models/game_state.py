from dataclasses import dataclass


@dataclass
class GameState:
    """Everything a manager needs to know about the current moment of a game.
    Later modules (bullpen, pinch hitting) will add pitch counts, bench and bullpen availability."""
    inning: int = 1
    top: bool = True                                        # True = away team batting
    outs: int = 0
    bases: tuple[bool, bool, bool] = (False, False, False)  # 1B, 2B, 3B occupied
    away_score: int = 0
    home_score: int = 0
    away_batter: int = 0                                    # next lineup slot (0-8) due up
    home_batter: int = 0

    @property
    def batting_home(self) -> bool:
        return not self.top

    @property
    def base_index(self) -> int:
        b1, b2, b3 = self.bases
        return int(b1) + 2 * int(b2) + 4 * int(b3)

    @property
    def batting_lead(self) -> int:
        return (self.home_score - self.away_score) if self.batting_home else (self.away_score - self.home_score)

    def __str__(self) -> str:
        runners = "".join(n if on else "-" for n, on in zip("123", self.bases))
        half = "Top" if self.top else "Bot"
        return f"{half} {self.inning}, {self.outs} out, runners {runners}, AWAY {self.away_score} HOME {self.home_score}"