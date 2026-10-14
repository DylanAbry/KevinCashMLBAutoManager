"""A persistent in-game session: unlike --situation (one-off, stateless), this remembers which relievers
and bench players have already been used, tracks pitch counts and times-through-the-order as the game
actually unfolds, and updates its recommendations each time you report what happened. This is the
foundation the CLI's --live mode is built on; an LLM wrapper could later sit on top of it and translate
free-text play-by-play into the same calls this class already exposes.
"""
from dataclasses import dataclass, field

from decisions.bullpen import AutoPolicy, leverage
from decisions.strategy import choose_defensive_sub, choose_pinch_hitter, should_bunt, should_ibb
from sim.types import Ctx


@dataclass
class Session:
    """our_batters/our_bench are mutated in place as pinch hits and defensive subs happen, so later
    recommendations reflect who's actually left. our_staff is mutated the same way for pitching changes."""
    our_batters: list                 # BatterModel, in batting order (index 0-8)
    our_bench: list                   # BenchPlayer, remaining bench
    our_staff: list                   # PitcherModel, staff[0] is the starter
    bench_used: set = field(default_factory=set)
    staff_used: set = field(default_factory=lambda: {0})
    cur: int = 0                      # index into our_staff of who's currently pitching
    pc: float = 0.0                   # this pitcher's pitch count this game
    bf: int = 0                       # batters faced by the current pitcher, this appearance
    faced: list = field(default_factory=list)   # faced[pitcher_idx][batter_idx] = times seen this game
    policy: object = field(default_factory=AutoPolicy)
    log: list = field(default_factory=list)     # (situation, recommendation) pairs, for a post-game review
    _pending: tuple | None = field(default=None, repr=False)   # (pitcher_idx, batter_idx) not yet committed

    def _commit_pending(self) -> None:
        """The at-bat asked about last time has now happened (a new one is being asked about): count it
        toward times-faced and the three-batter minimum before answering the new question."""
        if self._pending is not None:
            p_idx, b_idx = self._pending
            if p_idx == self.cur:                      # a pitching change since then already reset these
                self.bf += 1
                self.faced[p_idx][b_idx] += 1
        self._pending = None

    def __post_init__(self):
        if not self.faced:
            self.faced = [[0] * 9 for _ in self.our_staff]

    # --- reporting what actually happened, so later advice reflects reality --------------------------------
    def report_pitch_count(self, pitches: float) -> None:
        self.pc = pitches

    def report_batter_faced(self, batter_slot: int) -> None:
        self.bf += 1
        self.faced[self.cur][batter_slot] += 1

    def change_pitcher(self, name_substr: str) -> str:
        from data.names import find_by_name
        p = find_by_name(self.our_staff, name_substr, "our pitching staff", get=lambda x: x.name)
        idx = self.our_staff.index(p)
        if idx in self.staff_used:
            return f"[note] {p.name} has already appeared this game - re-entering isn't legal, but recording it anyway."
        self.cur, self.pc, self.bf = idx, 0.0, 0
        self.staff_used.add(idx)
        self._pending = None
        return f"{p.name} is now pitching."

    def pinch_hit(self, slot_1_based: int, name_substr: str) -> str:
        from data.names import find_by_name
        b = find_by_name(self.our_bench, name_substr, "our bench", get=lambda x: x.name)
        i = self.our_bench.index(b)
        if i in self.bench_used:
            return f"[note] {b.name} is already in the game."
        self.bench_used.add(i)
        self.our_batters[slot_1_based - 1] = b.as_batter()
        return f"{b.name} now hits in the #{slot_1_based} slot."

    def defensive_sub(self, slot_1_based: int, name_substr: str) -> str:
        from data.names import find_by_name
        b = find_by_name(self.our_bench, name_substr, "our bench", get=lambda x: x.name)
        i = self.our_bench.index(b)
        if i in self.bench_used:
            return f"[note] {b.name} is already in the game."
        self.bench_used.add(i)
        self.our_batters[slot_1_based - 1] = b.as_batter()
        return f"{b.name} is now in the game defensively (and bats #{slot_1_based})."

    # --- recommendations, given the situation right now --------------------------------------------------
    def recommend(self, inning: int, outs: int, runners: str, lead: int, batter_slot_1_based: int,
                 fielding: bool, first_batter_of_half: bool = False, opp_woba_vs: list | None = None) -> dict:
        """fielding=True means WE are on defense this half-inning (pitching change / IBB / defensive sub apply);
        fielding=False means WE are at bat (bunt / pinch hit apply). opp_woba_vs, if given, is the CURRENT
        opposing pitcher's expected wOBA against each of our 9 hitters (0-indexed) - needed for offense-side calls;
        without it, bunt/pinch-hit advice falls back to each hitter's plain talent level."""
        mask = sum({"1": 1, "2": 2, "3": 4}.get(ch, 0) for ch in runners if ch in "123")
        idx = (batter_slot_1_based - 1) % 9
        out = {"inning": inning, "outs": outs, "runners": runners, "lead": lead, "batter": batter_slot_1_based}
        li = leverage(inning, lead if fielding else -lead, mask, outs, fielding)
        out["leverage"] = li

        if fielding:
            self._commit_pending()
            ctx = Ctx(inning, outs, mask, lead, idx, first_batter_of_half, self.cur, self.pc, self.bf,
                      0, 0, self.faced[self.cur], self.our_staff, self.staff_used, True)
            self._pending = (self.cur, idx)
            out["pitching"] = self.policy.evaluate(ctx)
            nxt = (idx + 1) % 9
            out["ibb"] = should_ibb("auto", inning, outs, mask, lead,
                                    self.our_batters[idx].woba_starter, self.our_batters[nxt].woba_starter) \
                if mask % 2 == 0 else False           # first base must be open
            pick = choose_defensive_sub("auto", self.our_bench, self.bench_used, idx, inning, lead)
            out["defensive_sub"] = (self.our_bench[pick[0]].name, pick[1]) if pick else None
        else:
            batter_woba = (opp_woba_vs[idx] if opp_woba_vs else self.our_batters[idx].woba_starter)
            out["bunt"] = should_bunt("auto", inning, outs, mask, lead, batter_woba)
            pitcher_throws = "R"          # unknown from here without more context; pass explicitly if it matters
            pick = choose_pinch_hitter("auto", self.our_bench, self.bench_used, idx, self.our_batters,
                                       pitcher_throws, inning, li, lead)
            out["pinch_hit"] = (self.our_bench[pick[0]].name, pick[1]) if pick else None
        self.log.append(out)
        return out