"""Batting order: exact expected-runs search, with optional pinned slots (manager overrides)."""
import unicodedata
from dataclasses import replace

from sim.lineup_eval import expected_runs, optimize_order
from sim.setup import make_batter


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").split())


def parse_pins(pins: list[str], spots) -> dict[int, int]:
    """'Bichette:2' -> {index of that hitter in spots: 1}. Name match is a case/accent-insensitive substring."""
    out: dict[int, int] = {}
    for p in pins:
        name, _, slot = p.rpartition(":")
        if not name or not slot.strip().isdigit() or not 1 <= int(slot) <= 9:
            raise SystemExit(f'Bad --pin "{p}". Use "Name:slot", e.g. --pin "Bichette:2" (slot 1-9)')
        hits = [i for i, s in enumerate(spots) if _norm(name) in _norm(s.hitter.name)]
        if len(hits) != 1:
            names = ", ".join(s.hitter.name for s in spots)
            raise SystemExit(f'--pin "{name}" matched {len(hits)} hitters in the starting nine. Starters: {names}')
        out[hits[0]] = int(slot) - 1
    if len(set(out.values())) != len(out):
        raise SystemExit("Two --pin options use the same batting slot")
    return out


def order_lineup(lineup, opp_pitcher, opp_defense_rpg: float = 0.0, pins: list[str] | None = None,
                 starter_innings: int = 6):
    spots = lineup.spots                                   # already in a reasonable heuristic order
    batters = [make_batter(s.hitter, opp_pitcher, opp_defense_rpg) for s in spots]
    pin_map = parse_pins(pins or [], spots)
    order, runs = optimize_order(batters, pin_map, starter_innings)
    info = {"runs": runs, "heuristic_runs": expected_runs(batters, starter_innings), "free_runs": runs}
    if pin_map:
        _, info["free_runs"] = optimize_order(batters, {}, starter_innings)
    new_spots = [replace(spots[i], slot=k + 1) for k, i in enumerate(order)]
    return replace(lineup, spots=new_spots), info
