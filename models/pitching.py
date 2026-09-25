"""Reliever workload -> availability and tiredness."""


def rest_status(recent: list[tuple[int, int]]) -> tuple[bool, float, str]:
    """recent = [(days_ago, pitches), ...]. Returns (available, wOBA penalty for being tired, note)."""
    by_day: dict[int, int] = {}
    for d, p in recent:
        by_day[d] = by_day.get(d, 0) + p
    y, d2, d3 = by_day.get(1, 0), by_day.get(2, 0), by_day.get(3, 0)
    if y and d2 and d3:
        return False, 0.0, "pitched 3 days in a row"
    if y >= 35 or y + d2 >= 55:
        return False, 0.0, f"heavy workload ({y} pitches yesterday, {d2} the day before)"
    tired, notes = 0.0, []
    if y >= 15:
        tired += 0.008
        notes.append(f"{y} pitches yesterday")
    elif y > 0:
        tired += 0.004
        notes.append(f"{y} pitches yesterday")
    if d2 >= 25:
        tired += 0.003
        notes.append(f"{d2} pitches 2 days ago")
    return True, tired, ", ".join(notes) or "rested"