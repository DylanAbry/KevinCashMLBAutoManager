"""Synthetic data so you can test the whole pipeline with no network."""
import random

from .schema import Hitter, Pitcher

_ROSTER = [("C", "R"), ("1B", "L"), ("2B", "R"), ("3B", "R"), ("SS", "S"), ("LF", "L"),
           ("CF", "R"), ("RF", "L"), ("DH", "R"), ("SS", "R"), ("CF", "L"), ("1B", "R"), ("C", "L")]
_SECOND_POS = {"SS": ["2B", "3B"], "2B": ["SS"], "3B": ["1B"], "CF": ["LF", "RF"], "LF": ["RF"], "RF": ["LF"]}


def demo_hitters(seed: int = 7, prefix: str = "Player") -> list[Hitter]:
    rng = random.Random(seed)
    out = []
    for i, (pos, side) in enumerate(_ROSTER):
        woba = rng.uniform(0.290, 0.360)
        pa = rng.randint(120, 620)
        base_ops = woba / 0.43 - 0.02
        def_runs = {}
        if pos not in ("DH", "C"):
            def_runs[pos] = rng.uniform(-6, 9)
            for p2 in _SECOND_POS.get(pos, [])[:1]:
                def_runs[p2] = rng.uniform(-5, 5)
        if i == 1:      # a big, reliable bat with a bad glove: should end up at DH
            woba, pa, def_runs = 0.385, 600, {"1B": -9.0}
        k, bb = rng.uniform(0.15, 0.28), rng.uniform(0.05, 0.13)
        hr, s2, s3 = rng.uniform(0.015, 0.055), rng.uniform(0.03, 0.06), rng.uniform(0.002, 0.008)
        s1 = max(0.08, min(0.20, (woba - 0.695 * bb - 1.25 * s2 - 1.58 * s3 - 2.03 * hr) / 0.88))
        season = {"pa": pa, "k": k * pa, "bb": bb * pa, "s1": s1 * pa, "s2": s2 * pa, "s3": s3 * pa, "hr": hr * pa,
                  "sb": rng.randint(0, 25), "cs": rng.randint(0, 7), "ops": base_ops}
        out.append(Hitter(
            id=i, name=f"{prefix} {i+1}", pos=pos, bat_side=side, pa=pa,
            woba=woba, xwoba=woba + rng.uniform(-0.015, 0.015), def_runs=def_runs, season=season,
            risp=(base_ops + rng.uniform(-0.12, 0.12), int(pa * 0.25)),
            vs={"L": (base_ops + rng.uniform(-0.08, 0.08), int(pa * 0.3)),
                "R": (base_ops + rng.uniform(-0.08, 0.08), int(pa * 0.7))},
        ))
    return out


def demo_pitcher(name: str = "Demo Lefty", throws: str = "L") -> Pitcher:
    return Pitcher(id=999, name=name, throws=throws, vs={"L": (0.640, 180), "R": (0.760, 420)},
                   season={"gs": 25, "g": 25, "ip": 150.0})


def demo_staff(seed: int, prefix: str, starter_name: str, starter_throws: str = "R"):
    """(starter, relievers) with varied quality, roles, and a couple of tired arms."""
    rng = random.Random(seed)
    starter = Pitcher(id=900 + seed, name=starter_name, throws=starter_throws,
                      vs={"L": (rng.uniform(0.66, 0.74), 250), "R": (rng.uniform(0.64, 0.74), 500)},
                      season={"gs": 26, "g": 26, "ip": 155.0})
    relievers = []
    for i in range(8):
        throws = "L" if i in (2, 5) else "R"
        skill = 0.60 + 0.03 * i + rng.uniform(-0.03, 0.03)            # lower OPS allowed = better; i=0 is the closer
        p = Pitcher(id=1000 + seed * 20 + i, name=f"{prefix} RP{i + 1}", throws=throws,
                    vs={"L": (skill + rng.uniform(-0.06, 0.06), rng.randint(80, 200)),
                        "R": (skill + rng.uniform(-0.06, 0.06), rng.randint(120, 260))},
                    season={"gs": 0, "g": 55 - 2 * i, "ip": 58 - 3 * i + (12 if i == 6 else 0),
                            "sv": max(0, 28 - 9 * i), "hld": max(0, 24 - 6 * i), "gf": max(0, 40 - 8 * i)})
        relievers.append(p)
    if seed % 2 == 0:            # give one team a tired setup man and a burned-out middle man
        relievers[1].recent = [(1, 27)]
        relievers[3].recent = [(1, 24), (2, 22), (3, 18)]
    return starter, relievers