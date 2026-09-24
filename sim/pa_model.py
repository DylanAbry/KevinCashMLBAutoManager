"""Plate-appearance outcome model.

Each hitter has an outcome PROFILE (K, BB+HBP, 1B, 2B, 3B, HR per PA), taken from his own season line and
shrunk toward league rates (each component with its own stabilisation point). Given a target wOBA for a
specific matchup, we scale the positive outcomes by one factor so the hitter hits that target while
keeping his shape: a singles/contact hitter stays a contact hitter, a slugger stays a slugger.
A small share of ball-in-play outs become "reached on error".
"""
from bisect import bisect
from functools import lru_cache
from itertools import accumulate

K, OUT, BB, S1, S2, S3, HR, ROE = range(8)
OUTCOMES = ("K", "OUT", "BB", "1B", "2B", "3B", "HR", "ROE")

WEIGHTS = (0.695, 0.88, 1.25, 1.58, 2.03)                        # wOBA weights: BB, 1B, 2B, 3B, HR
LEAGUE_RATES = (0.225, 0.096, 0.140, 0.043, 0.004, 0.030)        # K, BB, 1B, 2B, 3B, HR per PA
PRIOR_PA = (60, 120, 300, 300, 600, 170)                         # PA of league prior for each component
ROE_SHARE_OF_BIP_OUTS = 0.02


def shrunk_profile(season: dict | None) -> tuple:
    pa = (season or {}).get("pa", 0)
    if not pa:
        return LEAGUE_RATES
    counts = [season.get(k, 0.0) for k in ("k", "bb", "s1", "s2", "s3", "hr")]
    return tuple(round((c + pr * lg) / (pa + pr), 5) for c, pr, lg in zip(counts, PRIOR_PA, LEAGUE_RATES))


@lru_cache(maxsize=None)
def _probs(profile: tuple, target: float) -> tuple:
    k, bb, s1, s2, s3, hr = profile
    pos = (bb, s1, s2, s3, hr)
    prof_woba = sum(w * p for w, p in zip(WEIGHTS, pos))
    f = min(max(target / prof_woba, 0.4), 2.0) if prof_woba > 0 else 1.0
    scaled = [p * f for p in pos]
    if sum(scaled) > 0.75:
        scaled = [p * 0.75 / sum(scaled) for p in scaled]
    outs = 1 - sum(scaled)
    bip_profile = max(1 - k - sum(pos), 0.02)
    k_p = outs * k / (k + bip_profile)
    bip = outs - k_p
    roe = bip * ROE_SHARE_OF_BIP_OUTS
    return (k_p, bip - roe, *scaled, roe)


def outcome_probs(profile: tuple, target_woba: float) -> tuple:
    return _probs(profile, round(target_woba, 4))


@lru_cache(maxsize=None)
def _cum(profile: tuple, target: float) -> list:
    cum = list(accumulate(_probs(profile, target)))
    cum[-1] = 1.0
    return cum


def sample_outcome(profile: tuple, target_woba: float, rng) -> int:
    return bisect(_cum(profile, round(target_woba, 4)), rng.random())
