"""Baserunning rules for one plate appearance, shared by the game simulator and the exact lineup evaluator.

transitions(bases, outs, outcome) -> ((probability, dests), ...) where dests has 4 entries, one per
"origin": [batter, runner on 1B, runner on 2B, runner on 3B]. Each entry is:
    -1 = no runner there,  0 = out,  1/2/3 = ends on that base,  4 = scores.
`bases` is a bitmask: 1 = runner on 1B, 2 = runner on 2B, 4 = runner on 3B.
"""
from functools import lru_cache
from itertools import accumulate

from sim.pa_model import BB, HR, K, OUT, ROE, S1, S2, S3

P_GIDP = 0.11                    # ground-ball double play with a runner on 1st and < 2 outs
P_SCORE_FROM_3B_ON_OUT = 0.42    # sacrifice fly / productive ground out
P_ADV_2B_TO_3B_ON_OUT = 0.22
P_ADV_1B_TO_2B_ON_OUT = 0.18
P_SCORE_FROM_2B_ON_1B = 0.60
P_1B_TO_3B_ON_1B = 0.28
P_SCORE_FROM_1B_ON_2B = 0.40


def _stay(b1, b2, b3):
    return (1 if b1 else -1, 2 if b2 else -1, 3 if b3 else -1)


def _single(b1, b2, b3):
    out = []
    r3 = 4 if b3 else -1
    opts2 = [(P_SCORE_FROM_2B_ON_1B, 4), (1 - P_SCORE_FROM_2B_ON_1B, 3)] if b2 else [(1.0, -1)]
    for p2, d2 in opts2:
        if not b1:
            opts1 = [(1.0, -1)]
        elif d2 == 3:
            opts1 = [(1.0, 2)]
        else:
            opts1 = [(P_1B_TO_3B_ON_1B, 3), (1 - P_1B_TO_3B_ON_1B, 2)]
        for p1, d1 in opts1:
            out.append((p2 * p1, (1, d1, d2, r3)))
    return out


def _double(b1, b2, b3):
    r2, r3 = (4 if b2 else -1), (4 if b3 else -1)
    opts1 = [(P_SCORE_FROM_1B_ON_2B, 4), (1 - P_SCORE_FROM_1B_ON_2B, 3)] if b1 else [(1.0, -1)]
    return [(p, (2, d1, r2, r3)) for p, d1 in opts1]


def _walk(b1, b2, b3):
    d1 = 2 if b1 else -1
    d2 = (3 if b1 else 2) if b2 else -1
    d3 = (4 if (b1 and b2) else 3) if b3 else -1
    return [(1.0, (1, d1, d2, d3))]


def _out(b1, b2, b3, outs):
    stay = _stay(b1, b2, b3)
    if outs >= 2:
        return [(1.0, (0, *stay))]
    res, p_normal = [], 1.0
    if b1:
        dp = (0, 0, 3 if b2 else -1, 4 if b3 else -1) if outs == 0 else (0, 0, stay[1], stay[2])
        res.append((P_GIDP, dp))
        p_normal = 1 - P_GIDP
    opts3 = [(P_SCORE_FROM_3B_ON_OUT, 4), (1 - P_SCORE_FROM_3B_ON_OUT, 3)] if b3 else [(1.0, -1)]
    for p3, d3 in opts3:
        if not b2:
            opts2 = [(1.0, -1)]
        elif d3 == 3:
            opts2 = [(1.0, 2)]
        else:
            opts2 = [(P_ADV_2B_TO_3B_ON_OUT, 3), (1 - P_ADV_2B_TO_3B_ON_OUT, 2)]
        for p2, d2 in opts2:
            if not b1:
                opts1 = [(1.0, -1)]
            elif d2 == 2:
                opts1 = [(1.0, 1)]
            else:
                opts1 = [(P_ADV_1B_TO_2B_ON_OUT, 2), (1 - P_ADV_1B_TO_2B_ON_OUT, 1)]
            for p1, d1 in opts1:
                res.append((p_normal * p3 * p2 * p1, (0, d1, d2, d3)))
    return res


@lru_cache(maxsize=None)
def transitions(bases: int, outs: int, outcome: int) -> tuple:
    b1, b2, b3 = bool(bases & 1), bool(bases & 2), bool(bases & 4)
    if outcome == K:
        res = [(1.0, (0, *_stay(b1, b2, b3)))]
    elif outcome == OUT:
        res = _out(b1, b2, b3, outs)
    elif outcome == BB:
        res = _walk(b1, b2, b3)
    elif outcome in (S1, ROE):
        res = _single(b1, b2, b3)
    elif outcome == S2:
        res = _double(b1, b2, b3)
    elif outcome == S3:
        res = [(1.0, (3, 4 if b1 else -1, 4 if b2 else -1, 4 if b3 else -1))]
    elif outcome == HR:
        res = [(1.0, (4, 4 if b1 else -1, 4 if b2 else -1, 4 if b3 else -1))]
    else:
        raise ValueError(outcome)
    return tuple(res)


@lru_cache(maxsize=None)
def transition_table(bases: int, outs: int, outcome: int) -> tuple:
    """(cumulative probabilities, dests list) for fast sampling."""
    tr = transitions(bases, outs, outcome)
    cum = list(accumulate(p for p, _ in tr))
    cum[-1] = 1.0
    return cum, [d for _, d in tr]


@lru_cache(maxsize=None)
def summary(bases: int, outs: int, outcome: int) -> tuple:
    """Anonymous-runner view: ((prob, new_outs, new_bases, runs), ...) - used by the exact lineup evaluator."""
    out = []
    for p, d in transitions(bases, outs, outcome):
        new_outs = outs + sum(1 for x in d if x == 0)
        runs = sum(1 for x in d if x == 4)
        mask = sum(1 << (x - 1) for x in d if 1 <= x <= 3)
        out.append((p, new_outs, mask, runs))
    return tuple(out)
