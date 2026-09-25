"""Pitching-change policies.

TypicalPolicy : how an average manager behaves - starter goes to a pitch limit, then relievers by role
                (closer in save spots, setup men in close games, mop-up men in blowouts), roughly one inning each.
AutoPolicy    : the AutoManager - pulls a pitcher when a reliever is expected to allow meaningfully less wOBA over the
                next three hitters (after times-through-the-order, fatigue and rest penalties), weighted by how
                high-leverage the moment is. It keeps the best relievers for the highest-leverage spots, picks the
                best matchup among the relievers allowed at this leverage, and respects the three-batter minimum
                (enforced by the simulator).
Both are driven by the same context object, so the exact logic used in simulation is what the CLI advisor explains.
"""
from math import exp, pi, sqrt

from models.run_expectancy import RE24, VAR_PER_HALF_INNING

# Leverage a reliever must see before he's used, by his quality rank among the relievers still available.
TIERS = (1.5, 1.15, 0.9, 0.7)
_LI_BASE = 0.129          # d(win prob)/d(run) in an average mid-game spot, used to normalise leverage to ~1


def leverage(inning: int, lead: int, mask: int, outs: int, fld_home: bool) -> float:
    """Approximate leverage index for the fielding team (1.0 = average). Normal-approximation win probability."""
    fut_away = max(9 - inning, 0)
    fut_home = max(9 - inning, 0) + (1 if fld_home else 0)
    fut_f, fut_b = (fut_home, fut_away) if fld_home else (fut_away, fut_home)
    m = lead - RE24[(mask, min(outs, 2))] + 0.5 * (fut_f - fut_b)
    sd = sqrt(VAR_PER_HALF_INNING * (fut_f + fut_b + 1))
    z = m / sd
    li = exp(-0.5 * z * z) / sqrt(2 * pi) / sd / _LI_BASE
    return min(max(li, 0.05), 4.0)


def _candidates(c) -> list[int]:
    return [j for j, p in enumerate(c.staff) if j not in c.used and p.available and not p.is_starter]


class TypicalPolicy:
    def __init__(self, pitch_limit: int = 100):
        self.limit = pitch_limit

    def choose(self, c):
        cur = c.staff[c.cur]
        if cur.is_starter:
            if not (c.pc >= self.limit or (c.first and c.inning >= 7 and c.pc >= 85)
                    or (c.runs_game >= 5 and c.pc >= 60)):
                return None
        elif not c.first and not (c.pc >= 30 or c.runs_app >= 3):
            return None
        avail = sorted(_candidates(c), key=lambda j: c.staff[j].role_rank)
        if not avail:
            return None
        if c.inning <= 5 and cur.is_starter:                      # starter knocked out early -> long man
            longs = [j for j in avail if c.staff[j].role == "LR"]
            if longs:
                return longs[0]
        if c.inning >= 9 and 0 < c.lead <= 3:                     # save situation
            return avail[0]
        if c.inning >= 7 and abs(c.lead) <= 3:                    # close game late -> best non-closer available
            return avail[1] if len(avail) > 1 else avail[0]
        if abs(c.lead) >= 5:                                      # blowout -> mop-up man
            return avail[-1]
        return avail[min(2, len(avail) - 1)]


class AutoPolicy:
    def __init__(self, pull_base: float = 0.008, pull_slope: float = 0.0015, hard_cap: int = 110,
                 relief_need: float = 0.010, mid_inning_need: float = 0.016):
        self.pull_base, self.pull_slope, self.hard_cap = pull_base, pull_slope, hard_cap
        self.relief_need, self.mid_need = relief_need, mid_inning_need

    @staticmethod
    def _look(p, batters, faced, pc) -> float:
        """Expected wOBA allowed to the next few hitters, including tiredness growing as the pitch count rises."""
        return sum(p.woba_vs[b] + p.penalty(faced[b] if faced else 0, pc + 3.9 * k)
                   for k, b in enumerate(batters)) / len(batters)

    def evaluate(self, c) -> dict:
        staff, cur = c.staff, c.staff[c.cur]
        nxt = [(c.batter + k) % 9 for k in range(3)]
        li = leverage(c.inning, c.lead, c.mask, c.outs, c.fld_home)
        w_cur = self._look(cur, nxt, c.faced, c.pc)
        ranked = sorted(_candidates(c), key=lambda j: staff[j].quality())
        options = []
        for r, j in enumerate(ranked):
            thr = TIERS[r] if r < len(TIERS) else 0.0
            options.append({"j": j, "look": self._look(staff[j], nxt, None, 0.0), "thr": thr, "eligible": thr <= li})
        res = {"li": li, "w_cur": w_cur, "options": options, "action": "stay", "pick": None,
               "gain": 0.0, "need": 0.0, "reason": ""}
        if not options:
            res["reason"] = "no relievers left"
            return res
        elig = [o for o in options if o["eligible"]]
        emergency = (cur.is_starter and (c.pc >= self.hard_cap or c.runs_game >= 7)) or (not cur.is_starter and c.pc >= 40)
        pool = elig or (options if emergency else [])
        if not pool:
            res["reason"] = f"leverage {li:.2f} is too low to use the remaining arms"
            return res
        best = min(pool, key=lambda o: o["look"])
        gain = w_cur - best["look"]
        if cur.is_starter:
            need = self.pull_base + self.pull_slope * max(0, 8 - c.inning)
        elif c.first:
            need = self.relief_need + (0.006 if c.pc <= 20 else 0.0)
        else:
            need = self.mid_need if li >= 1.0 else 99.0
        res.update(gain=gain, need=need, best=best["j"])
        if emergency:
            res.update(action="pull", pick=best["j"], reason="workload/blow-up limit reached")
        elif gain * li >= need:
            res.update(action="pull", pick=best["j"],
                       reason=f"{gain:.3f} wOBA edge x leverage {li:.2f} beats the {need:.3f} bar")
        else:
            res["reason"] = f"edge {gain:+.3f} x leverage {li:.2f} is below the {need:.3f} bar"
        return res

    def choose(self, c):
        cur = c.staff[c.cur]
        if cur.is_starter and c.pc < 55 and c.runs_game < 5:
            return None
        r = self.evaluate(c)
        return r["pick"] if r["action"] == "pull" else None