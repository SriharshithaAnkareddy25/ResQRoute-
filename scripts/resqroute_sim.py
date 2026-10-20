"""Person C: resource-condition simulator.

Generates synthetic device states (latency budget, free memory, network, battery, queue depth) so the
controller can be tested under varied conditions without edge hardware. Column names match 08_router.py,
so its feature list and baselines work on the output unchanged.

A *preset* is a named distribution over device states. "mixed" is the distribution the controller is
trained on; the others stress one resource at a time and are used for the per-scenario evaluation.
"""
from dataclasses import dataclass

import numpy as np

BIG = ["efficientnet", "resnet18"]
SMALL = "mobilenet"
LOW_BATTERY = 0.2   # below this the device halves its latency budget (power saving)


@dataclass(frozen=True)
class Preset:
    budgets: tuple      # candidate response-time targets (ms), drawn uniformly
    mems: tuple         # candidate free-memory values (MB), drawn uniformly
    p_net: float        # probability the uplink to the human reviewers / server is up
    battery: tuple      # (low, high) uniform
    queue: tuple        # (low, high) integer queue depth
    note: str = ""


PRESETS = {
    "mixed": Preset((60, 180, 400), (150, 250, 500), 0.7, (0.05, 1.0), (0, 50), "training distribution"),
    "normal": Preset((180, 400), (250, 500), 0.95, (0.5, 1.0), (0, 10), "healthy device"),
    "tight_latency": Preset((60,), (250, 500), 0.9, (0.5, 1.0), (0, 20), "60 ms response target"),
    "low_memory": Preset((180, 400), (150,), 0.9, (0.5, 1.0), (0, 20), "150 MB free: ResNet-18 never fits"),
    "low_battery": Preset((180, 400), (250, 500), 0.9, (0.05, 0.2), (0, 20), "power saving halves the budget"),
    "degraded_network": Preset((60, 180, 400), (150, 250, 500), 0.1, (0.3, 1.0), (0, 50), "uplink down 90% of the time"),
    "disaster_surge": Preset((60, 180), (150, 250), 0.3, (0.05, 0.5), (30, 100), "everything bad at once"),
}


def effective_budget(budget_ms, battery):
    return np.where(np.asarray(battery) < LOW_BATTERY, np.asarray(budget_ms) * 0.5, budget_ms)


def fits(t_small_ms, t_big_ms, mem_big_mb, budget_ms, battery, mem_free_mb):
    """Can the small model plus escalation to a bigger one finish inside the budget, in the free memory?"""
    return ((t_small_ms + t_big_ms <= effective_budget(budget_ms, battery)) & (mem_big_mb <= mem_free_mb)).astype(int)


def add_fits(rep):
    for m in BIG:
        rep[f"fits_{m}"] = fits(rep[f"{SMALL}_time_ms"], rep[f"{m}_time_ms"], rep[f"{m}_mem_mb"],
                                rep.budget_ms, rep.battery, rep.mem_free_mb)
    return rep


def simulate(df, k, rng, preset="mixed"):
    """Repeat each crop k times, each under an independently drawn device state from the preset."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    rep = df.loc[df.index.repeat(k)].reset_index(drop=True)
    n = len(rep)
    rep["budget_ms"] = rng.choice(p.budgets, n)
    rep["mem_free_mb"] = rng.choice(p.mems, n)
    rep["network_ok"] = (rng.random(n) < p.p_net).astype(int)
    rep["battery"] = rng.uniform(*p.battery, n).round(2)
    rep["queue_len"] = rng.integers(p.queue[0], p.queue[1] + 1, n)
    rep["scenario"] = preset if isinstance(preset, str) else "custom"
    return add_fits(rep)
