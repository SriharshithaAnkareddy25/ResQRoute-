"""Step 10 (Person C): evaluation harness for the six systems in the project plan.

  1 Small only (MobileNetV3-S)          4 Resource-aware cascade, no safety risk
  2 Advanced only (ResNet-18)           5 ResQRoute controller (no queue: deferred cases keep the small answer)
  3 Fixed-confidence cascade            6 ResQRoute controller + deferred priority queue (P = U x R x D x A)

All systems are scored on the TEST split, each crop under K simulated device states. Metrics: accuracy,
macro-F1, severe-damage recall, dangerous-FN rate (truly severe -> "no-damage"), latency (mean/p95), peak
memory, share of cases per stage, human-review load, resource-violation rate, and per-scenario results
including the degraded-network preset. Human review is assumed correct (and costs a human, so its load is
reported). Deferred cases are resolved by EfficientNet-B0 only if the queue reaches them in time, by running both
bigger models and keeping the more confident answer; otherwise the small model's answer stands.

  python scripts/10_evaluate.py
"""
import json
import heapq

import numpy as np
import pandas as pd

from pipeline import R, FEATS, get_controller, guard, ACTIONS
from common import CLASSES_3, OUTPUTS, SEED
import resqroute_sim as sim

SEVERE = CLASSES_3.index("severe-damage")
K = 8
QUEUE_RATE = 0.05        # deferred items the spare capacity can finish per arriving case
QUEUE_MAX = 100          # queue capacity; when full the lowest-priority item is dropped
SYSTEMS = ["1 Small only", "2 Advanced only (ResNet-18)", "3 Fixed-confidence cascade",
           "4 Resource-aware, no risk", "5 Controller (no queue)", "6 Controller + priority queue"]


# ---------------------------------------------------------------- deferred priority queue simulation
def queue_sim(rep, actions, policy="priority", rate=QUEUE_RATE, max_len=QUEUE_MAX, seed=SEED, return_wait=False):
    """Process arrivals in order; every arrival frees `rate` units of spare capacity, and each whole unit
    resolves one queued item. policy: priority (max P), fifo, random. Returns a boolean 'resolved' mask."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rep))                  # arrival order
    P = R.priority(rep).to_numpy()
    resolved = np.zeros(len(rep), bool)
    wait = np.full(len(rep), np.nan)          # arrivals between deferral and resolution
    arrived = {}
    heap, credit, tick = [], 0.0, 0

    def key(i):
        if policy == "priority":
            return -P[i]
        if policy == "fifo":
            return tick
        return rng.random()

    for i in order:
        tick += 1
        if actions[i] == 3:
            arrived[i] = tick
            heapq.heappush(heap, (key(i), i))
            if len(heap) > max_len:                    # overflow: drop the least urgent
                heap.remove(max(heap))
                heapq.heapify(heap)
        credit += rate
        while credit >= 1 and heap:
            credit -= 1
            j = heapq.heappop(heap)[1]
            resolved[j], wait[j] = True, tick - arrived[j]
        if not heap:
            credit = min(credit, 1.0)                  # idle capacity does not bank up
    return (resolved, wait) if return_wait else resolved


# ---------------------------------------------------------------- scoring
def predictions(rep, actions, resolved=None):
    y = rep.y.to_numpy()
    pred = rep[f"{sim.SMALL}_yhat"].to_numpy().copy()
    t_imm = rep[f"{sim.SMALL}_time_ms"].to_numpy().astype(float).copy()
    t_evt = t_imm.copy()
    mem = rep[f"{sim.SMALL}_mem_mb"].to_numpy().astype(float).copy()
    for a, m in zip([1, 2], sim.BIG):
        s = actions == a
        pred[s] = rep[f"{m}_yhat"].to_numpy()[s]
        t_imm[s] += rep[f"{m}_time_ms"].to_numpy()[s]
        t_evt[s] += rep[f"{m}_time_ms"].to_numpy()[s]
        mem[s] = np.maximum(mem[s], rep[f"{m}_mem_mb"].to_numpy()[s])
    d = (actions == 3) if resolved is None else ((actions == 3) & resolved)
    # A resolved deferral runs both bigger models (resources are free by definition) and keeps the more
    # confident answer.
    conf = {m: np.stack(rep[f"{m}_p"].to_numpy()).max(1) for m in sim.BIG}
    use_e = conf["efficientnet"] >= conf["resnet18"]
    best = np.where(use_e, rep["efficientnet_yhat"].to_numpy(), rep["resnet18_yhat"].to_numpy())
    pred[d] = best[d]
    t_evt[d] += (rep["efficientnet_time_ms"] + rep["resnet18_time_ms"]).to_numpy()[d]
    h = actions == 4
    pred[h] = y[h]
    eff = sim.effective_budget(rep.budget_ms.to_numpy(), rep.battery.to_numpy())
    viol = ((t_imm > eff) | (mem > rep.mem_free_mb.to_numpy())) & (actions != 3) & ~h
    return pred, t_imm, t_evt, mem, viol


def macro_f1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * prec * rec / np.maximum(prec + rec, 1e-12)))


def score(rep, actions, resolved=None, name=""):
    pred, t_imm, t_evt, mem, viol = predictions(rep, actions, resolved)
    y = rep.y.to_numpy()
    sev = y == SEVERE
    auto = actions != 4
    unresolved = (actions == 3) if resolved is None else ((actions == 3) & ~resolved)
    return {
        "system": name, "accuracy": (pred == y).mean(), "macro_F1": macro_f1(y, pred),
        "severe_recall": (pred[sev] == SEVERE).mean(), "dangerous_FN_%": 100 * (pred[sev] == 0).mean(),
        "auto_accuracy": (pred[auto] == y[auto]).mean(),
        "lat_mean_ms": t_imm.mean(), "lat_p95_ms": np.percentile(t_imm, 95), "eventual_ms": t_evt.mean(),
        "peak_mem_mb": mem.mean(), "violations_%": 100 * viol.mean(),
        "stage1_only_%": 100 * (actions == 0).mean(), "stage2_effnet_%": 100 * (actions == 1).mean(),
        "stage3_resnet_%": 100 * (actions == 2).mean(), "deferred_%": 100 * (actions == 3).mean(),
        "deferred_unresolved_%": 100 * unresolved.mean(), "human_review_%": 100 * (actions == 4).mean(),
    }


def run_systems(rep, clf, t_fixed, t_res, rate=QUEUE_RATE):
    n = len(rep)
    ctrl = guard(clf.predict(rep[FEATS]), rep)
    sets = [
        (np.zeros(n, int), None), (np.full(n, 2), None),
        (R.fixed_cascade(rep, t_fixed), None), (R.resource_cascade(rep, t_res), None),
        (ctrl, None), (ctrl, queue_sim(rep, ctrl, "priority", rate)),
    ]
    # systems without the queue never resolve a deferral
    return [score(rep, a, np.zeros(n, bool) if r is None else r, s) for (a, r), s in zip(sets, SYSTEMS)], ctrl


def tune(val, fn):
    grid = np.round(np.arange(0.40, 0.99, 0.02), 2)
    return float(max(grid, key=lambda t: macro_f1(val.y.to_numpy(), predictions(val, fn(val, t), np.zeros(len(val), bool))[0])))


def bootstrap(rep, actions_list, resolved_list, reps=200, seed=SEED):
    """95% CI over crops (all K device states of a crop move together) for macro-F1 and dangerous-FN."""
    rng = np.random.default_rng(seed)
    ids = rep.image_id.to_numpy()
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    y = rep.y.to_numpy()
    preds = [predictions(rep, a, r)[0] for a, r in zip(actions_list, resolved_list)]
    out = {s: ([], []) for s in SYSTEMS}
    for _ in range(reps):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        sev = y[idx] == SEVERE
        for s, p in zip(SYSTEMS, preds):
            out[s][0].append(macro_f1(y[idx], p[idx]))
            out[s][1].append(100 * (p[idx][sev] == 0).mean())
    return {s: {"macro_F1_ci": np.percentile(v[0], [2.5, 97.5]).round(3).tolist(),
                "dangerous_FN_ci": np.percentile(v[1], [2.5, 97.5]).round(2).tolist()} for s, v in out.items()}


def main():
    rng = np.random.default_rng(SEED)
    log = R.load()
    clf = get_controller(log)
    val = sim.simulate(log[log.split == "val"], K, np.random.default_rng(SEED), "mixed")
    t_fixed, t_res = tune(val, R.fixed_cascade), tune(val, R.resource_cascade)
    pd.options.display.width = 250
    pd.options.display.float_format = "{:.3f}".format

    # ---- 1) headline table: mixed device states
    test = sim.simulate(log[log.split == "test"], K, rng, "mixed")
    rows, ctrl = run_systems(test, clf, t_fixed, t_res)
    res = pd.DataFrame(rows).set_index("system")
    n = len(test)
    acts = [np.zeros(n, int), np.full(n, 2), R.fixed_cascade(test, t_fixed), R.resource_cascade(test, t_res), ctrl, ctrl]
    ress = [np.zeros(n, bool)] * 5 + [queue_sim(test, ctrl, "priority")]
    ci = bootstrap(test, acts, ress)
    res["macro_F1_CI95"] = [str(ci[s]["macro_F1_ci"]) for s in res.index]
    res["dangerous_FN_CI95"] = [str(ci[s]["dangerous_FN_ci"]) for s in res.index]
    res.to_csv(OUTPUTS / "eval_summary.csv")
    print(f"TEST: {test.image_id.nunique()} crops x {K} device states = {n} decisions "
          f"(fixed t={t_fixed}, resource t={t_res})\n")
    print(res.drop(columns=["eventual_ms", "auto_accuracy"]).T)

    # ---- 2) per-scenario (incl. degraded network)
    parts = []
    for name in sim.PRESETS:
        rep = sim.simulate(log[log.split == "test"], K, np.random.default_rng(SEED + 1), name)
        r, _ = run_systems(rep, clf, t_fixed, t_res)
        parts.append(pd.DataFrame(r).assign(scenario=name))
    by_sc = pd.concat(parts)
    by_sc.to_csv(OUTPUTS / "eval_by_scenario.csv", index=False)
    print("\nDangerous-FN % by scenario:\n", by_sc.pivot(index="system", columns="scenario", values="dangerous_FN_%")[list(sim.PRESETS)])
    print("\nMacro-F1 by scenario:\n", by_sc.pivot(index="system", columns="scenario", values="macro_F1")[list(sim.PRESETS)])
    print("\nResource-violation % by scenario:\n", by_sc.pivot(index="system", columns="scenario", values="violations_%")[list(sim.PRESETS)])

    # ---- 3) deferred queue: priority vs FIFO vs random across spare-capacity levels
    qrows = []
    sev_defer = test.y.to_numpy() == SEVERE
    for rate in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        for pol in ["priority", "fifo", "random"]:
            rs, wt = queue_sim(test, ctrl, pol, rate, return_wait=True)
            sc = score(test, ctrl, rs, pol)
            d = ctrl == 3
            qrows.append({"rate": rate, "policy": pol, "dangerous_FN_%": sc["dangerous_FN_%"], "macro_F1": sc["macro_F1"],
                          "severe_median_wait": float(np.nanmedian(wt[d & sev_defer])) if (rs[d & sev_defer]).any() else np.nan,
                          "other_median_wait": float(np.nanmedian(wt[d & ~sev_defer])) if (rs[d & ~sev_defer]).any() else np.nan,
                          "severe_resolved_%": 100 * rs[d & sev_defer].sum() / max((d & sev_defer).sum(), 1),
                          "other_resolved_%": 100 * rs[d & ~sev_defer].sum() / max((d & ~sev_defer).sum(), 1)})
    q = pd.DataFrame(qrows)
    q.to_csv(OUTPUTS / "eval_queue.csv", index=False)
    print("\nQueue policy comparison (dangerous-FN %, lower is better):\n", q.pivot(index="rate", columns="policy", values="dangerous_FN_%"))
    print("\nShare of truly-severe deferred cases resolved (%):\n", q.pivot(index="rate", columns="policy", values="severe_resolved_%"))

    # ---- 4) ablation: what each controller ingredient is worth (6 vs 5 vs 4 vs 3 headline deltas)
    s = res
    meta = {"k_scenarios": K, "n_decisions": int(n), "thresholds": {"fixed": t_fixed, "resource": t_res},
            "queue": {"rate": QUEUE_RATE, "max_len": QUEUE_MAX},
            "headline": {
                "dangerous_FN_small": s.loc[SYSTEMS[0], "dangerous_FN_%"], "dangerous_FN_fixed": s.loc[SYSTEMS[2], "dangerous_FN_%"],
                "dangerous_FN_controller": s.loc[SYSTEMS[4], "dangerous_FN_%"], "dangerous_FN_controller_queue": s.loc[SYSTEMS[5], "dangerous_FN_%"],
                "latency_controller_queue_ms": s.loc[SYSTEMS[5], "lat_mean_ms"], "latency_advanced_ms": s.loc[SYSTEMS[1], "lat_mean_ms"]}}
    json.dump(meta, open(OUTPUTS / "eval_meta.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
