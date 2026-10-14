"""Step 8: train the risk- and resource-aware routing controller and evaluate it against baselines.

Decision setting (what the controller can see when it decides):
  - the small model's (MobileNetV3-Small) softmax output, which always runs first: probabilities,
    confidence, entropy, top-2 margin
  - cheap image-quality features (07_quality_features.py)
  - the device state at that moment, which is simulated: latency budget, free memory, network, battery,
    queue length, plus whether each bigger model fits the budget

Actions: 0 accept the small model | 1 escalate to EfficientNet-B0 | 2 escalate to ResNet-18 |
         3 defer to the priority queue | 4 flag for human review
Escalation targets are ordered by measured CPU cost, not by name: EfficientNet-B0 (~105 ms) is cheaper
than ResNet-18 (~220 ms) on this laptop.

Training labels ("oracle route"), from model_log.csv, for each crop x scenario:
  small correct                                        -> accept
  else cheapest bigger model that is correct AND fits  -> escalate to it
  else a correct model exists but none fits right now  -> defer
  else (no model correct)                              -> human review (defer if the network is down)
Asymmetric safety: truly-severe buildings get 3x sample weight, and 5x when the small model misses them.

The controller is trained only on the VAL split (the base models never trained on those crops, so their
outputs there are honest), and evaluated on the TEST split. The TRAIN split is not used: the base models
memorised it.

  python scripts/08_router.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from common import CLASSES_3, OUTPUTS, SEED

ACTIONS = ["accept_small", "escalate_effnet", "escalate_resnet18", "defer", "human_review"]
SMALL, BIG = "mobilenet", ["efficientnet", "resnet18"]  # BIG ordered by measured cost
SEVERE = CLASSES_3.index("severe-damage")
K_SCEN = 8  # simulated device states per crop

QUAL = ["blur", "brightness", "contrast", "visible_frac", "prepost_diff", "dark_frac", "bright_frac"]
SCEN = ["budget_ms", "mem_free_mb", "network_ok", "battery", "queue_len", "fits_efficientnet", "fits_resnet18"]


def load():
    log = pd.read_csv(OUTPUTS / "model_log.csv")
    q = pd.read_csv(OUTPUTS / "quality_features.csv")
    log = log.merge(q, on="image_id")
    log["y"] = log.damage_class_3.map(CLASSES_3.index)
    for m in [SMALL] + BIG:
        log[f"{m}_p"] = log[f"{m}_probs"].map(json.loads)
        log[f"{m}_yhat"] = log[f"{m}_pred"].map(CLASSES_3.index)
    p = np.stack(log[f"{SMALL}_p"].to_numpy())
    s = np.sort(p, 1)
    log["conf"] = s[:, -1]
    log["margin"] = s[:, -1] - s[:, -2]
    log["entropy"] = -(p * np.log(p + 1e-12)).sum(1)
    for i, c in enumerate(["p_no", "p_minor", "p_severe"]):
        log[c] = p[:, i]
    return log


def simulate(df, k, rng):
    """Repeat each crop k times under random device states."""
    rep = df.loc[df.index.repeat(k)].reset_index(drop=True)
    n = len(rep)
    rep["budget_ms"] = rng.choice([60, 180, 400], n)            # tight / normal / relaxed response target
    rep["mem_free_mb"] = rng.choice([150, 250, 500], n)
    rep["network_ok"] = (rng.random(n) < 0.7).astype(int)
    rep["battery"] = rng.uniform(0.05, 1.0, n).round(2)
    rep["queue_len"] = rng.integers(0, 50, n)
    eff_budget = np.where(rep.battery < 0.2, rep.budget_ms * 0.5, rep.budget_ms)  # power saving
    for m in BIG:
        fits = (rep[f"{SMALL}_time_ms"] + rep[f"{m}_time_ms"] <= eff_budget) & (rep[f"{m}_mem_mb"] <= rep.mem_free_mb)
        rep[f"fits_{m}"] = fits.astype(int)
    return rep


def oracle(rep):
    lab = np.full(len(rep), -1)
    small_ok = rep[f"{SMALL}_correct"] == 1
    lab[small_ok] = 0
    for a, m in zip([1, 2], BIG):
        take = (lab == -1) & (rep[f"{m}_correct"] == 1) & (rep[f"fits_{m}"] == 1)
        lab[take] = a
    any_ok = (rep[[f"{m}_correct" for m in BIG]].sum(axis=1) > 0)
    lab[(lab == -1) & any_ok] = 3
    lab[(lab == -1) & (rep.network_ok == 1)] = 4
    lab[lab == -1] = 3
    return lab


def weights(rep):
    w = np.ones(len(rep))
    sev = rep.y == SEVERE
    w[sev] = 3.0
    w[sev & (rep[f"{SMALL}_correct"] == 0)] = 5.0
    return w


def priority(rep):
    """Deferred-queue priority P = U x R x D x A from the project plan."""
    U = rep.entropy / np.log(3)                              # uncertainty, 0..1
    R = rep.p_severe                                         # risk of a dangerous miss
    D = 0.5 * rep.p_minor + 1.0 * rep.p_severe               # expected damage severity
    A = 0.5 + 0.5 * rep.visible_frac                         # affected-area importance (building size)
    return U * R * D * A


def outcome(rep, actions, defer_model="efficientnet"):
    """Final prediction, compute cost and resource violation for each (crop, scenario, action)."""
    t_small = rep[f"{SMALL}_time_ms"].to_numpy()
    pred = rep[f"{SMALL}_yhat"].to_numpy().copy()
    time = t_small.copy()
    mem = rep[f"{SMALL}_mem_mb"].to_numpy().copy()
    for a, m in zip([1, 2], BIG):
        sel = actions == a
        pred[sel] = rep[f"{m}_yhat"].to_numpy()[sel]
        time[sel] += rep[f"{m}_time_ms"].to_numpy()[sel]
        mem[sel] = np.maximum(mem[sel], rep[f"{m}_mem_mb"].to_numpy()[sel])
    d = actions == 3  # processed later, when resources return
    pred[d] = rep[f"{defer_model}_yhat"].to_numpy()[d]
    h = actions == 4  # a human gives the correct label
    pred[h] = rep.y.to_numpy()[h]
    eff_budget = np.where(rep.battery < 0.2, rep.budget_ms * 0.5, rep.budget_ms)
    violated = ((time > eff_budget) | (mem > rep.mem_free_mb)) & ~d & ~h
    return pred, time, mem, violated


def metrics(rep, actions, name):
    pred, time, mem, viol = outcome(rep, actions)
    y = rep.y.to_numpy()
    auto = ~np.isin(actions, [4])
    sev = y == SEVERE
    return {
        "policy": name,
        "accuracy": (pred == y).mean(),
        "macro_F1": f1_score(y, pred, average="macro"),
        "severe_recall": (pred[sev] == SEVERE).mean(),
        "dangerous_FN_%": 100 * (pred[sev] == 0).mean(),   # truly severe -> "no-damage"
        "auto_accuracy": (pred[auto] == y[auto]).mean(),
        "avg_time_ms": time.mean(),
        "avg_peak_mem_mb": mem.mean(),
        "violations_%": 100 * viol.mean(),
        **{f"%{a}": 100 * (actions == i).mean() for i, a in enumerate(ACTIONS)},
    }


def tune_threshold(val, fn):
    grid = np.round(np.arange(0.40, 0.99, 0.02), 2)
    best = max(grid, key=lambda t: f1_score(val.y, outcome(val, fn(val, t))[0], average="macro"))
    return float(best)


def fixed_cascade(rep, t):
    """Conventional cascade: accept if confidence >= t, else next model; ignores resources and risk."""
    a = np.where(rep.conf >= t, 0, 1)
    eff_conf = np.stack(rep["efficientnet_p"].to_numpy()).max(1)
    a[(a == 1) & (eff_conf < t)] = 2
    return a


def resource_cascade(rep, t):
    """Resource-aware cascade without safety risk: confident -> accept; else the cheapest model that
    fits the budget (EfficientNet, then ResNet-18); else defer."""
    a = np.full(len(rep), 3)
    a[rep.fits_resnet18 == 1] = 2
    a[rep.fits_efficientnet == 1] = 1
    a[rep.conf >= t] = 0
    return a


def main():
    rng = np.random.default_rng(SEED)
    log = load()
    val = simulate(log[log.split == "val"], K_SCEN, rng)
    test = simulate(log[log.split == "test"], K_SCEN, rng)
    feats = ["p_no", "p_minor", "p_severe", "conf", "margin", "entropy"] + QUAL + SCEN

    yv = oracle(val)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=SEED)
    clf.fit(val[feats], yv, sample_weight=weights(val))
    act = clf.predict(test[feats])
    # hard safety and feasibility guards: never pick a model that does not fit; no human review offline
    act[(act == 1) & (test.fits_efficientnet == 0)] = 3
    act[(act == 2) & (test.fits_resnet18 == 0)] = 3
    act[(act == 4) & (test.network_ok == 0)] = 3

    t_fixed = tune_threshold(val, fixed_cascade)
    t_res = tune_threshold(val, resource_cascade)
    n = len(test)
    rows = [
        metrics(test, np.zeros(n, int), "Always small (MobileNetV3-S)"),
        metrics(test, np.ones(n, int), "Always EfficientNet-B0"),
        metrics(test, np.full(n, 2), "Always ResNet-18"),
        metrics(test, fixed_cascade(test, t_fixed), f"Fixed-confidence cascade (t={t_fixed})"),
        metrics(test, resource_cascade(test, t_res), f"Resource-aware cascade, no risk (t={t_res})"),
        metrics(test, act, "ResQRoute controller (ours)"),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "router_results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(f"TEST: {test.image_id.nunique()} crops x {K_SCEN} device states = {n} decisions\n")
        print(res.set_index("policy"))

    # adaptivity: how the controller's behaviour changes with the latency budget
    test["action"] = [ACTIONS[a] for a in act]
    pred, time, _, _ = outcome(test, act)
    test["correct"] = pred == test.y
    test["time"] = time
    by_budget = pd.crosstab(test.budget_ms, test.action, normalize="index").mul(100).round(1)
    by_budget["accuracy"] = test.groupby("budget_ms").correct.mean().round(3)
    by_budget["avg_time_ms"] = test.groupby("budget_ms").time.mean().round(1)
    print("\nController actions (%) by latency budget:\n", by_budget)
    by_budget.to_csv(OUTPUTS / "router_by_budget.csv")

    # deferred escalation: does the priority queue put dangerous cases first?
    q = test[act == 3].copy()
    q["P"] = priority(q)
    q = q.sort_values("P", ascending=False)
    half = len(q) // 2
    sev_top, sev_bot = (q.y.iloc[:half] == SEVERE).mean(), (q.y.iloc[half:] == SEVERE).mean()
    print(f"\nDeferred queue: {len(q)} items; truly-severe share in top half {sev_top:.1%} vs bottom half {sev_bot:.1%}")

    # what the controller learned to look at
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, val[feats], yv, n_repeats=3, random_state=SEED, n_jobs=1)
    top = pd.Series(imp.importances_mean, feats).sort_values(ascending=False).head(10)
    print("\nTop controller features (permutation importance on val):\n", top.round(4))
    top.to_csv(OUTPUTS / "router_feature_importance.csv", header=["importance"])
    oracle_dist = pd.Series(yv).map(dict(enumerate(ACTIONS))).value_counts(normalize=True).mul(100).round(1)
    print("\nOracle route distribution on val (%):\n", oracle_dist)
    json.dump({"thresholds": {"fixed": t_fixed, "resource": t_res},
               "deferred_queue": {"n": int(len(q)), "severe_top_half": sev_top, "severe_bottom_half": sev_bot}},
              open(OUTPUTS / "router_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
