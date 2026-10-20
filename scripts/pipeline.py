"""Person C: integrates A (model log) + B (features, router) + the resource simulator into one pipeline.

`ResQRoutePipeline.route(image_id, device)` replays one crop through the cascade the way a deployed system
would: MobileNet always runs; the controller then decides accept / escalate / defer / human review from the
small model's output, image quality and the device state. Per-model outputs come from outputs/model_log.csv
(so the demo needs no torch), and timings are the measured CPU times.
"""
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import CLASSES_3, OUTPUTS, SEED  # noqa: E402
import resqroute_sim as sim  # noqa: E402


def _load_router():
    spec = importlib.util.spec_from_file_location("router08", HERE / "08_router.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_router()          # Person B's feature lists, oracle labels, priority formula, baselines
ACTIONS = R.ACTIONS
FEATS = ["p_no", "p_minor", "p_severe", "conf", "margin", "entropy"] + R.QUAL + R.SCEN
CONTROLLER_PATH = OUTPUTS / "controller.joblib"


@dataclass
class Device:
    budget_ms: float = 180
    mem_free_mb: float = 250
    network_ok: int = 1
    battery: float = 0.8
    queue_len: int = 5


def train_controller(log, k=R.K_SCEN, seed=SEED):
    """Same recipe as 08_router.py: fit on the VAL split under mixed device states."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    val = sim.simulate(log[log.split == "val"], k, rng, "mixed")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(val[FEATS], R.oracle(val), sample_weight=R.weights(val))
    return clf


def get_controller(log):
    import joblib
    if CONTROLLER_PATH.exists():
        try:
            return joblib.load(CONTROLLER_PATH)
        except Exception:
            pass
    clf = train_controller(log)
    joblib.dump(clf, CONTROLLER_PATH, compress=3)
    return clf


def guard(actions, rep):
    """Hard feasibility rules applied on top of the learned policy (same as 08_router.py)."""
    a = np.array(actions, copy=True)
    a[(a == 1) & (rep.fits_efficientnet.to_numpy() == 0)] = 3
    a[(a == 2) & (rep.fits_resnet18.to_numpy() == 0)] = 3
    a[(a == 4) & (rep.network_ok.to_numpy() == 0)] = 3
    return a


def reasons(row, dev, action):
    """Plain-language reasons for the decision, from the same signals the controller sees."""
    out = []
    if row.conf >= 0.8:
        out.append(f"MobileNet is confident ({row.conf:.0%}, margin {row.margin:.2f})")
    elif row.conf >= 0.6:
        out.append(f"MobileNet is moderately confident ({row.conf:.0%}, margin {row.margin:.2f}, entropy {row.entropy:.2f})")
    else:
        out.append(f"MobileNet is uncertain (top class {row.conf:.0%}, margin {row.margin:.2f}, entropy {row.entropy:.2f})")
    if 0.2 <= row.p_severe and row.mobilenet_pred != "severe-damage":
        out.append(f"non-trivial severe-damage probability ({row.p_severe:.0%}) although it predicts "
                   f"{row.mobilenet_pred}: a miss would be dangerous")
    elif row.mobilenet_pred == "severe-damage":
        out.append(f"predicts severe damage ({row.p_severe:.0%}); the controller checks whether to confirm it")
    if row.blur < 20 or row.contrast < 20:
        out.append("low image quality (blurry or flat), so the small model's output is less trustworthy")
    if row.visible_frac < 0.6:
        out.append(f"only {row.visible_frac:.0%} of the building is visible")
    if row.prepost_diff > 40:
        out.append("large pre/post difference supports real change at this site")
    for m, name in (("efficientnet", "EfficientNet-B0"), ("resnet18", "ResNet-18")):
        if not row[f"fits_{m}"]:
            why = []
            if row[f"{sim.SMALL}_time_ms"] + row[f"{m}_time_ms"] > sim.effective_budget(dev.budget_ms, dev.battery):
                why.append("latency budget")
            if row[f"{m}_mem_mb"] > dev.mem_free_mb:
                why.append("free memory")
            out.append(f"{name} does not fit ({' and '.join(why)})")
    if dev.battery < sim.LOW_BATTERY:
        out.append(f"battery {dev.battery:.0%}: power-saving halves the latency budget")
    if not dev.network_ok:
        out.append("network is down: human review is unavailable")
    out.append(f"-> {ACTIONS[action].replace('_', ' ')}")
    return out


class ResQRoutePipeline:
    def __init__(self, log=None, controller=None):
        self.log = R.load() if log is None else log
        self.clf = controller or get_controller(self.log)
        self.by_id = self.log.set_index("image_id", drop=False)

    def row_with_device(self, image_id, dev: Device):
        row = self.by_id.loc[image_id].copy()
        for k, v in asdict(dev).items():
            row[k] = v
        for m in sim.BIG:
            row[f"fits_{m}"] = int(sim.fits(row[f"{sim.SMALL}_time_ms"], row[f"{m}_time_ms"], row[f"{m}_mem_mb"],
                                            dev.budget_ms, dev.battery, dev.mem_free_mb))
        return row

    def route(self, image_id, dev: Device):
        row = self.row_with_device(image_id, dev)
        x = pd.DataFrame([row[FEATS].astype(float)])
        proba = self.clf.predict_proba(x)[0]
        raw = int(self.clf.classes_[proba.argmax()])
        a = int(guard([raw], pd.DataFrame([row]).astype({"fits_efficientnet": int, "fits_resnet18": int,
                                                         "network_ok": int}))[0])
        probs = dict(zip([ACTIONS[int(c)] for c in self.clf.classes_], proba))
        steps = [{"stage": 1, "model": "MobileNetV3-Small", "pred": row.mobilenet_pred, "conf": float(row.conf),
                  "probs": row.mobilenet_p, "time_ms": float(row.mobilenet_time_ms), "mem_mb": float(row.mobilenet_mem_mb)}]
        final, status, total = row.mobilenet_pred, "accepted at stage 1", float(row.mobilenet_time_ms)
        for act, m, label in ((1, "efficientnet", "EfficientNet-B0"), (2, "resnet18", "ResNet-18")):
            if a == act:
                steps.append({"stage": 2 if act == 1 else 3, "model": label, "pred": row[f"{m}_pred"],
                              "conf": float(np.max(row[f"{m}_p"])), "probs": row[f"{m}_p"],
                              "time_ms": float(row[f"{m}_time_ms"]), "mem_mb": float(row[f"{m}_mem_mb"])})
                final, total = row[f"{m}_pred"], total + float(row[f"{m}_time_ms"])
                status = f"escalated to {label}"
        if a == 3:
            status = "deferred: kept in the priority queue until resources free up"
        if a == 4:
            final, status = "(human decides)", "flagged for human review"
        return {"row": row, "action": a, "action_name": ACTIONS[a], "controller_probs": probs, "steps": steps,
                "final_pred": final, "status": status, "immediate_ms": total,
                "priority": float(R.priority(pd.DataFrame([row]))[0]) if a == 3 else None,
                "reasons": reasons(row, dev, a)}


if __name__ == "__main__":
    p = ResQRoutePipeline()
    iid = p.log[p.log.split == "test"].image_id.iloc[0]
    out = p.route(iid, Device())
    print(iid, out["status"], out["final_pred"], "truth:", out["row"].damage_class_3)
    print("\n".join(out["reasons"]))
