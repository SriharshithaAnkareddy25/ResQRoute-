"""Step 6: acceptance checks on outputs/model_log.csv (brief section 9) + per-model test-set metrics.

1. exact columns, no missing values in probs/time/mem
2. every image_id in split_manifest.csv appears exactly once
3. *_pred == argmax(*_probs) and *_correct == (*_pred == damage_class_3) on every row
4. reloads the three .pth files and re-predicts random rows -> must match the logged probabilities

  python scripts/06_verify_log.py [--recheck 20]
"""
import argparse
import importlib
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from common import CLASSES_3, MODELS, OUTPUTS, REPO, SEED, eval_transform

inf = importlib.import_module("05_run_inference_log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", type=int, default=20, help="rows to re-predict from the saved weights")
    args = ap.parse_args()

    log = pd.read_csv(OUTPUTS / "model_log.csv")
    split = pd.read_csv(OUTPUTS / "split_manifest.csv")
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + msg)

    check(list(log.columns) == inf.COLS, "columns match the brief exactly")
    value_cols = [c for c in log.columns if c.endswith(("_probs", "_time_ms", "_mem_mb", "_pred", "_correct"))]
    check(log[value_cols].notna().all().all(), "no missing values in prediction/time/memory columns")
    check(log.image_id.is_unique and set(log.image_id) == set(split.image_id),
          f"one row per image_id in split_manifest ({len(log)} rows / {len(split)} ids)")
    check((log.set_index("image_id").split == split.set_index("image_id").split).all(), "split column agrees")

    probs = {}
    for _, p in MODELS.values():
        probs[p] = np.array([json.loads(s) for s in log[f"{p}_probs"]])
        argmax = np.array(CLASSES_3)[probs[p].argmax(1)]
        check(probs[p].shape[1] == 3 and np.allclose(probs[p].sum(1), 1, atol=1e-4), f"{p}: 3 probs summing to 1")
        check((argmax == log[f"{p}_pred"]).all(), f"{p}: pred == argmax(probs)")
        check(((log[f"{p}_pred"] == log.damage_class_3).astype(int) == log[f"{p}_correct"]).all(),
              f"{p}: correct == (pred == damage_class_3)")

    torch.set_num_threads(1)
    tf = eval_transform()
    sample = log.sample(min(args.recheck, len(log)), random_state=SEED)
    for arch, (_, p) in MODELS.items():
        m = inf.load(arch)
        with torch.inference_mode():
            x = torch.stack([tf(Image.open(REPO / f).convert("RGB")) for f in sample.post_crop_path])
            fresh = torch.softmax(m(x).float(), 1).numpy()
        diff = np.abs(fresh - probs[p][sample.index]).max()
        check(diff < 1e-3, f"{p}: reloaded weights reproduce logged probs on {len(sample)} rows (max diff {diff:.1e})")

    print("\nTest-split metrics (3-class):")
    test = log[log.split == "test"]
    for _, p in MODELS.values():
        y, yh = test.damage_class_3, test[f"{p}_pred"]
        sev = recall_score(y, yh, labels=["severe-damage"], average=None)[0]
        # dangerous false negative: truly severe building predicted as no-damage
        dfn = ((y == "severe-damage") & (yh == "no-damage")).sum() / max((y == "severe-damage").sum(), 1)
        print(f"  {p:13s} acc {np.mean(y == yh):.3f}  macroF1 {f1_score(y, yh, average='macro'):.3f}  "
              f"severe recall {sev:.3f}  severe->no-damage {dfn:.3f}  "
              f"median {test[f'{p}_time_ms'].median():.1f} ms  mem {test[f'{p}_mem_mb'].iloc[0]} MB")
        print("    confusion (rows=true, cols=pred, order " + "/".join(CLASSES_3) + "):",
              confusion_matrix(y, yh, labels=CLASSES_3).tolist())

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
