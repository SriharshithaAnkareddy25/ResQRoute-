"""Step 5: run all three trained models on every crop (train+val+test) on CPU and write outputs/model_log.csv.

Run this on the laptop CPU, not on Kaggle: the timings are meant to represent a resource-constrained device.

Timing  : torch.set_num_threads(1); 10 warm-up passes per model first; then for each crop the forward pass
          alone (batch of 1, preprocessing excluded) is timed 3x with time.perf_counter() -> median ms.
          The three models are interleaved per crop so they see the same machine conditions.
Memory  : per-model working memory = peak process RSS increase from "torch imported, nothing loaded"
          to "weights loaded + one forward pass", measured in a fresh subprocess (median of 3 runs), so
          allocations from other models cannot hide it. All inputs are 224x224, so this is a property of the
          model, not of the image; the same value is written to every row. (A per-image RSS delta was
          tried first - after warm-up it is ~0 MB for every image because the allocator reuses memory.)
          Written separately to outputs/model_memory.json.

Resumable: progress is appended to outputs/model_log.partial.csv; rerunning skips finished image_ids.

  python scripts/05_run_inference_log.py
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import pandas as pd
import psutil
import torch
from PIL import Image
from tqdm import tqdm

from common import CLASSES_3, MODELS, OUTPUTS, REPO, build_model, eval_transform

N_TIMED = 3
N_WARMUP = 10
COLS = (["image_id", "disaster_event", "split", "damage_class_3", "damage_class_4", "pre_crop_path", "post_crop_path"]
        + [f"{p}_{s}" for _, p in MODELS.values() for s in ("probs", "pred", "correct", "time_ms", "mem_mb")])


def load(arch):
    m = build_model(arch, len(CLASSES_3), pretrained=False)
    m.load_state_dict(torch.load(REPO / MODELS[arch][0], map_location="cpu", weights_only=True))
    return m.eval()


def measure_memory_once(arch):
    """Runs inside a fresh subprocess. Peak RSS increase (MB) for loading `arch` + one forward pass."""
    torch.set_num_threads(1)
    proc = psutil.Process()
    x = torch.zeros(1, 3, 224, 224)
    base = proc.memory_info().rss
    peak, stop = [base], threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    model = load(arch)
    with torch.inference_mode():
        model(x)
    stop.set()
    t.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return (peak[0] - base) / 2**20


def model_memory():
    mem = {}
    for arch in MODELS:
        runs = []
        for _ in range(3):
            out = subprocess.run([sys.executable, __file__, "--measure-memory", arch],
                                 capture_output=True, text=True, check=True, cwd=REPO)
            runs.append(float(out.stdout.strip().splitlines()[-1]))
        mem[arch] = {"mem_mb": round(statistics.median(runs), 2), "runs_mb": [round(r, 2) for r in runs]}
        print(f"memory {arch}: {mem[arch]}")
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-memory", help=argparse.SUPPRESS)  # internal: subprocess mode
    ap.add_argument("--limit", type=int, default=0, help="only first N crops (smoke test)")
    args = ap.parse_args()
    if args.measure_memory:
        print(measure_memory_once(args.measure_memory))
        return

    torch.set_num_threads(1)
    man = pd.read_csv(OUTPUTS / "crop_manifest.csv").merge(pd.read_csv(OUTPUTS / "split_manifest.csv"), on="image_id")
    if args.limit:
        man = man.head(args.limit)

    mem = model_memory()
    (OUTPUTS / "model_memory.json").write_text(json.dumps(
        {"method": "peak RSS increase for weight load + one forward pass, fresh subprocess, 1 thread, median of 3",
         **mem}, indent=2))

    models = {arch: load(arch) for arch in MODELS}
    tf = eval_transform()
    with torch.inference_mode():
        x = torch.zeros(1, 3, 224, 224)
        for m in models.values():
            for _ in range(N_WARMUP):
                m(x)

    partial = OUTPUTS / "model_log.partial.csv"
    done = set(pd.read_csv(partial).image_id) if partial.exists() else set()
    todo = man[~man.image_id.isin(done)]
    print(f"{len(man)} crops, {len(done)} already logged, {len(todo)} to go")

    buf = []
    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc="inference")):
        x = tf(Image.open(REPO / r.post_crop_path).convert("RGB")).unsqueeze(0)
        row = {c: getattr(r, c) for c in COLS[:7]}
        for arch, m in models.items():
            prefix = MODELS[arch][1]
            times = []
            with torch.inference_mode():
                for _ in range(N_TIMED):
                    t0 = time.perf_counter()
                    logits = m(x)
                    times.append((time.perf_counter() - t0) * 1000)
            probs = torch.softmax(logits[0].float(), 0).tolist()
            pred = CLASSES_3[max(range(len(probs)), key=probs.__getitem__)]
            row[f"{prefix}_probs"] = json.dumps([round(p, 6) for p in probs])
            row[f"{prefix}_pred"] = pred
            row[f"{prefix}_correct"] = int(pred == r.damage_class_3)
            row[f"{prefix}_time_ms"] = round(statistics.median(times), 3)
            row[f"{prefix}_mem_mb"] = mem[arch]["mem_mb"]
        buf.append(row)
        if len(buf) >= 100 or i == len(todo) - 1:
            pd.DataFrame(buf, columns=COLS).to_csv(partial, mode="a", header=not partial.exists(), index=False)
            buf = []

    log = pd.read_csv(partial).drop_duplicates("image_id", keep="last")
    log = log.set_index("image_id").loc[man.image_id].reset_index()[COLS]  # manifest order, exact columns
    log.to_csv(OUTPUTS / "model_log.csv", index=False)
    print(f"wrote outputs/model_log.csv ({len(log)} rows)")
    for arch, (_, p) in MODELS.items():
        print(f"  {p:13s} acc {log[f'{p}_correct'].mean():.3f}  median {log[f'{p}_time_ms'].median():.1f} ms  "
              f"mem {mem[arch]['mem_mb']} MB")


if __name__ == "__main__":
    main()
