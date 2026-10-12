"""Step 3: cut 224x224 pre/post crops for every building in outputs/crop_manifest.csv.

For each building: polygon bounding box -> grow by --pad (20%) on every side, and to at least
--min-side px so tiny buildings keep some context -> clip to the 1024x1024 tile -> crop the same
window from the pre and post PNG -> letterbox to 224x224 (aspect ratio kept, black padding).

Needs the raw images, so this is normally run inside a Kaggle notebook (see README).

  python scripts/03_extract_crops.py [--raw /kaggle/input/...] [--workers 4]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import OUTPUTS, REPO, find_raw_dir

SIZE = 224


def crop_box(r, w, h, pad, min_side):
    bw, bh = r.x1 - r.x0, r.y1 - r.y0
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    side_w, side_h = max(bw * (1 + 2 * pad), min_side), max(bh * (1 + 2 * pad), min_side)
    x0, x1 = int(max(0, np.floor(cx - side_w / 2))), int(min(w, np.ceil(cx + side_w / 2)))
    y0, y1 = int(max(0, np.floor(cy - side_h / 2))), int(min(h, np.ceil(cy + side_h / 2)))
    return x0, y0, x1, y1


def letterbox(img):
    h, w = img.shape[:2]
    s = SIZE / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((SIZE, SIZE, 3), img.dtype)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def process_tile(item, raw, pad, min_side):
    tile, rows = item
    imgs = {}
    for phase in ("pre", "post"):
        img = cv2.imread(str(raw / "images" / f"{tile}_{phase}_disaster.png"), cv2.IMREAD_COLOR)
        if img is None:
            return [f"missing image {tile}_{phase}"]
        imgs[phase] = img
    h, w = imgs["post"].shape[:2]
    errors = []
    for r in rows.itertuples():
        x0, y0, x1, y1 = crop_box(r, w, h, pad, min_side)
        if x1 - x0 < 2 or y1 - y0 < 2:
            errors.append(f"degenerate box {r.image_id}")
            continue
        for phase, rel in (("pre", r.pre_crop_path), ("post", r.post_crop_path)):
            cv2.imwrite(str(REPO / rel), letterbox(imgs[phase][y0:y1, x0:x1]))
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="folder containing images/ and labels/ (auto-detected if omitted)")
    ap.add_argument("--pad", type=float, default=0.20, help="context margin per side, fraction of box")
    ap.add_argument("--min-side", type=int, default=32, help="minimum crop window in source px")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    raw = find_raw_dir(args.raw)

    man = pd.read_csv(OUTPUTS / "crop_manifest.csv")
    (REPO / "data" / "crops").mkdir(parents=True, exist_ok=True)
    groups = list(man.groupby("tile"))
    fn = partial(process_tile, raw=raw, pad=args.pad, min_side=args.min_side)
    errors = []
    with ProcessPoolExecutor(args.workers) as ex:
        for errs in tqdm(ex.map(fn, groups, chunksize=4), total=len(groups), desc="tiles"):
            errors += errs

    missing = [p for p in pd.concat([man.pre_crop_path, man.post_crop_path]) if not (REPO / p).exists()]
    print(f"{len(man)} buildings, {len(groups)} tiles, {len(errors)} errors, {len(missing)} missing crop files")
    for e in errors[:10]:
        print("  ", e)
    if missing:
        raise SystemExit("some crops were not written - see errors above")


if __name__ == "__main__":
    main()
