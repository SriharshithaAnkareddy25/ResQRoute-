"""Step 2: pick events, draw a class-balanced subset, and fix the train/val/test split.

- Keeps only the chosen events, drops `un-classified` and tiny polygons (too few pixels to judge).
- Undersamples to at most --per-class buildings per damage_class_3 class. Inside `severe-damage`
  it draws major-damage and destroyed equally where possible, so the 4-class column stays usable.
- Splits 70/15/15, stratified by damage_class_3 AND grouped by source tile: all buildings from one
  1024x1024 image land in the same split, so neighbouring buildings cannot leak between train and test.

Writes outputs/crop_manifest.csv, outputs/split_manifest.csv, outputs/class_map.json.
Nobody re-splits after this - everyone uses split_manifest.csv.

  python scripts/02_build_subset.py --events hurricane-michael,santa-rosa-wildfire,... --per-class 2000
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES_3, CLASSES_4, OUTPUTS, SEED


def sample_class(df: pd.DataFrame, cls: str, n: int, rng) -> pd.DataFrame:
    pool = df[df.damage_class_3 == cls]
    if len(pool) <= n:
        return pool
    if cls != "severe-damage":
        return pool.sample(n, random_state=rng)
    # balance major-damage vs destroyed inside the merged severe class
    major, destroyed = pool[pool.damage_class_4 == "major-damage"], pool[pool.damage_class_4 == "destroyed"]
    n_major = min(len(major), max(n // 2, n - len(destroyed)))
    return pd.concat([major.sample(n_major, random_state=rng),
                      destroyed.sample(n - n_major, random_state=rng)])


def grouped_split(df: pd.DataFrame, frac: float, seed: int):
    """Return boolean mask selecting ~frac of rows, stratified by class and grouped by tile."""
    k = round(1 / frac)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(df, df.damage_class_3, groups=df.tile))
    mask = np.zeros(len(df), bool)
    mask[held] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="comma-separated disaster events")
    ap.add_argument("--per-class", type=int, default=2000, help="max buildings per 3-class label")
    ap.add_argument("--min-area", type=float, default=150.0, help="min polygon area in px (~12x12)")
    args = ap.parse_args()

    idx = pd.read_csv(OUTPUTS / "building_index.csv")
    events = [e.strip() for e in args.events.split(",")]
    missing = set(events) - set(idx.disaster_event)
    if missing:
        raise SystemExit(f"unknown events {missing}")

    df = idx[idx.disaster_event.isin(events) & idx.damage_class_3.notna() & (idx.poly_area_px >= args.min_area)]
    print(f"{len(df)} usable buildings in {events}")
    rng = np.random.RandomState(SEED)
    sub = pd.concat([sample_class(df, c, args.per_class, rng) for c in CLASSES_3]).reset_index(drop=True)

    sub["image_id"] = sub.tile + "_" + sub.uid.str[:8]
    assert sub.image_id.is_unique, "image_id collision - lengthen the uid prefix"
    sub["pre_crop_path"] = "data/crops/" + sub.image_id + "_pre.png"
    sub["post_crop_path"] = "data/crops/" + sub.image_id + "_post.png"

    # test first (~15%), then val (~15% of the total) from what remains
    sub["split"] = "train"
    test = grouped_split(sub, 0.15, SEED)
    sub.loc[test, "split"] = "test"
    rest = sub[~test]
    val = grouped_split(rest, 0.15 / 0.85, SEED)
    sub.loc[rest.index[val], "split"] = "val"
    assert sub.groupby("tile").split.nunique().max() == 1, "a tile spans two splits"

    sub = sub.sort_values("image_id").reset_index(drop=True)
    cols = ["image_id", "disaster_event", "disaster_type", "tile", "uid", "subtype_raw",
            "damage_class_3", "damage_class_4", "x0", "y0", "x1", "y1", "poly_area_px",
            "pre_crop_path", "post_crop_path"]
    sub[cols].to_csv(OUTPUTS / "crop_manifest.csv", index=False)
    sub[["image_id", "split"]].to_csv(OUTPUTS / "split_manifest.csv", index=False)
    # flat {"0": "no-damage", ...} exactly as the team brief specifies; 4-class order kept separately
    (OUTPUTS / "class_map.json").write_text(json.dumps({str(i): c for i, c in enumerate(CLASSES_3)}, indent=2))
    (OUTPUTS / "class_map_4.json").write_text(json.dumps({str(i): c for i, c in enumerate(CLASSES_4)}, indent=2))

    print(f"\n{len(sub)} crops, {sub.tile.nunique()} tiles")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.crosstab(sub.damage_class_3, sub.split, margins=True), "\n")
        print(pd.crosstab([sub.disaster_event], sub.damage_class_4, margins=True), "\n")
        print((pd.crosstab(sub.split, sub.damage_class_3, normalize="index") * 100).round(1))


if __name__ == "__main__":
    main()
