"""Step 1: index every building polygon in the xBD post-disaster labels (labels only, no images).

Schema verified against real files from the Kaggle mirror (train set, 2026-10-12):
  json["features"]["xy"][i]["properties"] -> {"feature_type": "building", "subtype": <damage>, "uid": ...}
  json["features"]["xy"][i]["wkt"]        -> "POLYGON ((x y, ...))" in pixel coords of the 1024x1024 PNG
The pre-disaster JSON has the same uids but no `subtype`, so only post JSONs are read.

Writes outputs/building_index.csv (one row per building) and prints per-event class counts,
which is what the choice of disasters in step 2 is based on.

  python scripts/01_index_buildings.py [--raw C:/data/xbd]
"""
import argparse
import json

import pandas as pd
from shapely import wkt
from tqdm import tqdm

from common import CLASSES_4, OUTPUTS, TO_3, TO_4, find_raw_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="folder containing images/ and labels/ (auto-detected if omitted)")
    args = ap.parse_args()
    raw = find_raw_dir(args.raw)
    print("raw data:", raw)

    rows = []
    for lab in tqdm(sorted((raw / "labels").glob("*_post_disaster.json")), desc="labels"):
        tile = lab.name.replace("_post_disaster.json", "")  # e.g. hurricane-michael_00000123
        meta = json.loads(lab.read_text())
        event = meta["metadata"].get("disaster") or tile.rsplit("_", 1)[0]
        for feat in meta["features"]["xy"]:
            props = feat["properties"]
            if props.get("feature_type") != "building":
                continue
            poly = wkt.loads(feat["wkt"])
            x0, y0, x1, y1 = poly.bounds
            rows.append({
                "tile": tile,
                "uid": props["uid"],
                "disaster_event": event,
                "disaster_type": meta["metadata"].get("disaster_type", ""),
                "subtype_raw": props.get("subtype", "<missing>"),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "poly_area_px": poly.area,
            })

    if not rows:
        raise SystemExit(f"no building polygons found under {raw / 'labels'}")
    df = pd.DataFrame(rows)
    df["damage_class_4"] = df["subtype_raw"].map(TO_4)
    df["damage_class_3"] = df["subtype_raw"].map(TO_3)
    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(OUTPUTS / "building_index.csv", index=False)
    print(f"{len(df)} buildings from {df.tile.nunique()} tiles -> outputs/building_index.csv")

    unexpected = set(df.subtype_raw) - set(CLASSES_4) - {"un-classified"}
    if unexpected:
        print("WARNING unexpected subtype values:", unexpected)

    counts = pd.crosstab(df.disaster_event, df.subtype_raw)
    counts = counts.reindex(columns=CLASSES_4 + ["un-classified"], fill_value=0)
    counts["tiles"] = df.groupby("disaster_event").tile.nunique()
    known = counts[CLASSES_4].sum(axis=1)
    counts["minor%"] = (100 * counts["minor-damage"] / known).round(1)
    counts["severe%"] = (100 * (counts["major-damage"] + counts["destroyed"]) / known).round(1)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(counts.sort_values("severe%", ascending=False))


if __name__ == "__main__":
    main()
