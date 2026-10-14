"""Step 7 (router input): cheap image-quality / difficulty features for every crop -> outputs/quality_features.csv

All are computable on-device in a few ms without any neural network:
  blur          variance of the Laplacian of the post crop (low = blurry)
  brightness    mean grey level of the visible part of the post crop
  contrast      std of grey level of the visible part
  visible_frac  fraction of the 224x224 crop that is image rather than letterbox padding (building size proxy)
  prepost_diff  mean absolute difference between pre and post crops (change magnitude)
  dark_frac / bright_frac  share of near-black / near-white pixels (shadow, cloud or smoke proxies)

  python scripts/07_quality_features.py
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import OUTPUTS, REPO


def features(pre_path, post_path):
    post = cv2.imread(str(REPO / post_path))
    pre = cv2.imread(str(REPO / pre_path))
    g = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = post.max(axis=2) > 0  # letterbox padding is exactly black
    vis = g[mask] if mask.any() else g.ravel()
    gp = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "blur": float(cv2.Laplacian(g, cv2.CV_32F)[mask].var()) if mask.any() else 0.0,
        "brightness": float(vis.mean()),
        "contrast": float(vis.std()),
        "visible_frac": float(mask.mean()),
        "prepost_diff": float(np.abs(g - gp)[mask].mean()) if mask.any() else 0.0,
        "dark_frac": float((vis < 40).mean()),
        "bright_frac": float((vis > 215).mean()),
    }


def main():
    man = pd.read_csv(OUTPUTS / "crop_manifest.csv")
    rows = [{"image_id": r.image_id, **features(r.pre_crop_path, r.post_crop_path)}
            for r in tqdm(man.itertuples(), total=len(man), desc="quality")]
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "quality_features.csv", index=False)
    print(df.describe().round(3).T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
