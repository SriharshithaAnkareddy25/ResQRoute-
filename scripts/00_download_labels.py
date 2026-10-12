"""Step 0: download the xBD post-disaster label JSONs (labels only, ~160 MB) to the laptop.

Source: Kaggle mirror tunguz/xview2-challenge-dataset-train-and-test (byte-for-byte xView2 challenge
release, CC BY-NC-SA 4.0). kagglehub fetches public files one by one without a Kaggle login.
Images are NOT downloaded here - crops are cut on Kaggle (step 3), where the dataset is already mounted.
Re-running skips files already on disk.

  python scripts/00_download_labels.py --out C:/data/xbd
"""
import argparse
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

DATASET = "tunguz/xview2-challenge-dataset-train-and-test"
LIST_URL = f"https://www.kaggle.com/api/v1/datasets/list/{DATASET}?pageSize=200"


def list_label_files():
    names, token = [], None
    while True:
        url = LIST_URL + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        page = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "resqroute"})))
        names += [f["name"] for f in page["datasetFiles"]]
        token = page.get("nextPageTokenNullable")
        if not token:
            break
    return [n for n in names if n.startswith("train/train/labels/") and n.endswith("_post_disaster.json")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("C:/data/xbd"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("KAGGLEHUB_CACHE", str(args.out / "_khcache"))
    import kagglehub  # after setting the cache location

    labels = args.out / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    files = list_label_files()
    todo = [n for n in files if not (labels / Path(n).name).exists()]
    print(f"{len(files)} label files, {len(todo)} to download")

    def get(name):
        for attempt in range(3):
            try:
                shutil.move(kagglehub.dataset_download(DATASET, path=name), labels / Path(name).name)
                return None
            except Exception as e:
                err = e
                time.sleep(2)
        return f"{name}: {err!r}"

    with ThreadPoolExecutor(args.workers) as ex:
        fails = [f for f in tqdm(ex.map(get, todo), total=len(todo)) if f]
    shutil.rmtree(args.out / "_khcache", ignore_errors=True)
    print(f"done, {len(fails)} failures (rerun to retry)", *fails[:5], sep="\n")


if __name__ == "__main__":
    main()
