# ResQRoute – Part A: data pipeline and base damage classifiers

This part of ResQRoute covers the data and the three base models: it extracts xBD building crops, trains three
damage classifiers of increasing size, and logs every crop's predictions, CPU latency and memory in
`outputs/model_log.csv`. Person B trains the routing controller on that file, and Person C uses it for the evaluation.

## Data source and licence

- **Dataset:** xBD (Gupta et al., 2019, *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*,
  arXiv:1911.09296). Imagery is from the Maxar Open Data Program.
- **Copy used:** the Kaggle mirror [`tunguz/xview2-challenge-dataset-train-and-test`](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test).
  It is a byte-for-byte copy of the xView2 challenge release. Only the labelled `train` set is used: 10 disasters,
  2,799 pre/post pairs of 1024×1024 images.
- **Licence:** CC BY-NC-SA 4.0. Non-commercial use only, credit the authors, and share derived data (crops, logs)
  under the same licence. Raw data is not committed (see `.gitignore`).

## Pipeline

| Step | Script | Runs on | Output |
|---|---|---|---|
| 0 | `scripts/00_download_labels.py` | laptop | `C:/data/xbd/labels/*.json` (no Kaggle login) |
| 1 | `scripts/01_index_buildings.py` | laptop (needs labels only) | `outputs/building_index.csv` plus per-event class counts |
| 2 | `scripts/02_build_subset.py` | laptop | `outputs/crop_manifest.csv`, `split_manifest.csv`, `class_map.json`, `class_map_4.json` |
| 3 | `scripts/03_extract_crops.py` | **Kaggle** (needs the 8 GB of images) | `data/crops/<image_id>_{pre,post}.png` |
| 4 | `scripts/04_train_model.py` | **Kaggle** (GPU if available; this run used 4 CPU cores) | `models/*.pth`, `outputs/train_history_*.csv` |
| 5 | `scripts/05_run_inference_log.py` | **laptop CPU**, deliberately | `outputs/model_log.csv`, `outputs/model_memory.json` |
| 6 | `scripts/06_verify_log.py` | laptop | acceptance checks plus test-set metrics |

Steps 3 and 4 run on Kaggle because downloading the 10 GB archive to the laptop ran at about 300 KB/s
(around 10 hours). The images never leave Kaggle; only the crops (540 MB zipped) and the weights are downloaded.

Kaggle attached no GPU to this account's jobs, even with `enable_gpu` set and the phone verified; a one-line
probe job confirmed `torch.cuda.is_available() == False`. The models were therefore trained on Kaggle's free
4-core CPU sessions. That is slower (about 2 / 11 / 12 min per epoch for MobileNet / ResNet / EfficientNet)
but gives the same results.

## For teammates: getting the data

```powershell
git clone https://github.com/SriharshithaAnkareddy25/ResQRoute-.git
```

- **What's in the repo:** `outputs/model_log.csv`, `outputs/class_map.json`, `outputs/crop_manifest.csv`,
  `outputs/split_manifest.csv` and `models/*.pth`.
- **Crop images:** the private Kaggle dataset `harshithaankareddy/resqroute-crops`. Ask the owner to add you as
  a collaborator. Either attach it to a Kaggle notebook, or download it and unzip so that the files land in
  `data/crops/<image_id>_{pre,post}.png`. The paths in the CSVs are relative to the repo root.
- **The fixed split:** never re-split; always join on `image_id` with `outputs/split_manifest.csv`.

### Setup (laptop)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Steps 1–2: index and subset (laptop, labels only)

```powershell
python scripts/00_download_labels.py --out C:/data/xbd     # 2,799 label JSONs, no Kaggle login, ~25 min
python scripts/01_index_buildings.py --raw C:/data/xbd
python scripts/02_build_subset.py --events hurricane-michael,hurricane-matthew,santa-rosa-wildfire --per-class 2000
```

### Steps 3–4: crops and training (Kaggle)

The notebook is [`kaggle/resqroute_kaggle.ipynb`](kaggle/resqroute_kaggle.ipynb). It was run as a private
Kaggle job (`harshithaankareddy/resqroute-train`). It needs no internet and has three inputs:

- the xView2 dataset;
- a private dataset `harshithaankareddy/resqroute-code`, containing `scripts/*.py`, `crop_manifest.csv`,
  `split_manifest.csv` and `class_map*.json` uploaded as flat files;
- a private dataset `harshithaankareddy/resqroute-imagenet-weights`, containing the torchvision ImageNet
  checkpoints, served to torchvision via `TORCH_HOME`.

It uses a GPU when Kaggle attaches one and otherwise trains on CPU. The Kaggle job was created and run from the
laptop with the Kaggle CLI. A `kernel-metadata.json` (not in the repo) lists the three input datasets:

```powershell
kaggle datasets create -p <folder with the flat files + dataset-metadata.json>
kaggle kernels push -p <folder with the notebook + kernel-metadata.json>
kaggle kernels output harshithaankareddy/resqroute-train -p kaggle_out
```

Clicking through the Kaggle UI also works: import the notebook, add the three inputs, then Run All.
Unzip `models_outputs.zip` into the repo root to get `models/` and `outputs/`. With `ZIP_CROPS = True` the job
also writes `crops.zip`, which unzips to `data/crops/`.

On the laptop, `data/crops` is a directory junction to `C:/data/resqroute_crops`. This keeps 12,000 PNGs out of
OneDrive sync; the scripts don't notice the difference.

### Steps 5–6: CPU log and checks (laptop)

```powershell
python scripts/05_run_inference_log.py          # resumable; about 1-2 h for ~6k crops
python scripts/06_verify_log.py
```

## Design decisions (and where they differ from the original brief)

- **Script numbering.** Indexing (1) and subsetting (2) come *before* crop extraction (3). Only the sampled
  buildings are cropped, so the 8 GB of images are only needed on Kaggle.
- **Label schema** was checked against real files, not assumed. Each building is
  `json["features"]["xy"][i]` with `properties.{feature_type, subtype, uid}` and a pixel-coordinate
  `wkt` polygon. Pre-disaster JSONs share the uids but have no `subtype`. Some tiles contain no buildings.
- **Classes.**
  - `damage_class_4` is the raw `subtype`.
  - `damage_class_3` merges `major-damage` and `destroyed` into `severe-damage`.
  - `un-classified` buildings are dropped.
  - `class_map.json` fixes the probability order as `0 no-damage, 1 minor-damage, 2 severe-damage`.
- **Filtering.** Polygons under 150 px² (about 12×12 px, roughly 3–4 m across) are dropped, because they are
  too small to judge damage.
- **Balancing.** At most `--per-class` buildings per 3-class label. Inside `severe-damage`, major-damage and
  destroyed are drawn 50/50 where possible.
- **Split.** 70/15/15, stratified by `damage_class_3` **and grouped by source tile**
  (`StratifiedGroupKFold`, seed 42). All buildings from one 1024×1024 image are in the same split. Otherwise
  neighbouring buildings from the same scene would leak between train and test and inflate test scores.
- **Crops.**
  1. Take the polygon bounding box and add 20% padding on each side.
  2. Enlarge the window to at least 32 px, then clip it to the tile.
  3. Cut the same window from the pre and post images.
  4. Letterbox to 224×224, keeping the aspect ratio with black padding.
- **Training.**
  - Inputs and labels: post-disaster crop only, 3-class labels (`--classes 4` is supported).
  - Model and optimiser: ImageNet weights, full fine-tune, AdamW (lr 3e-4) with a cosine schedule,
    inverse-frequency class weights and **label smoothing 0.1**.
  - Augmentation: flips, 90° rotations and light colour jitter.
  - Stopping: up to 20 epochs. The checkpoint with the best **val macro-F1** is kept, and training stops after
    6 epochs without improvement.
  - Why this recipe:
    - A first run selected checkpoints by val loss (patience 5). That stopped the larger models early;
      EfficientNet-B0 was kept at epoch 3 (val acc 0.762) although epoch 7 reached 0.782.
    - Macro-F1 is the metric the project reports.
    - Label smoothing reduces over-confident softmax outputs. That matters because the routing controller
      reads the confidence, entropy and top-2 margin from these probabilities.
    - Both settings are flags (`--select loss`, `--label-smoothing 0`).
- **Timing.** CPU only with `torch.set_num_threads(1)`.
  - Each model gets 10 warm-up passes first.
  - Only the forward pass is timed (batch of 1), 3 times per crop, and the median is recorded.
  - The three models run interleaved per crop, so they see the same machine conditions.
- **Memory (`*_mem_mb`)** is a **per-model** number, not a per-image one.
  - What it measures: the peak process-RSS increase from loading the weights plus one forward pass, in a fresh
    subprocess, median of 3 runs. The same value is written to every row.
  - Why not per-image RSS deltas: every input is 224×224, so an image's memory cost depends only on the model.
    In testing, per-image deltas measured ~0 MB after warm-up, because the allocator reuses freed memory.
  - Details are in `outputs/model_memory.json`.

## Things the team needs to know

### Building counts per event (labelled train set, before filtering)

| event | no | minor | major | destroyed | minor % | severe % |
|---|---:|---:|---:|---:|---:|---:|
| hurricane-harvey | 11,423 | 2,663 | 8,238 | 401 | 11.7 | 38.0 |
| hurricane-matthew | 2,515 | 6,548 | 1,945 | 2,147 | 49.8 | 31.1 |
| santa-rosa-wildfire | 9,285 | 78 | 63 | 3,471 | 0.6 | 27.4 |
| hurricane-florence | 4,689 | 132 | 1,245 | 54 | 2.2 | 21.2 |
| palu-tsunami | 25,455 | 1 | 571 | 4,966 | 0.0 | 17.9 |
| socal-fire | 8,638 | 83 | 51 | 1,329 | 0.8 | 13.7 |
| hurricane-michael | 14,588 | 5,207 | 1,902 | 757 | 23.2 | 11.8 |
| guatemala-volcano | 639 | 9 | 9 | 25 | 1.3 | 5.0 |
| midwest-flooding | 8,128 | 149 | 119 | 75 | 1.8 | 2.3 |
| mexico-earthquake | 32,066 | 110 | 18 | 2 | 0.3 | 0.1 |

### Chosen events and why

The chosen events are **hurricane-michael, hurricane-matthew and santa-rosa-wildfire**.

- **Minor damage is the scarce class.** Only Matthew, Michael and Harvey have more than a few hundred
  minor-damage buildings, so at least two of them are needed.
- **Michael and Matthew** provide 11.7k minor-damage buildings plus wind damage of every severity.
- **Santa Rosa wildfire** adds a second disaster type (fire) with 3.5k destroyed buildings. That helps
  out-of-distribution and per-event analysis later.
- **Harvey was left out.** Its 8k "major-damage" labels are mostly flood damage, which is often invisible
  from above, so they are known to be noisy labels.

### Final subset (6,000 crops from 668 tiles)

| split | no-damage | minor-damage | severe-damage | total |
|---|---:|---:|---:|---:|
| train | 1,434 | 1,433 | 1,432 | 4,299 |
| val | 282 | 283 | 284 | 849 |
| test | 284 | 284 | 284 | 852 |

- **Severe-damage composition:** 1,000 major-damage plus 1,000 destroyed, so the 4-class column is usable
  (no 1,000 / minor 2,000 / major 1,000 / destroyed 1,000).
- **Per event:**

  | event | crops | composition |
  |---|---:|---|
  | michael | 2,702 | mixed classes |
  | matthew | 1,848 | mostly minor and severe; only 150 no-damage |
  | santa-rosa | 1,450 | almost only no-damage or destroyed |

**Caveat:** class mix differs a lot by event, so a model could partly learn "which scene is this" as a
shortcut. Report per-event metrics as well as the overall numbers.
- **Model cost ordering:** on this laptop's CPU, ResNet-18 is *slower and uses more memory* than
  EfficientNet-B0, even though ResNet-18 is meant to be the "medium" stage. A smoke test measured about 100 ms
  vs 51 ms per image and 173 MB vs 122 MB. The routing controller should use the measured `*_time_ms`/`*_mem_mb`
  costs, not assume that stage order equals cost order. Alternatively the team can swap the advanced stage
  (for example to EfficientNet-B2 or ResNet-50).
