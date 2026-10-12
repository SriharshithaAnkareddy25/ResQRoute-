"""Step 4: fine-tune one ImageNet-pretrained classifier on the post-disaster crops.

Train split for fitting, val split for early stopping (on val loss); test is never touched here.
Run on a Kaggle GPU (see README); works on CPU too, just slowly.

  python scripts/04_train_model.py --arch mobilenet_v3_small
  python scripts/04_train_model.py --arch resnet18
  python scripts/04_train_model.py --arch efficientnet_b0
  python scripts/04_train_model.py --arch resnet18 --classes 4   # 4-class variant -> *_4class.pth

Saves models/<name>.pth (state_dict of the best-val-loss epoch) and outputs/train_history_<arch>.csv.
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import CLASSES_3, CLASSES_4, IMAGENET_MEAN, IMAGENET_STD, MODELS, OUTPUTS, REPO, SEED, build_model, eval_transform


class CropDataset(Dataset):
    def __init__(self, df, label_col, classes, tf):
        self.paths = [REPO / p for p in df.post_crop_path]
        self.labels = [classes.index(c) for c in df[label_col]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def train_transform():
    # overhead imagery has no canonical "up", so all flips / 90-degree turns are valid
    return T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, n, preds, labels = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            preds += out.argmax(1).tolist()
            labels += y.tolist()
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    return total / n, acc, f1_score(labels, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(MODELS))
    ap.add_argument("--classes", type=int, default=3, choices=[3, 4])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5, help="early stopping on val loss")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_col, classes = ("damage_class_3", CLASSES_3) if args.classes == 3 else ("damage_class_4", CLASSES_4)

    man = pd.read_csv(OUTPUTS / "crop_manifest.csv").merge(pd.read_csv(OUTPUTS / "split_manifest.csv"), on="image_id")
    tr, va = man[man.split == "train"], man[man.split == "val"]
    dl_tr = DataLoader(CropDataset(tr, label_col, classes, train_transform()), args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(CropDataset(va, label_col, classes, eval_transform()), args.batch_size,
                       num_workers=args.workers, pin_memory=True)

    # inverse-frequency class weights, in case undersampling could not fully balance a class
    counts = tr[label_col].value_counts().reindex(classes).fillna(0).to_numpy()
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    print(f"{args.arch} on {device} | train {len(tr)} val {len(va)} | class counts {dict(zip(classes, counts))}")

    model = build_model(args.arch, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    weights_path, _ = MODELS[args.arch]
    if args.classes == 4:
        weights_path = weights_path.replace(".pth", "_4class.pth")
    (REPO / "models").mkdir(exist_ok=True)

    best, bad, hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, dl_tr, loss_fn, device, opt, scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, dl_va, loss_fn, device)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                         val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))
        flag = ""
        if va_loss < best:
            best, bad, flag = va_loss, 0, "  *saved*"
            torch.save(model.state_dict(), REPO / weights_path)
        else:
            bad += 1
        print(f"ep {ep:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} "
              f"acc {va_acc:.3f} macroF1 {va_f1:.3f} | {time.time() - t:.0f}s{flag}")
        if bad >= args.patience:
            print("early stop")
            break

    suffix = "" if args.classes == 3 else "_4class"
    pd.DataFrame(hist).to_csv(OUTPUTS / f"train_history_{args.arch}{suffix}.csv", index=False)
    best_row = min(hist, key=lambda h: h["val_loss"])
    (OUTPUTS / f"train_summary_{args.arch}{suffix}.json").write_text(json.dumps(
        {"arch": args.arch, "classes": classes, "device": str(device), **vars(args), "best": best_row}, indent=2))
    print(f"best epoch {best_row['epoch']}: val acc {best_row['val_acc']:.3f} macroF1 {best_row['val_f1']:.3f} -> {weights_path}")


if __name__ == "__main__":
    main()
