"""Shared constants and helpers for the ResQRoute data pipeline."""
import os
from pathlib import Path

# Root that data/, models/, outputs/ live under. Defaults to the repo; on Kaggle set
# RESQROUTE_ROOT=/kaggle/working because the uploaded scripts folder is read-only.
REPO = Path(os.environ.get("RESQROUTE_ROOT", Path(__file__).resolve().parent.parent))
OUTPUTS = REPO / "outputs"
SEED = 42

# Fixed class order: probability vectors in model_log.csv are read positionally by the controller.
CLASSES_3 = ["no-damage", "minor-damage", "severe-damage"]
CLASSES_4 = ["no-damage", "minor-damage", "major-damage", "destroyed"]

TO_4 = {c: c for c in CLASSES_4}  # 'un-classified' and anything unexpected map to None
TO_3 = {"no-damage": "no-damage", "minor-damage": "minor-damage",
        "major-damage": "severe-damage", "destroyed": "severe-damage"}


# arch name -> (weights file, column prefix in model_log.csv). Ordered small -> large as in the cascade.
MODELS = {
    "mobilenet_v3_small": ("models/mobilenetv3_small.pth", "mobilenet"),
    "resnet18": ("models/resnet18.pth", "resnet18"),
    "efficientnet_b0": ("models/efficientnet_b0.pth", "efficientnet"),
}

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """torchvision backbone with its final layer replaced by a num_classes head."""
    import torch.nn as nn
    from torchvision import models as tvm

    if arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


def eval_transform():
    """Crops are already 224x224, so evaluation is just tensor conversion + ImageNet normalisation."""
    from torchvision import transforms as T

    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def find_raw_dir(raw: str | None) -> Path:
    """Locate the folder holding xBD `images/` and `labels/`.

    Accepts an explicit path, else searches the usual Kaggle-notebook and local locations,
    since the Kaggle input mount path differs between notebook versions.
    """
    if raw:
        p = Path(raw)
        if not (p / "labels").is_dir():
            raise SystemExit(f"{p} has no labels/ folder")
        return p
    for root in [Path("/kaggle/input"), Path("C:/data/xbd")]:
        if not root.exists():
            continue
        if (root / "labels").is_dir():
            return root
        for lab in sorted(root.rglob("labels")):
            if lab.is_dir() and any(lab.glob("*_post_disaster.json")) and "test" not in lab.parts:
                return lab.parent
    raise SystemExit("Could not find xBD labels/ folder; pass --raw")
