#!/usr/bin/env python3
"""Train a CNN classifier on an Alzheimer MRI dataset (PyTorch + GPU/AMP, stratified val, early stop)."""

from __future__ import annotations
import argparse, json, logging, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

from app.models import SimpleCNN


logger = logging.getLogger(__name__)


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split_indices(
    targets: List[int], val_frac: float, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Return train_idx, val_idx with per-class stratification."""
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(idx)

    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac))) if len(idxs) > 1 else 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -----------------------
# Model
# -----------------------
# Architecture is defined in app.models.SimpleCNN

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=preds.device
    )
    for t, p in zip(targets, preds):
        confusion[t.long(), p.long()] += 1
    correct = confusion.diagonal().sum().item()
    total = confusion.sum().item()
    accuracy = correct / total if total else 0.0

    recalls = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        support = confusion[c, :].sum().item()
        recalls.append((tp / support) if support else 0.0)
    balanced_acc = sum(recalls) / num_classes if num_classes else 0.0
    return accuracy, balanced_acc, confusion


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(dim=1))
        targs.append(y)
    preds = torch.cat(preds)
    targs = torch.cat(targs)
    num_classes = (
        len(loader.dataset.dataset.classes)
        if isinstance(loader.dataset, Subset)
        else len(loader.dataset.classes)
    )
    acc, bacc, cm = compute_metrics(preds, targs, num_classes)
    return acc, bacc, cm


# -----------------------
# Data
# -----------------------
class TransformSubset(Dataset):
    """Wrap an ImageFolder and apply a transform to a subset of indices."""

    def __init__(
        self,
        dataset: datasets.ImageFolder,
        indices: Sequence[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.loader = dataset.loader
        self.target_transform = dataset.target_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        path, target = self.dataset.samples[sample_idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    @property
    def classes(self):
        return self.dataset.classes


def resolve_num_workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    cpu_count = os.cpu_count() or 2
    return max(2, min(8, cpu_count // 2))


def build_transforms(
    image_size: Tuple[int, int],
    augment: bool,
    grayscale: bool,
    weights: EfficientNet_B0_Weights | None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    height, width = image_size

    if weights:
        preprocess = weights.transforms()
        mean = preprocess.mean
        std = preprocess.std
    elif grayscale:
        mean, std = [0.5], [0.25]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    base_transforms: List = [transforms.Resize((height, width))]
    if grayscale:
        channels = 3 if weights else 1
        base_transforms.append(transforms.Grayscale(num_output_channels=channels))

    def compose_pipeline(steps: Sequence) -> transforms.Compose:
        return transforms.Compose([*steps, transforms.ToTensor(), transforms.Normalize(mean, std)])

    if augment:
        augmentations: List = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15), shear=8, scale=(0.85, 1.15)
            ),
        ]
        if not grayscale:
            augmentations.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            )
        train_transforms = compose_pipeline([*base_transforms, *augmentations])
    else:
        train_transforms = compose_pipeline(base_transforms)

    eval_transforms = compose_pipeline(base_transforms)
    return train_transforms, eval_transforms


def prepare_dataloaders(
    root: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int | None,
    augment: bool,
    grayscale: bool,
    val_split: float,
    device: torch.device,
    weights: EfficientNet_B0_Weights | None,
):
    train_dir, test_dir = root / "train", root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"'train' ve 'test' klasörleri {root} içinde bekleniyor."
        )

    train_tf, eval_tf = build_transforms(img_size, augment, grayscale, weights)

    base_train = datasets.ImageFolder(train_dir)
    targets: List[int] = list(base_train.targets)
    train_indices, val_indices = stratified_split_indices(targets, val_split)
    train_ds = TransformSubset(base_train, train_indices, train_tf)
    val_ds = TransformSubset(base_train, val_indices, eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    pin = device.type == "cuda"
    resolved_workers = resolve_num_workers(num_workers)
    loader_kwargs = dict(batch_size=batch_size, num_workers=resolved_workers, pin_memory=pin)
    if resolved_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    loaders = dict(
        train=DataLoader(train_ds, shuffle=True, **loader_kwargs),
        val=DataLoader(val_ds, shuffle=False, **loader_kwargs),
        test=DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )

    logger.info("Train samples: %s", len(train_ds))
    logger.info("Val samples: %s", len(val_ds))
    logger.info("Test samples: %s", len(test_ds))
    logger.info("Classes: %s", base_train.classes)

    return loaders, base_train.classes, targets, train_indices


def compute_class_weights(
    targets: Sequence[int], indices: Sequence[int], num_classes: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        counts[targets[idx]] += 1
    weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
    return weights.to(device)


def build_model(
    num_classes: int,
    arch: str,
    grayscale: bool,
    efficientnet_train_from: int,
    weights: EfficientNet_B0_Weights | None,
) -> nn.Module:
    if arch == "efficientnet_b0":
        if weights is None:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

        for param in model.parameters():
            param.requires_grad = False

        freeze_from = max(0, min(len(model.features), efficientnet_train_from))
        for idx, block in enumerate(model.features):
            if idx >= freeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    if arch == "simple_cnn":
        in_ch = 1 if grayscale else 3
        return SimpleCNN(num_classes=num_classes, in_channels=in_ch)

    raise ValueError(f"Unknown architecture '{arch}'")



# -----------------------
# Train loop
# -----------------------
def train(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    early_stop: bool,
    patience: int,
    class_weights: torch.Tensor,
    writer: "SummaryWriter | None" = None,
):
    if class_weights.device != device:
        class_weights = class_weights.to(device)
    logger.info("Class weights: %s", class_weights.cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(loaders["train"]))
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    history = {"train_loss": [], "val_acc": [], "val_bacc": []}
    best_bacc = -1.0
    best_state = None
    patience_left = patience

    model.to(device)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.time()

        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(use_amp and device.type in ("cuda", "mps"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * x.size(0)

        train_loss = running / len(loaders["train"].dataset)
        history["train_loss"].append(train_loss)

        val_acc, val_bacc, _ = evaluate(model, loaders["val"], device)
        history["val_acc"].append(val_acc)
        history["val_bacc"].append(val_bacc)
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
            writer.add_scalar("balanced_accuracy/val", val_bacc, epoch)

        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %s/%s | loss=%.4f | val_acc=%.4f | val_bacc=%.4f | %.1fs",
            epoch,
            epochs,
            train_loss,
            val_acc,
            val_bacc,
            epoch_time,
        )

        improved = val_bacc > best_bacc + 1e-6
        if improved:
            best_bacc = val_bacc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
            logger.info("  [+] Best model updated! (patience: %s)", patience_left)
        else:
            if early_stop:
                patience_left -= 1
                logger.info("  No improvement (patience: %s/%s)", patience_left, patience)
                if patience_left <= 0:
                    logger.info(
                        "Early stopping @ epoch %s (best val_bacc=%.4f).",
                        epoch,
                        best_bacc,
                    )
                    break

    total_time = time.time() - start_time
    logger.info("Training completed in %.1fs (%.1fm)", total_time, total_time / 60)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return history, best_bacc



# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on Alzheimer MRI (PyTorch).")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--img-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W")
    )
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count (auto when omitted)")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--fp16", action="store_true", help="Enable CUDA AMP.")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of train used as validation.",
    )
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grayscale",
        action="store_true",
        default=False,
        help="Use single-channel pipeline & model.",
    )
    p.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    p.add_argument("--model-path", type=Path, default=Path("alzheimer_cnn_best.pt"))
    p.add_argument("--log-dir", type=Path, default=None, help="TensorBoard log directory (disabled by default)")
    p.add_argument(
        "--arch",
        type=str,
        choices=("efficientnet_b0", "simple_cnn"),
        default="efficientnet_b0",
        help="Model architecture to train.",
    )
    p.add_argument(
        "--efficientnet-train-from",
        type=int,
        default=4,
        help="For EfficientNetB0, unfreeze blocks starting from this index (0-based).",
    )
    return p.parse_args()


# -----------------------
# Main
# -----------------------
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ==> EKLEMENİZ GEREKEN EKSİK SATIR BU <==
    global logger
    logger = logging.getLogger(__name__)

    logger.info("Training run started")
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type in ["cuda", "mps"]:
        if device.type == "cuda":
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
            logger.info(
                "GPU Memory: %.1f GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        else:  # mps
            logger.info("Using Apple Metal GPU: %s", device)
    else:
        if args.require_gpu:
            raise RuntimeError(
                "GPU device yok. CPU eğitimine izin vermek için --require-gpu bayrağını kaldırın."
            )
        logger.warning("Warning: Uyumlu bir GPU bulunamadı, CPU üzerinde eğitim.")

    logger.info(
        "Config: epochs=%s, batch_size=%s, lr=%s, arch=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.arch,
    )
    logger.info("Augment: %s, Grayscale: %s, FP16: %s", args.augment, args.grayscale, args.fp16)

    weights = None
    if args.arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        if args.grayscale:
            logger.warning("EfficientNetB0 RGB giriş bekler; grayscale devre dışı bırakıldı.")
            args.grayscale = False

    loaders, classes, targets, train_indices = prepare_dataloaders(
        root=data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        grayscale=args.grayscale,
        val_split=args.val_split,
        device=device,
        weights=weights,
    )

    class_weights = compute_class_weights(
        targets=targets,
        indices=train_indices,
        num_classes=len(classes),
        device=device,
    )

    writer = None
    if args.log_dir:
        # SummaryWriter'ın import edildiğini varsayıyoruz
        # from torch.utils.tensorboard import SummaryWriter
        if SummaryWriter is None:
            logger.warning("TensorBoard bulunamadı; `pip install tensorboard` ile yükleyin.")
        else:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(args.log_dir))
            logger.info("TensorBoard logları: %s", args.log_dir)

    logger.info("DataLoader workers: %s", loaders['train'].num_workers)

    model = build_model(
        num_classes=len(classes),
        arch=args.arch,
        grayscale=args.grayscale,
        efficientnet_train_from=args.efficientnet_train_from,
        weights=weights,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: %s (trainable: %s)",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    history, best_val_bacc = train(
        model=model,
        loaders=loaders,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=args.fp16,
        early_stop=args.early_stop,
        patience=args.patience,
        class_weights=class_weights,
        writer=writer,
    )

    # Final evaluation on held-out test
    test_acc, test_bacc, cm = evaluate(model, loaders["test"], device)
    logger.info("Test accuracy: %.4f | Test balanced accuracy: %.4f", test_acc, test_bacc)
    print(f"\nTest accuracy: {test_acc:.4f} | Test balanced accuracy: {test_bacc:.4f}")
    cm_cpu = cm.cpu().tolist()

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "history": history,
            "val_best_balanced_accuracy": best_val_bacc,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_bacc,
            "confusion_matrix": cm_cpu,
            "grayscale": args.grayscale,
            "img_size": tuple(args.img_size),
            "input_channels": 1 if args.grayscale else 3,
            "arch": args.arch,
            "efficientnet_train_from": args.efficientnet_train_from,
            "weights": weights.name if weights else None,
            "model_version": "1.0", # DEFAULT_MODEL_VERSION yerine bir değer girdim
            "max_batch_size": args.batch_size,
        },
        args.model_path,
    )
    logger.info("Checkpoint saved: %s", args.model_path)

    print(
        json.dumps(
            {
                "val_best_balanced_accuracy": best_val_bacc,
                "test_accuracy": test_acc,
                "test_balanced_accuracy": test_bacc,
                "saved_model": str(args.model_path.resolve()),
                "classes": classes,
                "arch": args.arch,
                "model_version": "1.0", # DEFAULT_MODEL_VERSION yerine bir değer girdim
                "training_time": "check console output",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()














