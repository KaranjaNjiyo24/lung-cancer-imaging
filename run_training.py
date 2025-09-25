"""Training script for multimodal NSCLC classification."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

import yaml

from src.data.metadata_handler import NSCLCMetadataHandler
from src.data.multimodal_dataset import NSCLCMultimodalDataset, create_nsclc_dataloader
from src.models.multimodal_classifier import build_multimodal_classifier
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger("train")


# ---------------------------------------------------------------------------
def setup_logging(log_dir: str) -> None:
    """Configure logging output to file and stdout."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)


# ---------------------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


# ---------------------------------------------------------------------------
def create_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    architecture = model_cfg.get("architecture", "resnet18")
    input_channels = model_cfg.get("input_channels", {"ct": 1, "pet": 1})
    num_classes = int(model_cfg.get("num_classes", 2))
    dropout = float(model_cfg.get("dropout", 0.3))
    fusion = model_cfg.get("fusion", "concat")

    model = build_multimodal_classifier(
        architecture=architecture,
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=dropout,
        fusion=fusion,
    )
    return model


# ---------------------------------------------------------------------------
def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        inputs = _prepare_inputs(batch, device)
        targets = batch["label"].to(device)

        outputs = model(**inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(train_loader))


# ---------------------------------------------------------------------------
def validate(model: nn.Module, val_loader: DataLoader, criterion, device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = _prepare_inputs(batch, device)
            targets = batch["label"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

    val_loss = running_loss / max(1, len(val_loader))
    accuracy = correct / max(1, total)
    return {"loss": val_loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
def _prepare_inputs(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    inputs: Dict[str, torch.Tensor] = {}
    if batch.get("ct") is not None:
        inputs["ct"] = batch["ct"].to(device)
    if batch.get("pet") is not None:
        inputs["pet"] = batch["pet"].to(device)
    inputs["ct_available"] = batch["ct_available"].to(device)
    inputs["pet_available"] = batch["pet_available"].to(device)
    return inputs


# ---------------------------------------------------------------------------
def save_checkpoint(state: Dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)


# ---------------------------------------------------------------------------
def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Path, device: torch.device) -> int:
    if not checkpoint_path.exists():
        LOGGER.warning("Checkpoint %s not found; skipping resume", checkpoint_path)
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)
    LOGGER.info("Loaded checkpoint from %s (epoch %s)", checkpoint_path, start_epoch)
    return start_epoch


# ---------------------------------------------------------------------------
def prepare_dataloaders(config: Dict[str, Any], data_root: Path, metadata_handler: NSCLCMetadataHandler):
    data_cfg = config.get("data", {})

    train_dataset = NSCLCMultimodalDataset(
        data_root=str(data_root),
        metadata_handler=metadata_handler,
        transform=None,
        target_size=tuple(data_cfg.get("target_size", [128, 128, 64])),
        require_both_modalities=data_cfg.get("require_both_modalities", False),
        label_field=data_cfg.get("label_field"),
        fallback_label=int(data_cfg.get("fallback_label", 0)),
        augmentation_prob=float(data_cfg.get("augmentation_prob", 0.0)),
        seed=int(data_cfg.get("seed", 42)),
        fill_missing_with_zeros=bool(data_cfg.get("fill_missing_with_zeros", True)),
    )

    val_ratio = float(data_cfg.get("val_split", 0.2))
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(int(data_cfg.get("seed", 42)))
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = create_nsclc_dataloader(
        train_subset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )

    val_loader = create_nsclc_dataloader(
        val_subset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
def configure_optimizer(model: nn.Module, config: Dict[str, Any]):
    optim_cfg = config.get("optimizer", {})
    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_cfg = config.get("scheduler", {})
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max", 10)),
            eta_min=float(sched_cfg.get("eta_min", 1e-6)),
        )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
def maybe_init_wandb(config: Dict[str, Any]) -> Optional[Any]:
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enable", False) or wandb is None:
        return None

    wandb.init(
        project=wandb_cfg.get("project", "nsclc"),
        config=config,
        name=wandb_cfg.get("run_name"),
        mode=wandb_cfg.get("mode", "online"),
    )
    return wandb


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="NSCLC Multimodal Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data_path", type=str, required=True, help="Data root path")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV path")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    setup_logging(args.log_dir)
    LOGGER.info("Starting training with args: %s", args)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    metadata_handler = NSCLCMetadataHandler(args.metadata_csv)
    metadata_handler.load_metadata()

    train_loader, val_loader = prepare_dataloaders(config, Path(args.data_path), metadata_handler)

    model = create_model(config).to(device)
    optimizer, scheduler = configure_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_accuracy = 0.0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_path
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)

    wandb_run = maybe_init_wandb(config)

    train_cfg = config.get("train", {})
    num_epochs = int(train_cfg.get("epochs", 10))
    validate_every = int(train_cfg.get("validate_every", 1))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1))

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        LOGGER.info("Epoch %d | Train Loss: %.4f", epoch + 1, train_loss)

        if scheduler:
            scheduler.step()

        if (epoch + 1) % validate_every == 0:
            metrics = validate(model, val_loader, criterion, device)
            LOGGER.info("Epoch %d | Val Loss: %.4f | Val Acc: %.4f", epoch + 1, metrics["loss"], metrics["accuracy"])
            if wandb_run:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}})

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }, best_path)

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
            }, checkpoint_path)

    final_path = checkpoint_dir / "final.pt"
    save_checkpoint({
        "epoch": num_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
    }, final_path)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
