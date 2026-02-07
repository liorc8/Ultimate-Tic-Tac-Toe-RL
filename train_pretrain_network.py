from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from game_network import GameNetwork


class PretrainDataset(Dataset):
    """Memory-mapped dataset for (state, pi, z) training.

    Expected .npz keys:
        - states: (N, C, 9, 9) float32
        - pis:    (N, 81)      float32 (each row sums to 1)
        - zs:     (N,)         float32 in {-1,0,+1}
    """

    def __init__(self, npz_path: str, indices: np.ndarray):
        self.data = np.load(npz_path, mmap_mode="r")
        self.states = self.data["states"]
        self.pis = self.data["pis"]
        self.zs = self.data["zs"]
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        s = self.states[idx].astype(np.float32, copy=False)
        pi = self.pis[idx].astype(np.float32, copy=False)
        z = np.float32(self.zs[idx])
        return s, pi, z


def policy_loss_from_logits(logits: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a distribution target (soft labels)."""
    logp = torch.log_softmax(logits, dim=1)
    return -(target_pi * logp).sum(dim=1).mean()


@dataclass
class TrainConfig:
    dataset_path: str = "mcts_pretrain_data.npz"
    out_path: str = "game_network_pretrained.pt"
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.5
    val_split: float = 0.05
    seed: int = 42
    num_workers: int = 0
    device: Optional[str] = None
    amp: bool = True
    patience: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    data = np.load(cfg.dataset_path, mmap_mode="r")
    n = int(data["states"].shape[0])

    indices = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)

    val_n = max(1, int(n * cfg.val_split))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    train_ds = PretrainDataset(cfg.dataset_path, train_idx)
    val_ds = PretrainDataset(cfg.dataset_path, val_idx)

    pin = (pick_device(cfg.device) == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    return train_loader, val_loader, n


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: str, value_loss_weight: float) -> Tuple[float, float, float]:
    model.eval()
    mse = nn.MSELoss()

    total_loss = 0.0
    total_pol = 0.0
    total_val = 0.0
    batches = 0

    for states, pis, zs in val_loader:
        states = states.to(device, non_blocking=True)
        pis = pis.to(device, non_blocking=True)
        zs = zs.to(device, non_blocking=True)

        logits, v = model(states)
        pol = policy_loss_from_logits(logits, pis)
        val = mse(v, zs)
        loss = pol + value_loss_weight * val

        total_loss += float(loss.item())
        total_pol += float(pol.item())
        total_val += float(val.item())
        batches += 1

    if batches == 0:
        return math.inf, math.inf, math.inf

    return total_loss / batches, total_pol / batches, total_val / batches


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    print("=== Pre-training GameNetwork from MCTS dataset ===")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Device:  {device}")
    print(f"Epochs:  {cfg.epochs}")
    print(f"Batch:   {cfg.batch_size}")
    print(f"LR:      {cfg.lr}")
    print(f"ValSplit:{cfg.val_split}")

    train_loader, val_loader, n = make_loaders(cfg)
    print(f"Samples: {n} | Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = GameNetwork()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))

    mse = nn.MSELoss()

    best_val = math.inf
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_pol = 0.0
        running_val = 0.0

        for step, (states, pis, zs) in enumerate(train_loader, start=1):
            states = states.to(device, non_blocking=True)
            pis = pis.to(device, non_blocking=True)
            zs = zs.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device == "cuda")):
                logits, v = model(states)
                pol = policy_loss_from_logits(logits, pis)
                val = mse(v, zs)
                loss = pol + cfg.value_loss_weight * val

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item())
            running_pol += float(pol.item())
            running_val += float(val.item())

            if step % 200 == 0:
                avg = running_loss / step
                print(f"Epoch {epoch}/{cfg.epochs} | step {step:5d}/{len(train_loader)} | loss {avg:.4f}")

        val_loss, val_pol, val_val = evaluate(model, val_loader, device, cfg.value_loss_weight)
        train_loss = running_loss / max(1, len(train_loader))
        train_pol = running_pol / max(1, len(train_loader))
        train_val = running_val / max(1, len(train_loader))

        print(f"Epoch {epoch}: train loss={train_loss:.4f} (pol={train_pol:.4f}, val={train_val:.4f})")
        print(f"          val   loss={val_loss:.4f} (pol={val_pol:.4f}, val={val_val:.4f})")

        scheduler.step(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            model.save(cfg.out_path)
            print(f"✅ Saved best model to: {cfg.out_path} (epoch {epoch}, val={val_loss:.4f})")
        else:
            bad_epochs += 1
            print(f"⚠️  No improvement. bad_epochs={bad_epochs}/{cfg.patience}")
            if bad_epochs >= cfg.patience:
                print(f"🛑 Early stopping. Best epoch={best_epoch}, val={best_val:.4f}")
                break

    print("Done.")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Pre-train GameNetwork on MCTS self-play dataset (.npz).")
    parser.add_argument("--data", type=str, default="mcts_pretrain_data.npz")
    parser.add_argument("--out", type=str, default="game_network_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--value-w", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    device = pick_device(args.device)
    epochs = (6 if device == "cuda" else 3) if args.epochs is None else int(args.epochs)
    batch = (512 if device == "cuda" else 256) if args.batch is None else int(args.batch)

    return TrainConfig(
        dataset_path=args.data,
        out_path=args.out,
        epochs=epochs,
        batch_size=batch,
        lr=float(args.lr),
        weight_decay=float(args.wd),
        value_loss_weight=float(args.value_w),
        val_split=float(args.val_split),
        seed=int(args.seed),
        num_workers=int(args.workers),
        device=args.device,
        amp=(not args.no_amp),
        patience=int(args.patience),
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
