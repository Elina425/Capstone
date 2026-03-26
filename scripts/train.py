#!/usr/bin/env python3
"""Train BiLSTM (angles) and ST-GCN (joint coords) on cached NPZ windows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from exercise_cls.config import ARTIFACTS_DIR, SEQUENCES_DIR
from exercise_cls.extract import load_split
from exercise_cls.models import AngleBiLSTM, STGCNClassifier


class WindowDataset(Dataset):
    def __init__(self, manifest_dir: Path, files: list[str], mode: str):
        self.manifest_dir = manifest_dir
        self.files = files
        self.mode = mode
        self._samples: list[tuple[str, int]] = []
        for f in files:
            path = manifest_dir / f
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            n = z["angle_windows"].shape[0]
            for i in range(n):
                self._samples.append((f, i))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        fn, wi = self._samples[idx]
        z = np.load(self.manifest_dir / fn, allow_pickle=True)
        y = int(z["label"])
        if self.mode == "bilstm":
            x = np.nan_to_num(z["angle_windows"][wi].astype(np.float32))
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = np.nan_to_num(z["stgcn_windows"][wi].astype(np.float32))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def run_epoch(model, loader, opt, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    if len(loader) == 0:
        return float("nan"), float("nan")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        if train:
            loss.backward()
            opt.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences-dir", type=Path, default=SEQUENCES_DIR)
    parser.add_argument("--model", choices=("bilstm", "stgcn"), default="bilstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "checkpoints")
    args = parser.parse_args()
    manifest = args.sequences_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}. Run extract pipeline first.")

    train_f, val_f, test_f = load_split(manifest)
    train_ds = WindowDataset(args.sequences_dir, train_f, args.model)
    val_ds = WindowDataset(args.sequences_dir, val_f, args.model)
    test_ds = WindowDataset(args.sequences_dir, test_f, args.model)
    if len(train_ds) == 0:
        raise SystemExit("No training samples. Build manifest with scripts/extract_sequences.py")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    has_val = len(val_loader) > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "bilstm":
        model = AngleBiLSTM(num_classes=2).to(device)
    else:
        model = STGCNClassifier(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_path = args.out / f"best_{args.model}.pt"
    last_path = args.out / f"last_{args.model}.pt"
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, None, device, train=False)
        if has_val:
            print(
                f"epoch {ep:03d}  train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"val loss={va_loss:.4f} acc={va_acc:.3f}"
            )
            metric = va_acc
        else:
            print(f"epoch {ep:03d}  train loss={tr_loss:.4f} acc={tr_acc:.3f}  (no val split)")
            metric = tr_acc
        torch.save({"model": model.state_dict(), "epoch": ep}, last_path)
        if metric > best_val:
            best_val = metric
            torch.save({"model": model.state_dict(), "epoch": ep}, best_path)

    load_path = best_path if best_path.exists() else last_path
    model.load_state_dict(torch.load(load_path, map_location=device)["model"])
    te_loss, te_acc = run_epoch(model, test_loader, None, device, train=False)
    print(f"test loss={te_loss:.4f} acc={te_acc:.3f}")
    report = {
        "model": args.model,
        "test_accuracy": te_acc,
        "test_loss": te_loss,
        "best_metric": best_val,
        "checkpoint": str(load_path),
    }
    (args.out / f"report_{args.model}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
