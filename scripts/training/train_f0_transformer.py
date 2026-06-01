#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.training.tone_dataset import CLASS_TO_TONE, SyllableF0Dataset, compute_train_norm_stats  # noqa: E402


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class F0Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.positional = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(f0)
        x = self.positional(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((5, 5), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            f0 = batch["f0"].to(device)
            labels = batch["label"].to(device)
            logits = model(f0)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += int((preds == labels).sum().item())
            loss_sum += float(loss.item()) * labels.numel()
            for y_true, y_pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(y_true), int(y_pred)] += 1

    per_tone = {}
    f1_values = []
    for idx, tone in CLASS_TO_TONE.items():
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_tone[tone] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[idx, :].sum()),
        }

    return {
        "loss": loss_sum / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_tone": per_tone,
        "confusion": confusion.tolist(),
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.contour_key != "f0_hz":
        raise ValueError("train_f0_transformer.py expects raw --contour-key f0_hz for train-only normalization")

    norm_stats = compute_train_norm_stats(args.features, args.contours, contour_key=args.contour_key)
    train_ds = SyllableF0Dataset(
        args.features,
        args.contours,
        split="train",
        contour_key=args.contour_key,
        norm_stats=norm_stats,
    )
    val_ds = SyllableF0Dataset(
        args.features,
        args.contours,
        split="val",
        contour_key=args.contour_key,
        norm_stats=norm_stats,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = F0Transformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best = {"epoch": 0, "accuracy": -1.0, "macro_f1": -1.0}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        for batch in train_loader:
            f0 = batch["f0"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(f0)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_total += labels.numel()
            train_correct += int((preds == labels).sum().item())
            train_loss_sum += float(loss.item()) * labels.numel()

        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **val_metrics}
            if args.checkpoint:
                checkpoint = Path(args.checkpoint)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "best": best,
                        "normalization": {
                            "source_split": "train",
                            "contour_key": args.contour_key,
                            "stats": {key: {"mean": value[0], "std": value[1]} for key, value in norm_stats.items()},
                        },
                    },
                    checkpoint,
                )

    return {
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "train_tones": dict(Counter(row["tone"] for row in train_ds.rows)),
        "val_tones": dict(Counter(row["tone"] for row in val_ds.rows)),
        "normalization": {
            "source_split": "train",
            "contour_key": args.contour_key,
            "stats": {key: {"mean": value[0], "std": value[1]} for key, value in norm_stats.items()},
        },
        "history": history,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an F0-only Transformer tone classifier.")
    parser.add_argument("--features", default="data/aishell3/features/syllable_f0_train100_split.csv")
    parser.add_argument("--contours", default="data/aishell3/features/syllable_f0_train100.npz")
    parser.add_argument("--contour-key", default="f0_hz")
    parser.add_argument("--out", default="runs/f0_transformer_train100/metrics.json")
    parser.add_argument("--checkpoint", default="runs/f0_transformer_train100/best.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    result = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote={out}")
    if args.checkpoint:
        print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
