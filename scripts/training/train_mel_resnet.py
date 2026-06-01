#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from collections import Counter
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.training.tone_dataset import CLASS_TO_TONE, TONE_TO_CLASS  # noqa: E402
from scripts.training.train_f0_transformer import evaluate, set_seed  # noqa: E402


def load_logmel_index(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["utt_id"]: row["logmel_npz_path"] for row in csv.DictReader(f)}


def resize_time(segment: np.ndarray, frames: int) -> np.ndarray:
    if segment.shape[1] == frames:
        return segment
    if segment.shape[1] == 0:
        return np.zeros((segment.shape[0], frames), dtype=np.float32)
    source = np.linspace(0.0, 1.0, num=segment.shape[1], dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=frames, dtype=np.float32)
    out = np.empty((segment.shape[0], frames), dtype=np.float32)
    for idx in range(segment.shape[0]):
        out[idx] = np.interp(target, source, segment[idx]).astype(np.float32)
    return out


class TriToneMelDataset(Dataset):
    def __init__(
        self,
        features_csv: str | Path,
        logmel_summary: str | Path,
        split: str,
        frames: int = 96,
        use_tritone: bool = True,
        cache_size: int = 128,
    ) -> None:
        self.features_csv = Path(features_csv)
        self.logmel_summary = Path(logmel_summary)
        self.split = split
        self.frames = frames
        self.use_tritone = use_tritone
        self.cache_size = cache_size
        self.logmel_index = load_logmel_index(self.logmel_summary)
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        with self.features_csv.open(newline="", encoding="utf-8") as f:
            self.rows = [row for row in csv.DictReader(f) if row.get("data_split") == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split} in {self.features_csv}")

        missing = sorted({row["utt_id"] for row in self.rows if row["utt_id"] not in self.logmel_index})
        if missing:
            raise ValueError(f"{len(missing)} utterances are missing log-mel features; first={missing[:5]}")

    def __len__(self) -> int:
        return len(self.rows)

    def load_utt(self, utt_id: str) -> dict[str, np.ndarray]:
        if utt_id not in self.cache:
            with np.load(self.logmel_index[utt_id]) as data:
                self.cache[utt_id] = {key: data[key] for key in data.files}
            if self.cache_size > 0:
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(utt_id)
        return self.cache[utt_id]

    def __getitem__(self, index: int):
        row = self.rows[index]
        utt = self.load_utt(row["utt_id"])
        logmel = utt["logmel"]
        times = utt["times"]
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = max(0.0, end - start)
        if self.use_tritone:
            start = max(0.0, start - duration * 0.5)
            end = end + duration * 0.5

        mask = (times >= start) & (times < end)
        segment = logmel[:, mask]
        segment = resize_time(segment, self.frames)
        segment = (segment + 80.0) / 80.0
        segment = np.clip(segment, 0.0, 1.0).astype(np.float32)
        label = TONE_TO_CLASS[row["tone"]]
        return {
            "mel": torch.from_numpy(segment).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "tone": row["tone"],
            "utt_id": row["utt_id"],
            "syllable_index": int(row["syllable_index"]),
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class SmallMelResNet(nn.Module):
    def __init__(self, num_classes: int = 5, width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        x = self.body(x)
        x = self.pool(x)
        return self.classifier(x)


def evaluate_mel(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((5, 5), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            logits = model(mel)
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
        per_tone[tone] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[idx, :].sum())}

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
    logmel_meta_path = Path(args.logmel_summary).with_suffix(Path(args.logmel_summary).suffix + ".meta.json")
    logmel_meta = json.loads(logmel_meta_path.read_text(encoding="utf-8")) if logmel_meta_path.exists() else {}
    train_ds = TriToneMelDataset(
        args.features,
        args.logmel_summary,
        "train",
        frames=args.frames,
        use_tritone=not args.current_only,
        cache_size=args.cache_size,
    )
    val_ds = TriToneMelDataset(
        args.features,
        args.logmel_summary,
        "val",
        frames=args.frames,
        use_tritone=not args.current_only,
        cache_size=args.cache_size,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SmallMelResNet(width=args.width, dropout=args.dropout).to(device)
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
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            preds = logits.argmax(dim=1)
            train_total += labels.numel()
            train_correct += int((preds == labels).sum().item())
            train_loss_sum += float(loss.item()) * labels.numel()

        val_metrics = evaluate_mel(model, val_loader, device)
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
                        "input": {
                            "type": "tri-tone_logmel" if not args.current_only else "current_syllable_logmel",
                            "frames": args.frames,
                            "n_mels": 80,
                            "features_csv": args.features,
                            "logmel_summary": args.logmel_summary,
                            "logmel_meta": logmel_meta,
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
        "input": {
            "type": "tri-tone_logmel" if not args.current_only else "current_syllable_logmel",
            "frames": args.frames,
            "n_mels": 80,
            "features_csv": args.features,
            "logmel_summary": args.logmel_summary,
            "logmel_meta": logmel_meta,
        },
        "history": history,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet-style classifier on syllable/tri-tone log-mel segments.")
    parser.add_argument("--features", default="data/aishell3/features/syllable_f0_train_full_split.csv")
    parser.add_argument("--logmel-summary", default="data/aishell3/features/logmel_utterance_train_full.csv")
    parser.add_argument("--out", default="runs/mel_resnet_train_full/metrics.json")
    parser.add_argument("--checkpoint", default="runs/mel_resnet_train_full/best.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
