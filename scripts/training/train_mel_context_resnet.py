#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from scripts.training.tone_dataset import CLASS_TO_TONE, TONE_TO_CLASS  # noqa: E402
from scripts.training.train_f0_transformer import set_seed  # noqa: E402
from scripts.training.train_mel_resnet import ResidualBlock, evaluate_mel, load_logmel_index, resize_time  # noqa: E402


def base_syllable(pinyin: str) -> str:
    return re.sub(r"[1-5]$", "", pinyin)


def parse_bool(value: str) -> float:
    return 1.0 if value.lower() == "true" else 0.0


def build_vocab(rows: list[dict[str, str]], min_count: int = 1) -> dict[str, int]:
    counts = Counter(base_syllable(row["pinyin"]) for row in rows)
    vocab = {"<unk>": 0}
    for token, count in sorted(counts.items()):
        if count >= min_count:
            vocab[token] = len(vocab)
    return vocab


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_vocab(features_csv: Path, vocab_path: Path) -> dict[str, int]:
    if vocab_path.exists():
        return json.loads(vocab_path.read_text(encoding="utf-8"))
    with features_csv.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("data_split") == "train"]
    vocab = build_vocab(rows)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return vocab


class ContextMelDataset(Dataset):
    def __init__(
        self,
        features_csv: str | Path,
        context_features_csv: str | Path,
        logmel_summary: str | Path,
        split: str,
        vocab: dict[str, int],
        frames: int = 96,
        cache_size: int = 128,
        use_syllable_embedding: bool = False,
    ) -> None:
        self.features_csv = Path(features_csv)
        self.context_features_csv = Path(context_features_csv)
        self.logmel_summary = Path(logmel_summary)
        self.split = split
        self.vocab = vocab
        self.frames = frames
        self.cache_size = cache_size
        self.use_syllable_embedding = use_syllable_embedding
        self.logmel_index = load_logmel_index(self.logmel_summary)
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        with self.features_csv.open(newline="", encoding="utf-8") as f:
            self.rows = [row for row in csv.DictReader(f) if row.get("data_split") == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split} in {self.features_csv}")
        with self.context_features_csv.open(newline="", encoding="utf-8") as f:
            context_rows = list(csv.DictReader(f))
        missing = sorted({row["utt_id"] for row in self.rows if row["utt_id"] not in self.logmel_index})
        if missing:
            raise ValueError(f"{len(missing)} utterances are missing log-mel features; first={missing[:5]}")
        target_utts = {row["utt_id"] for row in self.rows}
        self.row_by_position = {
            (row["utt_id"], int(row["syllable_index"])): row
            for row in context_rows
            if row["utt_id"] in target_utts
        }

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

    def segment_for_row(self, row: dict[str, str] | None, fallback_utt: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        if row is None:
            return np.zeros((80, self.frames), dtype=np.float32), 0.0
        utt = self.load_utt(row["utt_id"]) if row["utt_id"] in self.logmel_index else fallback_utt
        logmel = utt["logmel"]
        times = utt["times"]
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = max(0.0, end - start)
        start = max(0.0, start - duration * 0.5)
        end = end + duration * 0.5
        segment = logmel[:, (times >= start) & (times < end)]
        segment = resize_time(segment, self.frames)
        segment = np.clip((segment + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
        return segment, 1.0

    def __getitem__(self, index: int):
        row = self.rows[index]
        current_idx = int(row["syllable_index"])
        utt = self.load_utt(row["utt_id"])
        prev_row = self.row_by_position.get((row["utt_id"], current_idx - 1))
        next_row = self.row_by_position.get((row["utt_id"], current_idx + 1))
        prev_segment, has_prev = self.segment_for_row(prev_row, utt)
        current_segment, _ = self.segment_for_row(row, utt)
        next_segment, has_next = self.segment_for_row(next_row, utt)
        mel_context = np.stack([prev_segment, current_segment, next_segment], axis=0).astype(np.float32)

        syllable_count = max(1, int(row["syllable_count"]))
        relative_index = current_idx / max(1, syllable_count - 1)
        segment_features = np.asarray(
            [
                float(row["duration_sec"]),
                relative_index,
                parse_bool(row["word_boundary_after"]),
                parse_bool(row["phrase_boundary_after"]),
                has_prev,
                has_next,
            ],
            dtype=np.float32,
        )
        syllable_id = self.vocab.get(base_syllable(row["pinyin"]), 0) if self.use_syllable_embedding else 0
        return {
            "mel_context": torch.from_numpy(mel_context).unsqueeze(1),
            "context_mask": torch.tensor([has_prev, 1.0, has_next], dtype=torch.float32),
            "segment_features": torch.from_numpy(segment_features),
            "syllable_id": torch.tensor(syllable_id, dtype=torch.long),
            "label": torch.tensor(TONE_TO_CLASS[row["tone"]], dtype=torch.long),
            "tone": row["tone"],
            "utt_id": row["utt_id"],
            "syllable_index": current_idx,
        }


class MelEncoder(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.1) -> None:
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
        self.dropout = nn.Dropout(dropout)
        self.out_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.dropout(x)


class MelContextResNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        width: int = 32,
        syllable_embed_dim: int = 32,
        segment_feature_dim: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 5,
        use_syllable_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MelEncoder(width=width, dropout=dropout)
        self.use_syllable_embedding = use_syllable_embedding
        self.syllable_embedding = nn.Embedding(vocab_size, syllable_embed_dim) if use_syllable_embedding else None
        input_dim = self.encoder.out_dim * 3 + segment_feature_dim + (syllable_embed_dim if use_syllable_embedding else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, mel_context: torch.Tensor, segment_features: torch.Tensor, syllable_id: torch.Tensor) -> torch.Tensor:
        batch, slots, channels, mels, frames = mel_context.shape
        encoded = self.encoder(mel_context.reshape(batch * slots, channels, mels, frames))
        encoded = encoded.reshape(batch, slots, -1).reshape(batch, -1)
        parts = [encoded, segment_features]
        if self.use_syllable_embedding and self.syllable_embedding is not None:
            parts.append(self.syllable_embedding(syllable_id))
        fused = torch.cat(parts, dim=-1)
        return self.classifier(fused)


def evaluate_context(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    confusion = np.zeros((5, 5), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            mel_context = batch["mel_context"].to(device)
            segment_features = batch["segment_features"].to(device)
            syllable_id = batch["syllable_id"].to(device)
            labels = batch["label"].to(device)
            logits = model(mel_context, segment_features, syllable_id)
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
    features = Path(args.features)
    vocab_path = Path(args.vocab)
    vocab = load_or_create_vocab(features, vocab_path)
    logmel_meta_path = Path(args.logmel_summary).with_suffix(Path(args.logmel_summary).suffix + ".meta.json")
    logmel_meta = json.loads(logmel_meta_path.read_text(encoding="utf-8")) if logmel_meta_path.exists() else {}
    context_features = Path(args.context_features)
    train_ds = ContextMelDataset(
        features,
        context_features,
        args.logmel_summary,
        "train",
        vocab,
        frames=args.frames,
        cache_size=args.cache_size,
        use_syllable_embedding=args.use_syllable_embedding,
    )
    val_ds = ContextMelDataset(
        features,
        context_features,
        args.logmel_summary,
        "val",
        vocab,
        frames=args.frames,
        cache_size=args.cache_size,
        use_syllable_embedding=args.use_syllable_embedding,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = MelContextResNet(
        vocab_size=len(vocab),
        width=args.width,
        dropout=args.dropout,
        use_syllable_embedding=args.use_syllable_embedding,
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
            mel_context = batch["mel_context"].to(device)
            segment_features = batch["segment_features"].to(device)
            syllable_id = batch["syllable_id"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mel_context, segment_features, syllable_id)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            preds = logits.argmax(dim=1)
            train_total += labels.numel()
            train_correct += int((preds == labels).sum().item())
            train_loss_sum += float(loss.item()) * labels.numel()

        val_metrics = evaluate_context(model, val_loader, device)
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
                            "type": "end_to_end_style_mel_C1_segment_features",
                            "frames": args.frames,
                            "n_mels": 80,
                            "context_slots": ["prev", "current", "next"],
                            "features_csv": args.features,
                            "features_sha256": sha256_file(features),
                            "context_features_csv": args.context_features,
                            "context_features_sha256": sha256_file(context_features),
                            "logmel_summary": args.logmel_summary,
                            "logmel_summary_sha256": sha256_file(Path(args.logmel_summary)),
                            "logmel_meta": logmel_meta,
                            "vocab": str(vocab_path),
                            "vocab_sha256": sha256_file(vocab_path),
                            "vocab_size": len(vocab),
                            "use_syllable_embedding": args.use_syllable_embedding,
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
            "type": "end_to_end_style_mel_C1_segment_features",
            "frames": args.frames,
            "n_mels": 80,
            "context_slots": ["prev", "current", "next"],
            "features_csv": args.features,
            "features_sha256": sha256_file(features),
            "context_features_csv": args.context_features,
            "context_features_sha256": sha256_file(context_features),
            "logmel_summary": args.logmel_summary,
            "logmel_summary_sha256": sha256_file(Path(args.logmel_summary)),
            "logmel_meta": logmel_meta,
            "vocab": str(vocab_path),
            "vocab_sha256": sha256_file(vocab_path),
            "vocab_size": len(vocab),
            "use_syllable_embedding": args.use_syllable_embedding,
        },
        "history": history,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an End-to-End-paper-style C1 mel ResNet with segment features.")
    parser.add_argument("--features", default="data/aishell3/features/syllable_f0_train_full_split.csv")
    parser.add_argument("--context-features", default="data/aishell3/features/syllable_f0_train_full.csv")
    parser.add_argument("--logmel-summary", default="data/aishell3/features/logmel_utterance_train_full.csv")
    parser.add_argument("--vocab", default="data/aishell3/features/base_syllable_vocab.json")
    parser.add_argument("--out", default="runs/mel_context_resnet_train_full/metrics.json")
    parser.add_argument("--checkpoint", default="runs/mel_context_resnet_train_full/best.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--use-syllable-embedding", action="store_true")
    parser.add_argument("--frames", type=int, default=96)
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
