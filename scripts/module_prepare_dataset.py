"""Merge seed + synth into train/val/test split with BIO label2id.

Rules:
  - test set is 100% real seed hold-out
  - val mixes real + synthetic data
  - train uses synthetic data plus the remaining seed rows
  - validate char offsets before accepting examples
  - skip malformed examples with invalid or overlapping spans

Usage:
  python module_prepare_dataset.py --test-ratio 0.2 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ADDR", "I-ADDR", "B-NOTE", "I-NOTE"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_valid(ex: dict) -> bool:
    text = ex["text"]
    n = len(text)
    sorted_ents = sorted(ex["entities"], key=lambda e: e["start"])
    prev_end = -1
    for ent in sorted_ents:
        s, e = ent["start"], ent["end"]
        if not (0 <= s < e <= n):
            return False
        if not text[s:e]:
            return False
        if ent["label"] not in {"PER", "ADDR", "NOTE"}:
            return False
        if s < prev_end:  # overlap
            return False
        prev_end = e
    return True


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", default="seed.jsonl")
    parser.add_argument("--synth", default="synth_v1.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of seed for test")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of total for val")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)

    seed_data = load_jsonl(Path(args.seed))
    synth_data = load_jsonl(Path(args.synth))

    seed_valid = [ex for ex in seed_data if is_valid(ex)]
    synth_valid = [ex for ex in synth_data if is_valid(ex)]
    seed_drop = len(seed_data) - len(seed_valid)
    synth_drop = len(synth_data) - len(synth_valid)

    random.shuffle(seed_valid)
    random.shuffle(synth_valid)

    # Test uses seed data as the real hold-out set.
    n_test = max(int(len(seed_valid) * args.test_ratio), 1)
    test_set = seed_valid[:n_test]
    seed_remaining = seed_valid[n_test:]

    # Val: 50/50 real + synth
    n_val_each = max(int((len(seed_valid) + len(synth_valid)) * args.val_ratio / 2), 1)
    val_seed = seed_remaining[:n_val_each]
    val_synth = synth_valid[:n_val_each]
    val_set = val_seed + val_synth
    random.shuffle(val_set)

    # Train uses the remaining rows.
    train_set = seed_remaining[n_val_each:] + synth_valid[n_val_each:]
    random.shuffle(train_set)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_set)
    write_jsonl(out_dir / "val.jsonl", val_set)
    write_jsonl(out_dir / "test.jsonl", test_set)

    label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    with open(out_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    stats = {
        "seed_total": len(seed_data),
        "seed_valid": len(seed_valid),
        "seed_dropped": seed_drop,
        "synth_total": len(synth_data),
        "synth_valid": len(synth_valid),
        "synth_dropped": synth_drop,
        "train": len(train_set),
        "val": {"total": len(val_set), "seed": len(val_seed), "synth": len(val_synth)},
        "test": {"total": len(test_set), "seed": len(test_set), "synth": 0},
        "labels": LABEL_LIST,
    }
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
 
