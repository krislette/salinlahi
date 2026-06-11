import csv
import json
import random
from pathlib import Path


def prepare_data_from_csv(
    csv_path: str,
    output_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    random.seed(seed)
    output_dir = Path(output_dir)
    pair_dir = output_dir / "fil_war"

    if all((pair_dir / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
        print("fil_war: already processed, skipping.")
        return

    pair_dir.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            record = {
                "source_tokens": row["source_text"].lower().strip().split(),
                "target_tokens": row["target_text"].lower().strip().split(),
            }
            if row["split"] == "test":
                test_rows.append(record)
            else:
                train_rows.append(record)

    random.shuffle(train_rows)
    val_size = max(1, int(len(train_rows) * val_split))
    splits = {
        "train": train_rows[val_size:],
        "val": train_rows[:val_size],
        "test": test_rows,
    }
    for split_name, rows in splits.items():
        out_path = pair_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"fil_war | {split_name}: {len(rows)} rows -> {out_path}")
