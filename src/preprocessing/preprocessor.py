import csv
import json
import random
from pathlib import Path

DATASETS = {
    "fil_war": "translation_pairs_fil_war.csv",
    "war_fil": "translation_pairs_war_fil.csv",
}


def prepare_data_from_csv(
    csv_dir: str,
    output_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Reads each CSV in DATASETS and writes per-direction JSONL files
    to output_dir/<lang_dir>/{train,val,test}.jsonl.
    Skips if files already exist.
    """
    random.seed(seed)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    for lang_dir, csv_filename in DATASETS.items():
        pair_dir = output_dir / lang_dir
        if all((pair_dir / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
            print(f"{lang_dir}: already processed, skipping.")
            continue

        pair_dir.mkdir(parents=True, exist_ok=True)
        train_rows, test_rows = [], []

        with open(csv_dir / csv_filename, "r", encoding="utf-8") as f:
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
            print(f"{lang_dir} | {split_name}: {len(rows)} rows -> {out_path}")
