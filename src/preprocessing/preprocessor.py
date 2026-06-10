import csv
import json
import random
from pathlib import Path

LANGUAGE_PAIR_DIRS = {
    "Tagalog-Waray": "fil_war",
    "Tagalog-Hiligaynon": "fil_hil",
    "Tagalog-Kapampangan": "fil_pam",
}


def prepare_data_from_csv(
    csv_path: str,
    output_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Reads translation_pairs.csv and writes per-language-pair JSONL files
    to output_dir/<lang_dir>/{train,val,test}.jsonl.
    Skips writing if files already exist.
    """
    random.seed(seed)
    output_dir = Path(output_dir)

    buckets: dict[str, dict[str, list]] = {
        lp: {"train": [], "test": []} for lp in LANGUAGE_PAIR_DIRS
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lp = row["language_pair"]
            if lp not in buckets:
                continue
            record = {
                "source_tokens": row["source_text"].lower().strip().split(),
                "target_tokens": row["target_text"].lower().strip().split(),
            }
            split = "train" if row["split"] == "vocab" else row["split"]
            if split in ("train", "test"):
                buckets[lp][split].append(record)

    for lp, lang_dir in LANGUAGE_PAIR_DIRS.items():
        pair_dir = output_dir / lang_dir
        if all((pair_dir / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
            print(f"{lang_dir}: already processed, skipping.")
            continue

        pair_dir.mkdir(parents=True, exist_ok=True)
        all_train = buckets[lp]["train"]
        random.shuffle(all_train)

        val_size = max(1, int(len(all_train) * val_split))
        splits = {
            "train": all_train[val_size:],
            "val": all_train[:val_size],
            "test": buckets[lp]["test"],
        }
        for split_name, rows in splits.items():
            out_path = pair_dir / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{lang_dir} | {split_name}: {len(rows)} rows → {out_path}")
