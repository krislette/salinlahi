import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessor import process_csv

if __name__ == "__main__":
    process_csv(
        csv_path="data/external/translation_pairs_tagalog_waray.csv",
        output_dir="data/processed",
        val_split=0.1,
    )
