import yaml
from pathlib import Path

# Safely resolve the absolute path to the root of your repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load config
with open(BASE_DIR / "config/data_config.yml", "r") as f:
    config = yaml.safe_load(f)

# Resolve paths explicitly
TRANSFORMER_MODEL = BASE_DIR / config["paths"]["transformer_model"]
TOKENIZER_MODEL = BASE_DIR / config["paths"]["tokenizer_model"]

JSON_DATA = BASE_DIR / config["paths"]["json_data"]
CSV_DATA = BASE_DIR / config["paths"]["csv_data"]

