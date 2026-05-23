<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">Salinlahi</h1>
  <p align="center">
    Lightweight Multilingual Neural Machine Translation for Related Low-Resource Philippine Languages
</div>

---

This repository houses the **Data and ML pipeline** for Salinlahi, exposed as a REST API for the Full-Stack team to consume. It covers everything from raw dataset collection to trained model weights and inference endpoints.

## Table of Contents

- [Who Does What](#who-does-what)
- [Folder Structure](#folder-structure)
- [Folder Responsibilities by Team](#folder-responsibilities-by-team)
- [Getting Started](#getting-started)
- [Workflow Overview](#workflow-overview)
- [API Endpoints](#api-endpoints)
- [Conventions](#conventions)

---

## Who Does What

| Team                | Members         | Owns                          |
| ------------------- | --------------- | ----------------------------- |
| **Data**            | Arwen, Kristine | `src/data/`, `data/`          |
| **ML: Recurrent**   | Acelle, Regina  | `src/models/recurrent/`       |
| **ML: Transformer** | Hans, Fervicmar | `src/models/transformer/`     |
| **Full-Stack**      | Kyla, Kyros     | Separate repo, calls this API |

---

## Folder Structure

```
salinlahi/
│
├── .github/
│   └── workflows/              # CI/CD, automated lint and tests on push
│
├── app/                        # FastAPI layer (thin, only wiring, no logic here)
│   ├── server.py               # Route definitions
│   ├── schemas.py              # Pydantic request/response models
│   ├── utils.py                # Server-side helpers
│   └── validators.py           # Input validation
│
├── config/                     # All hyperparameters and settings live here (no hardcoding)
│   ├── data_config.yml         # Dataset paths, language pairs, split ratios
│   ├── model_config.yml        # Learning rate, hidden size, layers, epochs, etc.
│   └── server_config.yml       # CORS, port, rate limits
│
├── data/
│   ├── raw/                    # Original, untouched source files
│   │   ├── waray/
│   │   ├── hiligaynon/
│   │   └── kapampangan/
│   ├── processed/              # Cleaned CSVs and stats, Data team's final deliverables
│   │   ├── translation_pairs.csv
│   │   ├── alpaca_waray_clean.csv
│   │   └── dataset_stats.json
│   └── external/               # Benchmark data (OPUS, biblical texts, SEACrowd)
│
├── models/                     # Saved model weights (gitignored, use Git LFS or HuggingFace)
│   ├── recurrent/              # RNN/GRU/BiLSTM weights, Acelle & Regina
│   └── transformer/            # Transformer weights, Hans & Fervicmar
│
├── results/                    # BLEU scores, F-scores, speed benchmarks, comparison tables
│
├── scripts/                    # Entry-point CLI scripts (call into src/)
│   ├── train.py                # Run model training
│   ├── evaluate.py             # Evaluate on test split
│   └── predict.py              # One-off CLI inference
│
├── src/                        # All core logic lives here, modular and importable
│   │
│   ├── data/                   # Data team
│   │   ├── collector.py        # Dataset downloading and sourcing
│   │   ├── cleaner.py          # Cleaning scripts
│   │   └── tokenizer.py        # Inference-ready tokenizer
│   │
│   ├── models/                 # ML team
│   │   ├── recurrent/          # Acelle & Regina
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   ├── seq2seq.py      # Assembles encoder + decoder + attention
│   │   │   └── trainer.py
│   │   └── transformer/        # Hans & Fervicmar
│   │       ├── model.py        # Scaled-down transformer definition
│   │       └── trainer.py
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py     # Noise removal, filtering, subword tokenization
│   │
│   ├── evaluation/
│   │   └── metrics.py          # BLEU, F-score, inference speed, VRAM tracking
│   │
│   └── utils/
│       └── helpers.py          # Shared utility functions
│
├── .gitignore
└── README.md
```

## Folder Responsibilities by Team

### Data Team, Arwen & Kristine

Your primary workspace is `src/data/` and `data/`.

- Download and organize raw sources into `data/raw/<language>/`
- Write cleaning logic in `src/data/cleaner.py`
- Port the tokenizer from the old repo (`tokenizer.py`) into `src/data/tokenizer.py`
- Output the final cleaned files into `data/processed/`, **this is the handoff point for the ML team**
- Document dataset stats in `data/processed/dataset_stats.json`

> **Rule:** The ML team reads only from `data/processed/`. Nothing in `data/raw/` should be consumed directly by training scripts.

---

### ML Team, Recurrent (Acelle & Regina)

Your primary workspace is `src/models/recurrent/`.

- `encoder.py`, RNN/GRU/BiLSTM encoder
- `decoder.py`, decoder with Bahdanau attention
- `seq2seq.py`, assembles the full model
- `trainer.py`, training loop, checkpointing, logging

Save model weights to `models/recurrent/` and log training runs to `logs/`.

---

### ML Team, Transformer (Hans & Fervicmar)

Your primary workspace is `src/models/transformer/`.

- `model.py`, scaled-down transformer definition
- `trainer.py`, training loop and checkpointing

This serves as the **baseline for comparison** against the recurrent models. Evaluation metrics go into `results/`.

---

### Both ML Sub-teams

- Use `config/model_config.yml` for all hyperparameters, no magic numbers in code
- Use `src/preprocessing/preprocessor.py` for the shared preprocessing pipeline
- Use `src/evaluation/metrics.py` for BLEU and F-score, keep metrics consistent across models

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<org>/salinlahi.git
cd salinlahi

# 2. Install dependencies
pip install -r requirements.txt

# 4. Run the API server locally
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`.
Interactive docs are available at `http://localhost:8000/docs`.

---

## Workflow Overview

```
Data Team                ML Team                  Full-Stack Team
─────────────────────    ─────────────────────    ─────────────────
collect raw data   ->    read processed/ CSVs ->  call /translate
clean + tokenize   ->    train recurrent model    call /languages
write to processed/->    train transformer
                         evaluate + compare
                         save weights to models/
                         expose via app/server.py
```

---

## API Endpoints

> TENTATIVE: Full-Stack team: these are the endpoints your frontend will call.

| Method | Endpoint             | Description                                  |
| ------ | -------------------- | -------------------------------------------- |
| `POST` | `/api/v1/translate`  | Translate Filipino text to a target language |
| `GET`  | `/api/v1/languages`  | List supported language pairs                |
| `GET`  | `/api/v1/model/info` | Model metadata and architecture info         |
| `GET`  | `/`                  | Health check and endpoint listing            |

## Conventions

- **Branch naming:** `feature/<your-name>-<short-description>` (e.g., `feature/arwen-data-cleaning`)
- **No hardcoded values**, use `config/*.yml` for all paths, hyperparameters, and settings
- **No logic in `app/`**, the API layer only imports from `src/`
- **`data/raw/` is read-only**, never modify raw files; always write cleaned output to `data/processed/`
- **Model weights go in `models/`**, this folder is gitignored; upload to HuggingFace and link in `docs/`

---

<p align="right">[<a href="#readme-top">Back to top</a>]</p>
