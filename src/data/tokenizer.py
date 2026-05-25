"""
tokenizer.py — Philippine Language Dataset Tokenizer for Inference
==================================================================
Supports:
  - translation_pairs.csv  (Tagalog ↔ Waray / Hiligaynon / Kapampangan)
  - alpaca_waray_clean.csv (Waray instruction-tuning, Alpaca format)

Usage
-----
  # Tokenize all translation pairs (train split only):
  python tokenizer.py --dataset translation_pairs.csv --mode translation

  # Tokenize a single sentence:
  python tokenizer.py --text "Magandang umaga po" --mode translation

  # Tokenize alpaca dataset:
  python tokenizer.py --dataset alpaca_waray_clean.csv --mode alpaca

  # Save tokenized output to JSONL:
  python tokenizer.py --dataset translation_pairs.csv --mode translation --output out.jsonl

Requirements: pip install transformers pandas torch
"""

import re
import json
import argparse
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Whitespace/rule-based tokenizer (no model needed — fast, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

class PhilippineTokenizer:
    """
    Lightweight rule-based tokenizer tuned for Filipino/Philippine languages.

    Handles:
      - Unicode-aware word splitting
      - Tagalog/Waray/Hiligaynon/Kapampangan morphological affixes (surface-level)
      - Punctuation splitting
      - Lowercasing + normalization
    """

    # Common affixes across PH languages — used for subword hint generation
    PREFIXES = ['nag', 'mag', 'ma', 'na', 'pa', 'ka', 'ipa', 'pag',
                'ipag', 'maka', 'makapag', 'han', 'gin', 'in', 'i']
    SUFFIXES = ['an', 'in', 'han', 'on', 'ng', 'ngan', 'non']

    def __init__(self, lowercase: bool = True, split_punctuation: bool = True):
        self.lowercase = lowercase
        self.split_punctuation = split_punctuation
        # Matches Unicode word characters (covers Filipino diacritics)
        self._word_re = re.compile(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+|[^\s\w]", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        """Return a flat list of string tokens."""
        if not isinstance(text, str) or not text.strip():
            return []
        if self.lowercase:
            text = text.lower()
        tokens = self._word_re.findall(text)
        if not self.split_punctuation:
            tokens = [t for t in tokens if re.match(r'\w', t, re.UNICODE)]
        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Map tokens to integer IDs via a simple vocabulary built on the fly.
        For inference with a pretrained model, use HuggingFaceTokenizerWrapper below.
        """
        tokens = self.tokenize(text)
        return [hash(t) & 0xFFFF for t in tokens]   # 16-bit fingerprint

    def batch_tokenize(self, texts: list[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]

    def get_affixes(self, token: str) -> dict:
        """Surface-level affix detection (informational, not used in base tokenize)."""
        token = token.lower()
        found_pre  = [p for p in self.PREFIXES if token.startswith(p)]
        found_suf  = [s for s in self.SUFFIXES if token.endswith(s) and len(token) > len(s) + 2]
        return {'prefix': found_pre, 'suffix': found_suf}


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace wrapper (drop-in for inference with a trained model)
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceTokenizerWrapper:
    """
    Wraps any HuggingFace tokenizer for use with the datasets here.

    Recommended models:
      - jcblaise/roberta-tagalog-base  (general Tagalog)
      - danjohnvelasco/bert-tagalog    (Tagalog BERT)
      - (fine-tune your own on translation_pairs.csv for Waray/Hiligaynon/Kapampangan)

    Usage:
        wrapper = HuggingFaceTokenizerWrapper("jcblaise/roberta-tagalog-base")
        result  = wrapper.encode_translation_pair("Magandang umaga", "Maupay nga aga")
    """

    def __init__(self, model_name: str, max_length: int = 128):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model_name = model_name

    def encode_translation_pair(
        self, source_text: str, target_text: Optional[str] = None
    ) -> dict:
        """
        Encode a source (and optionally target) text for seq2seq or classification.
        Returns a dict with input_ids, attention_mask, token_type_ids (if present).
        """
        enc = self.tokenizer(
            source_text,
            text_pair=target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {k: v.squeeze(0).tolist() for k, v in enc.items()}

    def encode_alpaca(self, instruction: str, input_ctx: str, output: str) -> dict:
        """
        Format an Alpaca record into the standard prompt template and tokenize.
        Returns source_ids (prompt) and target_ids (response).
        """
        if input_ctx.strip():
            prompt = (
                f"Instruction: {instruction}\n"
                f"Input: {input_ctx}\n"
                "Response:"
            )
        else:
            prompt = (
                f"Instruction: {instruction}\n"
                "Response:"
            )
        src = self.tokenizer(prompt, max_length=self.max_length,
                             padding='max_length', truncation=True, return_tensors='pt')
        tgt = self.tokenizer(output, max_length=self.max_length,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'source_ids':       src['input_ids'].squeeze(0).tolist(),
            'source_mask':      src['attention_mask'].squeeze(0).tolist(),
            'target_ids':       tgt['input_ids'].squeeze(0).tolist(),
            'target_mask':      tgt['attention_mask'].squeeze(0).tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_translation_pairs(
    path: str,
    language_pair: Optional[str] = None,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load translation_pairs.csv with optional filters.

    Args:
        path:          Path to translation_pairs.csv
        language_pair: e.g. 'Tagalog-Waray', 'Tagalog-Hiligaynon', 'Tagalog-Kapampangan'
        split:         'train', 'test', or 'vocab'
    """
    df = pd.read_csv(path)
    if language_pair:
        df = df[df['language_pair'] == language_pair]
    if split:
        df = df[df['split'] == split]
    df = df.dropna(subset=['source_text', 'target_text'])
    df = df[df['source_text'].str.strip().ne('') & df['target_text'].str.strip().ne('')]
    return df.reset_index(drop=True)


def load_alpaca(path: str, has_input: Optional[bool] = None) -> pd.DataFrame:
    """
    Load alpaca_waray_clean.csv.

    Args:
        path:      Path to alpaca_waray_clean.csv
        has_input: True = only records with context input, False = instruction-only
    """
    df = pd.read_csv(path)
    if has_input is not None:
        df = df[df['has_input'] == has_input]
    df = df.dropna(subset=['instruction', 'output'])
    df['input'] = df['input'].fillna('')
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Batch tokenization helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_translation_dataset(
    df: pd.DataFrame,
    tokenizer: PhilippineTokenizer,
) -> list[dict]:
    """
    Tokenize all rows in a translation pairs DataFrame.
    Returns a list of dicts ready for JSON/JSONL export.
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            'language_pair':   row['language_pair'],
            'split':           row['split'],
            'source_lang':     row['source_lang'],
            'target_lang':     row['target_lang'],
            'source_text':     row['source_text'],
            'target_text':     row['target_text'],
            'source_tokens':   tokenizer.tokenize(row['source_text']),
            'target_tokens':   tokenizer.tokenize(row['target_text']),
        })
    return records


def tokenize_alpaca_dataset(
    df: pd.DataFrame,
    tokenizer: PhilippineTokenizer,
) -> list[dict]:
    """
    Tokenize all rows in an Alpaca DataFrame.
    Returns a list of dicts ready for JSON/JSONL export.
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            'language':            row['language'],
            'instruction':         row['instruction'],
            'input':               row['input'],
            'output':              row['output'],
            'has_input':           bool(row['has_input']),
            'instruction_tokens':  tokenizer.tokenize(row['instruction']),
            'input_tokens':        tokenizer.tokenize(row['input']) if row['input'] else [],
            'output_tokens':       tokenizer.tokenize(row['output']),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tokenize Philippine language datasets for ML inference.'
    )
    parser.add_argument('--dataset',  type=str, help='Path to CSV dataset file')
    parser.add_argument('--mode',     type=str, choices=['translation', 'alpaca'],
                        default='translation', help='Dataset type')
    parser.add_argument('--text',     type=str, help='Tokenize a single string directly')
    parser.add_argument('--pair',     type=str, help='Filter by language pair (translation mode)')
    parser.add_argument('--split',    type=str, help='Filter by split: train/test/vocab')
    parser.add_argument('--output',   type=str, help='Save tokenized output to JSONL file')
    parser.add_argument('--hf-model', type=str, help='HuggingFace model name (optional)')
    parser.add_argument('--no-lower', action='store_true', help='Disable lowercasing')
    args = parser.parse_args()

    tokenizer = PhilippineTokenizer(lowercase=not args.no_lower)

    # ── Single text mode ──────────────────────────────────────────────────
    if args.text:
        tokens = tokenizer.tokenize(args.text)
        print(f'Input : {args.text}')
        print(f'Tokens: {tokens}')
        print(f'Count : {len(tokens)}')
        return

    if not args.dataset:
        parser.error('Provide --dataset or --text')

    # ── Dataset mode ──────────────────────────────────────────────────────
    if args.mode == 'translation':
        df = load_translation_pairs(args.dataset, language_pair=args.pair, split=args.split)
        print(f'Loaded {len(df)} translation pairs')
        records = tokenize_translation_dataset(df, tokenizer)

    elif args.mode == 'alpaca':
        df = load_alpaca(args.dataset)
        print(f'Loaded {len(df)} Alpaca records')
        records = tokenize_alpaca_dataset(df, tokenizer)

    # ── HuggingFace encoding (optional) ───────────────────────────────────
    if args.hf_model:
        print(f'Encoding with HuggingFace model: {args.hf_model}')
        hf = HuggingFaceTokenizerWrapper(args.hf_model)
        for rec in records:
            if args.mode == 'translation':
                enc = hf.encode_translation_pair(rec['source_text'], rec['target_text'])
            else:
                enc = hf.encode_alpaca(rec['instruction'], rec['input'], rec['output'])
            rec['hf_encoding'] = enc

    # ── Print sample ──────────────────────────────────────────────────────
    print('\n── Sample output (first record) ──')
    sample = records[0]
    for k, v in sample.items():
        display = v if not isinstance(v, list) else f'{v[:8]} … ({len(v)} tokens)'
        print(f'  {k}: {display}')

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'\nSaved {len(records)} records → {args.output}')


if __name__ == '__main__':
    main()
