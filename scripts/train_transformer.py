import json
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Imports mainly for the transformer model
from src.utils.helpers import JSON_DATA, TOKENIZER_MODEL, TRANSFORMER_MODEL
from src.models.transformer.seq2seq import BaselineSeq2SeqTransformer
from src.models.transformer.tokenizer import TranslationDataset, collate_fn, PAD_IDX
from src.models.transformer.helpers import (
    EarlyStopping, 
    get_transformer_scheduler, 
    create_mask
)

def train_transformer(batch_size: int = 512, epochs: int = 100):
    """
    Main training execution loop for the Sequence-to-Sequence Transformer. 
    Handles data splitting, model initialization, AMP mixed-precision training, 
    and early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Transformer Training on {device.type.upper()}...")

    # Data Preparation
    all_data = []
    with open(JSON_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))

    # 90/10 Train-Validation Split
    train_size = int(0.9 * len(all_data))
    val_size = len(all_data) - train_size

    train_raw = all_data[:train_size]
    val_raw = all_data[train_size:]

    # Use secure temporary files that self-clean after training
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as train_tmp, \
         tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as val_tmp:
        
        for item in train_raw: train_tmp.write(json.dumps(item) + '\n')
        for item in val_raw: val_tmp.write(json.dumps(item) + '\n')
        
        train_tmp_path = train_tmp.name
        val_tmp_path = val_tmp.name

    try:
        # Initialize datasets with the SentencePiece BPE model
        train_dataset = TranslationDataset(train_tmp_path, spm_model_path=str(TOKENIZER_MODEL), max_seq_len=128)
        val_dataset = TranslationDataset(val_tmp_path, spm_model_path=str(TOKENIZER_MODEL), max_seq_len=128)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        src_vocab_size = len(train_dataset.src_vocab)
        tgt_vocab_size = len(train_dataset.tgt_vocab)

        # Transformer initialization
        model = BaselineSeq2SeqTransformer(
            num_encoder_layers=6,
            num_decoder_layers=6,
            emb_size=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.3,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size
        ).to(device)

        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

        # Base LR is 1.0 because the Noam scheduler handles the dynamic mathematical scaling
        optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = get_transformer_scheduler(optimizer, d_model=512, warmup_steps=4000)

        # Main training loop
        early_stopper = EarlyStopping(patience=10, save_path=TRANSFORMER_MODEL)
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Starting AMP Training with {len(all_data)} total rows (Train: {train_size}, Val: {val_size})...")
        print(f"  src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:-1, :]
                tgt_expected = tgt[1:, :]

                # Generate all required masking matrices
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
                
                optimizer.zero_grad()

                # Autocast forces Tensor Cores to use 16-bit precision for matrix math
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

                scaler.scale(loss).backward()
                
                # GRADIENT CLIPPING: Unscale first, then clip to 1.0 to prevent gradient explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- VALIDATION PHASE ---
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_input = tgt[:-1, :]
                    tgt_expected = tgt[1:, :]

                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch: {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # --- EARLY STOPPING CHECK ---
            early_stopper(avg_val_loss, model)
            if early_stopper.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

        print("\nTraining complete.")
        
    finally:
        # MEMORY CLEANUP
        # Guarantees the temp files are deleted even if training crashes/interrupts
        if os.path.exists(train_tmp_path): os.remove(train_tmp_path)
        if os.path.exists(val_tmp_path): os.remove(val_tmp_path)
        print("Cleaned up temporary split files.")

if __name__ == "__main__":
    # Choose whichever training you need to do
    # train_recurrent(x, y)
    train_transformer(batch_size=512, epochs=100)