import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from src.utils.helpers import TRANSFORMER_MODEL
from models.transformer.tokenizer import PAD_IDX

class EarlyStopping:
    """
    Monitors validation loss and stops training early if the model 
    stops improving, saving the best weights to disk.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001, save_path=TRANSFORMER_MODEL):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss: float, model: nn.Module):
        # Check if the new loss is at least 'min_delta' better than the best loss
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"   -> Validation loss improved. Saved best weights to {self.save_path}")
        else:
            self.counter += 1
            print(f"   -> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def get_transformer_scheduler(optimizer: Optimizer, d_model: int = 512, warmup_steps: int = 4000) -> LambdaLR:
    """
    Wraps the custom "Attention is All You Need" learning rate scheduling math 
    into a native PyTorch LambdaLR.
    """
    def lr_lambda(step: int) -> float:
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    return LambdaLR(optimizer, lr_lambda)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on the diagonal.
    Crucial for the Decoder to prevent it from "looking ahead" at future words.
    """
    # PyTorch's native implementation is highly optimized for C++ execution
    return nn.Transformer.generate_square_subsequent_mask(sz)


def create_mask(src: torch.Tensor, tgt: torch.Tensor, device: torch.device):
    """
    Generates all four required masks for the PyTorch Transformer:
    source mask, target look-ahead mask, and padding masks for both.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Target look-ahead mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    # Empty source mask (Encoder can see everything)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device, dtype=torch.bool)

    # Padding masks (True where the token is PAD_IDX)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1).type(torch.bool)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).type(torch.bool)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask