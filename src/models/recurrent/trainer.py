import json
import math
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.recurrent.seq2seq import Seq2Seq

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the full training lifecycle for the Salinlahi recurrent seq2seq model.

    Responsibilities:
      - Running the training loop with teacher forcing
      - Evaluating on the validation set each epoch
      - Saving the best model checkpoint to models/recurrent/
      - Logging training progress to the console and a JSON log file under logs/

    Teacher forcing ratio is linearly annealed from its starting value down to a
    minimum floor across epochs, so the model gradually learns to rely on itself
    rather than the gold target tokens.

    Usage:
        trainer = Trainer(model, optimizer, criterion, config, device)
        trainer.train(train_loader, val_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: Seq2Seq,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device

        training_cfg = config["training"]
        self.teacher_forcing_start = training_cfg["teacher_forcing_ratio"]
        self.teacher_forcing_min = training_cfg.get("teacher_forcing_min", 0.1)
        self.gradient_clip = training_cfg.get("gradient_clip", 1.0)

        # Reduce learning rate when validation loss stops improving
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_cfg.get("lr_decay_factor", 0.5),
            patience=training_cfg.get("lr_patience", 2),
        )

        # Where to save the best weights
        self.checkpoint_dir = Path(config["paths"]["model_checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Where to write the training log
        log_dir = Path(config["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = log_dir / "training_log.json"

        self.best_validation_loss = float("inf")
        self.training_history = []

    def compute_teacher_forcing_ratio(
        self, current_epoch: int, total_epochs: int
    ) -> float:
        # Linearly reduce the teacher forcing ratio from start to min over training
        # Early epochs benefit from guidance; later epochs build independence
        decay = (self.teacher_forcing_start - self.teacher_forcing_min) * (
            current_epoch / total_epochs
        )
        return max(self.teacher_forcing_start - decay, self.teacher_forcing_min)

    def run_training_epoch(
        self,
        train_loader: DataLoader,
        teacher_forcing_ratio: float,
    ) -> float:
        """
        Runs one full pass over the training data.
        Returns the average cross-entropy loss over all batches.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            source_tokens = batch["source_tokens"].to(self.device)
            source_lengths = batch["source_lengths"].to(self.device)
            target_tokens = batch["target_tokens"].to(self.device)
            # source_tokens:  (batch_size, source_len)
            # target_tokens:  (batch_size, target_len) — includes <sos> ... <eos>

            self.optimizer.zero_grad()

            # Forward pass through the full seq2seq model
            all_logits = self.model(
                source_tokens,
                source_lengths,
                target_tokens,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            # all_logits: (batch_size, target_len, vocab_size)

            # Reshape for the loss function:
            #   logits shape expected by CrossEntropyLoss: (N, vocab_size)
            #   targets shape expected:                    (N,)
            # We skip position 0 (the <sos> token) because there's no prediction before it
            logits_flat = all_logits[:, 1:].reshape(-1, all_logits.shape[2])
            targets_flat = target_tokens[:, 1:].reshape(-1)
            # The criterion ignores pad positions automatically via ignore_index

            loss = self.criterion(logits_flat, targets_flat)
            loss.backward()

            # Clip gradients to prevent exploding gradients, which are common in RNNs
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def run_validation_epoch(self, val_loader: DataLoader) -> float:
        """
        Runs one full pass over the validation data without updating parameters.
        Returns the average cross-entropy loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                source_tokens = batch["source_tokens"].to(self.device)
                source_lengths = batch["source_lengths"].to(self.device)
                target_tokens = batch["target_tokens"].to(self.device)

                # During validation, teacher forcing is off (ratio = 0.0)
                # so the model's true generalization is measured
                all_logits = self.model(
                    source_tokens,
                    source_lengths,
                    target_tokens,
                    teacher_forcing_ratio=0.0,
                )

                logits_flat = all_logits[:, 1:].reshape(-1, all_logits.shape[2])
                targets_flat = target_tokens[:, 1:].reshape(-1)
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_checkpoint(
        self, epoch: int, validation_loss: float, filename: str = "best_model.pt"
    ) -> None:
        """
        Saves a full model checkpoint to models/recurrent/.

        The checkpoint includes the model weights, optimizer state, and enough
        metadata to resume training or reproduce results later.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "validation_loss": validation_loss,
            "config": self.config,
        }
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, filename: str = "best_model.pt") -> int:
        """
        Loads a saved checkpoint and restores model and optimizer state.
        Returns the epoch the checkpoint was saved at.
        """
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_validation_loss = checkpoint["validation_loss"]

        logger.info(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def write_log(self) -> None:
        # Persist the training history to a JSON file after every epoch
        # so progress isn't lost if the run is interrupted
        with open(self.log_file_path, "w") as log_file:
            json.dump(self.training_history, log_file, indent=2)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        """
        Runs the full training loop for the given number of epochs.

        After each epoch:
          - Logs train and validation loss (and perplexity)
          - Saves a checkpoint if validation loss improved
          - Steps the learning rate scheduler

        Args:
            train_loader: DataLoader yielding batches from data/processed/
            val_loader:   DataLoader for the validation split
            num_epochs:   total number of training epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            teacher_forcing_ratio = self.compute_teacher_forcing_ratio(
                epoch, num_epochs
            )

            train_loss = self.run_training_epoch(train_loader, teacher_forcing_ratio)
            val_loss = self.run_validation_epoch(val_loader)

            self.scheduler.step(val_loss)

            elapsed = time.time() - epoch_start_time

            # Perplexity is exp(loss) — a standard interpretable metric for language models
            # Lower is better; a perplexity of 1 would mean perfect prediction
            train_perplexity = math.exp(train_loss)
            val_perplexity = math.exp(val_loss)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_perplexity": round(train_perplexity, 4),
                "val_perplexity": round(val_perplexity, 4),
                "teacher_forcing_ratio": round(teacher_forcing_ratio, 4),
                "elapsed_seconds": round(elapsed, 2),
            }
            self.training_history.append(epoch_record)
            self.write_log()

            logger.info(
                f"Epoch {epoch:>3}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train PPL: {train_perplexity:.2f} | Val PPL: {val_perplexity:.2f} | "
                f"TF Ratio: {teacher_forcing_ratio:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                self.save_checkpoint(epoch, val_loss, filename="best_model.pt")
                logger.info(f"  New best model saved (val loss: {val_loss:.4f})")

            # Always save the most recent epoch so training can be resumed
            self.save_checkpoint(epoch, val_loss, filename="latest_model.pt")

        logger.info("Training complete.")
        logger.info(f"Best validation loss: {self.best_validation_loss:.4f}")


def build_trainer(model: Seq2Seq, config: dict, device: torch.device) -> Trainer:
    """
    Convenience factory that reads the training config and returns a ready-to-use Trainer.

    Expects the config to have a 'training' key with the following structure:
        training:
            learning_rate: 0.001
            teacher_forcing_ratio: 0.5
            teacher_forcing_min: 0.1
            gradient_clip: 1.0
            lr_decay_factor: 0.5
            lr_patience: 2
            pad_token_idx: 0

    Args:
        model:  the Seq2Seq model to train
        config: parsed YAML config dict
        device: torch.device

    Returns:
        A configured Trainer instance.
    """
    training_cfg = config["training"]

    optimizer = Adam(model.parameters(), lr=training_cfg["learning_rate"])

    # CrossEntropyLoss with ignore_index so padding positions don't contribute to the loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=training_cfg["pad_token_idx"], label_smoothing=0.1
    )

    return Trainer(model, optimizer, criterion, config, device)
