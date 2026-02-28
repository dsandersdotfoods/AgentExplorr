"""
Reusable PyTorch Training Loop
================================

WHY A REUSABLE TRAINER?
  Writing training loops from scratch every time is error-prone and tedious.
  This Trainer class provides a battle-tested loop with:
  - Epoch-level training and validation
  - Early stopping (prevent overfitting)
  - Learning rate scheduling (cosine annealing)
  - Checkpoint saving/loading
  - Structured progress logging

THE TRAINING LOOP EXPLAINED:
  For each epoch:
    1. TRAIN PHASE — model.train()
       - Forward pass: predictions = model(inputs)
       - Loss: how wrong are the predictions?
       - Backward pass: compute gradients (∂loss/∂weights)
       - Optimizer step: adjust weights to reduce loss

    2. VALIDATION PHASE — model.eval()
       - Same as training but WITHOUT gradient computation
       - Measures generalization (does it work on unseen data?)
       - If val loss stops improving → early stopping

LEARNING RESOURCES:
  - PyTorch training loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
  - Early stopping explained: https://machinelearningmastery.com/early-stopping-to-avoid-overtraining/
  - Learning rate scheduling: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  - VIDEO: "PyTorch Training Loop" — https://www.youtube.com/watch?v=c36lUUr864M
  - VIDEO: "Train Neural Networks in PyTorch" — https://www.youtube.com/watch?v=Jy4wM2X21u0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Record of metrics across training epochs.

    Attributes:
        train_losses: Loss for each training epoch.
        val_losses: Loss for each validation epoch.
        train_accuracies: Accuracy for each training epoch (if applicable).
        val_accuracies: Accuracy for each validation epoch (if applicable).
        learning_rates: Learning rate at each epoch.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0


class Trainer:
    """Reusable PyTorch training loop with best practices.

    This trainer works with ANY nn.Module — CNNs, Transformers, MLPs, etc.
    Just provide a model, optimizer, and loss function.

    Example:
        >>> model = SimpleCNN()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> trainer = Trainer(model, optimizer, loss_fn)
        >>> history = trainer.fit(train_loader, val_loader, epochs=20, patience=5)
        >>> print(f"Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer (Adam, SGD, etc.).
        loss_fn: Loss function (CrossEntropyLoss, MSELoss, etc.).
        device: Device to train on ("cpu", "cuda", "mps").
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train_epoch(self, dataloader: DataLoader[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        """Run one training epoch.

        HOW TRAINING WORKS:
          1. model.train() — enables dropout, batch norm in training mode
          2. For each batch:
             a. Zero gradients (they accumulate by default in PyTorch!)
             b. Forward pass: compute predictions
             c. Compute loss (how wrong are we?)
             d. Backward pass: compute gradients via backpropagation
             e. Optimizer step: update weights (gradient descent)

        Args:
            dataloader: Training data loader.

        Returns:
            Dict with "loss" and optionally "accuracy".
        """
        self.model.train()  # Enable training mode
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            # Zero gradients — CRITICAL! PyTorch accumulates gradients by default.
            # If you forget this, gradients from previous batches add up and
            # your training will be unstable or diverge.
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self.loss_fn(outputs, targets)

            # Backward pass — compute ∂loss/∂weights for every parameter
            # This is the magic of autograd: PyTorch automatically computes
            # gradients through the entire computation graph.
            loss.backward()

            # Optimizer step — update weights using computed gradients
            # For SGD: weight = weight - lr * gradient
            # For Adam: more sophisticated update with momentum and adaptive lr
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            # Track accuracy for classification tasks
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / total
        metrics: dict[str, float] = {"loss": avg_loss}
        if total > 0 and correct > 0:
            metrics["accuracy"] = correct / total

        return metrics

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader[tuple[torch.Tensor, ...]]) -> dict[str, float]:
        """Run one validation epoch (no gradient computation).

        WHY torch.no_grad()?
          During validation, we don't need gradients because we're not
          updating weights. Disabling gradient computation:
          - Saves memory (no computation graph stored)
          - Speeds up computation (~30% faster)

        WHY model.eval()?
          Switches batch normalization and dropout to evaluation mode:
          - BatchNorm uses running statistics instead of batch statistics
          - Dropout is disabled (all neurons active)

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict with "loss" and optionally "accuracy".
        """
        self.model.eval()  # Disable dropout, use running batch norm stats
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / total
        metrics: dict[str, float] = {"loss": avg_loss}
        if total > 0 and correct > 0:
            metrics["accuracy"] = correct / total

        return metrics

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, ...]],
        val_loader: DataLoader[tuple[torch.Tensor, ...]] | None = None,
        epochs: int = 10,
        patience: int = 5,
        checkpoint_dir: str | None = None,
    ) -> TrainingHistory:
        """Train the model for multiple epochs with early stopping.

        EARLY STOPPING:
          If validation loss doesn't improve for `patience` epochs, training
          stops. This prevents overfitting — the point where the model
          memorizes training data instead of learning general patterns.

          Analogy: Studying for an exam. At some point, more studying leads
          to memorizing specific practice problems rather than understanding
          the concepts. Early stopping catches this point.

        Args:
            train_loader: Training data.
            val_loader: Validation data (recommended!).
            epochs: Maximum training epochs.
            patience: Epochs without improvement before stopping.
            checkpoint_dir: Directory to save model checkpoints.

        Returns:
            TrainingHistory with metrics for every epoch.
        """
        history = TrainingHistory()
        patience_counter = 0

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # --- Training ---
            train_metrics = self.train_epoch(train_loader)
            history.train_losses.append(train_metrics["loss"])
            if "accuracy" in train_metrics:
                history.train_accuracies.append(train_metrics["accuracy"])

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history.learning_rates.append(current_lr)

            # --- Validation ---
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                history.val_losses.append(val_metrics["loss"])
                if "accuracy" in val_metrics:
                    history.val_accuracies.append(val_metrics["accuracy"])

                val_loss = val_metrics["loss"]

                # Check for improvement
                if val_loss < history.best_val_loss:
                    history.best_val_loss = val_loss
                    history.best_epoch = epoch
                    patience_counter = 0

                    # Save best model checkpoint
                    if checkpoint_dir:
                        self.save_checkpoint(f"{checkpoint_dir}/best_model.pt")
                else:
                    patience_counter += 1
            else:
                val_metrics = {}

            # --- Logging ---
            log_parts = [f"Epoch {epoch}/{epochs}"]
            log_parts.append(f"train_loss={train_metrics['loss']:.4f}")
            if "accuracy" in train_metrics:
                log_parts.append(f"train_acc={train_metrics['accuracy']:.4f}")
            if val_metrics:
                log_parts.append(f"val_loss={val_metrics['loss']:.4f}")
                if "accuracy" in val_metrics:
                    log_parts.append(f"val_acc={val_metrics['accuracy']:.4f}")
            log_parts.append(f"lr={current_lr:.6f}")

            logger.info("epoch_complete", **dict(s.split("=") for s in log_parts[1:]))

            # --- Learning Rate Scheduling ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Early Stopping ---
            if patience_counter >= patience and val_loader is not None:
                logger.info(
                    "early_stopping",
                    epoch=epoch,
                    best_epoch=history.best_epoch,
                    best_val_loss=f"{history.best_val_loss:.4f}",
                )
                break

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state to disk.

        WHY SAVE BOTH MODEL AND OPTIMIZER?
          The optimizer state (momentum buffers, adaptive learning rates)
          is needed to resume training. Without it, training would
          effectively restart from scratch with a warm model.

        Args:
            path: File path for the checkpoint (.pt file).
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("checkpoint_saved", path=path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from disk.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("checkpoint_loaded", path=path)
