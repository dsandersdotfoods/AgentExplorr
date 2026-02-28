r"""
Reusable PyTorch Training Loop
================================

A professional, reusable training loop for any PyTorch model. Handles
training, validation, learning rate scheduling, early stopping, checkpoint
saving/loading, and progress logging.

WHY A REUSABLE TRAINING LOOP?
    Every PyTorch training script follows the same pattern:
        for epoch in epochs:
            for batch in dataloader:
                loss = model(batch)
                loss.backward()
                optimizer.step()

    But in practice, you need MUCH more:
      - Validation after each epoch (catch overfitting)
      - Learning rate scheduling (decay LR over time)
      - Early stopping (stop when validation loss stops improving)
      - Checkpoint saving (don't lose progress if training crashes)
      - Gradient clipping (prevent exploding gradients in RNNs/Transformers)
      - Progress logging (know what's happening during long training runs)

    Writing this boilerplate for every project is tedious and error-prone.
    This Trainer class encapsulates all of it into a clean, reusable interface.

THE TRAINING LOOP EXPLAINED:

    +-------------------------------------------------------------+
    |  for epoch in range(max_epochs):                            |
    |                                                              |
    |    # --- TRAINING PHASE ---                                  |
    |    model.train()           <-- Enable dropout, batch norm    |
    |    for batch in train_loader:                                |
    |      optimizer.zero_grad()  <-- Clear old gradients          |
    |      loss = criterion(model(x), y)  <-- Forward pass         |
    |      loss.backward()        <-- Compute gradients            |
    |      clip_grad_norm_()      <-- Prevent exploding grads      |
    |      optimizer.step()       <-- Update weights               |
    |                                                              |
    |    # --- VALIDATION PHASE ---                                |
    |    model.eval()            <-- Disable dropout, use          |
    |    with torch.no_grad():    running batch norm stats         |
    |      val_loss = ...         <-- No gradients needed          |
    |                                                              |
    |    # --- SCHEDULING & CHECKPOINTING ---                      |
    |    scheduler.step()         <-- Adjust learning rate         |
    |    if val_loss improved:                                     |
    |      save_checkpoint()      <-- Save best model              |
    |    if patience exhausted:                                    |
    |      break (early stop)     <-- Stop overfitting             |
    |                                                              |
    +-------------------------------------------------------------+

EARLY STOPPING:
    Without early stopping, training continues even after the model starts
    overfitting (validation loss goes UP while training loss goes DOWN).
    Early stopping watches validation loss and stops training when it hasn't
    improved for `patience` consecutive epochs.

    Training loss:    -------> (always decreasing)
    Validation loss:  ------\                (decreases, then...)
                             \------/\/\--   (starts oscillating/increasing)
                              ^
                          STOP HERE (best model)

LEARNING RATE SCHEDULING:
    The learning rate (LR) controls how big each weight update is:
      - Too high: training is unstable, loss oscillates or diverges
      - Too low: training is slow, might get stuck in local minima

    Common strategies:
      - StepLR: divide LR by gamma every N epochs
      - CosineAnnealing: smoothly decay LR following a cosine curve
      - ReduceOnPlateau: reduce LR when validation loss stalls
      - Warmup + Decay: start with tiny LR, increase, then decay (Transformers)

LEARNING RESOURCES:
    - PyTorch Training Loop: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    - Learning Rate Scheduling: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    - VIDEO: "PyTorch Training Loop" — https://www.youtube.com/watch?v=c36lUUr864M
    - VIDEO: "Train Neural Networks in PyTorch" — https://www.youtube.com/watch?v=Jy4wM2X21u0
    - VIDEO: "Learning Rate Schedules" — https://www.youtube.com/watch?v=kk2BQRhmGRY
    - VIDEO: "Early Stopping Explained" — https://www.youtube.com/watch?v=zmhfBAiG3ak
    - PyTorch Lightning (production alternative): https://lightning.ai/docs/pytorch/stable/
    - Andrej Karpathy's "A Recipe for Training NNs": https://karpathy.github.io/2019/04/25/recipe/
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Training History — tracks all metrics across epochs
# =============================================================================

@dataclass
class TrainingHistory:
    """Record of metrics across training epochs.

    This dataclass stores the complete training history so you can
    plot loss curves, analyze learning rate schedules, and compare
    train vs validation metrics.

    Attributes:
        train_losses: Loss for each training epoch.
        val_losses: Loss for each validation epoch.
        train_accuracies: Accuracy for each training epoch (classification only).
        val_accuracies: Accuracy for each validation epoch (classification only).
        learning_rates: Learning rate at each epoch.
        epoch_times: Time taken for each epoch (seconds).
        best_val_loss: Best validation loss seen so far.
        best_epoch: Epoch at which best validation loss was achieved.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert history to a plain dictionary (for serialization).

        Returns:
            Dictionary with all training metrics.
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving.

    HOW IT WORKS:
        Track the best validation loss seen so far. If the loss doesn't
        improve (decrease) for `patience` consecutive epochs, trigger
        early stopping. This prevents overfitting by stopping training
        at the point where the model generalizes best.

    THE min_delta PARAMETER:
        Sometimes the loss improves by a tiny amount (e.g., 0.00001).
        That's noise, not real improvement. min_delta sets a minimum
        improvement threshold — the loss must improve by at least this
        much to reset the patience counter.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        counter: How many epochs since last improvement.
        best_loss: Best validation loss seen so far.
        should_stop: Whether training should stop.

    Example:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = validate(model)
        ...     early_stop(val_loss)
        ...     if early_stop.should_stop:
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.best_loss: float = float("inf")
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Meaningful improvement — reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement — increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "early_stopping_triggered",
                    patience=self.patience,
                    best_loss=round(self.best_loss, 6),
                    current_loss=round(val_loss, 6),
                    epochs_without_improvement=self.counter,
                )

        return self.should_stop


# =============================================================================
# The Trainer
# =============================================================================

class Trainer:
    """Reusable PyTorch training loop with all best practices built in.

    This trainer works with ANY nn.Module — CNNs, Transformers, MLPs, etc.
    Just provide a model, optimizer, and loss function.

    Features:
      - Training and validation epochs
      - Learning rate scheduling (any PyTorch scheduler)
      - Early stopping (configurable patience and min_delta)
      - Gradient clipping (prevent exploding gradients)
      - Checkpoint saving/loading (resume training, deploy best model)
      - Structured progress logging
      - Automatic best-model restoration after training

    Example:
        >>> model = SimpleCNN()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> trainer = Trainer(model, optimizer, loss_fn)
        >>> history = trainer.fit(train_loader, val_loader, epochs=20, patience=5)
        >>> print(f"Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")

    Args:
        model: PyTorch model to train (any nn.Module).
        optimizer: Optimizer (Adam, SGD, AdamW, etc.).
        loss_fn: Loss function (CrossEntropyLoss, MSELoss, etc.).
        device: Device to train on ("cpu", "cuda", "mps").
        scheduler: Optional learning rate scheduler.
        grad_clip_value: Maximum gradient norm for clipping. Set to None
            to disable. 1.0 is a safe default for Transformers/RNNs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
        scheduler: Any | None = None,
        grad_clip_value: float | None = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_value = grad_clip_value

        # Best model state (for restoring after training)
        self._best_model_state: dict[str, Any] | None = None

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            "trainer_initialized",
            model_class=model.__class__.__name__,
            optimizer_class=optimizer.__class__.__name__,
            loss_fn_class=loss_fn.__class__.__name__,
            device=str(self.device),
            grad_clip=grad_clip_value,
            total_params=total_params,
            trainable_params=trainable_params,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch.

        HOW TRAINING WORKS (for each batch):
          1. model.train()          -- enables dropout, batch norm training mode
          2. optimizer.zero_grad()  -- clear gradients (they accumulate by default!)
          3. outputs = model(inputs) -- forward pass
          4. loss = loss_fn(outputs, targets) -- compute loss
          5. loss.backward()        -- backward pass (compute gradients via autograd)
          6. clip_grad_norm_()      -- prevent exploding gradients (optional)
          7. optimizer.step()       -- update weights using computed gradients

        Args:
            dataloader: Training data loader.

        Returns:
            Dict with "loss" and optionally "accuracy".
        """
        self.model.train()  # Enable training mode (dropout ON, batch norm training stats)
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            # Step 1: Zero gradients — CRITICAL! PyTorch accumulates gradients by default.
            # If you forget this, gradients from previous batches add up and
            # your training will be unstable or diverge.
            self.optimizer.zero_grad()

            # Step 2: Forward pass — push data through the network
            outputs = self.model(inputs)

            # Step 3: Compute loss — measure how wrong the predictions are
            loss = self.loss_fn(outputs, targets)

            # Step 4: Backward pass — compute d(loss)/d(weight) for every parameter
            # This is the magic of autograd: PyTorch automatically computes
            # gradients through the entire computation graph using the chain rule.
            loss.backward()

            # Step 5: Gradient clipping — prevent exploding gradients
            # This is especially important for RNNs and Transformers where
            # gradients can explode due to long sequences.
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_value,
                )

            # Step 6: Optimizer step — update weights using computed gradients
            # For SGD: weight = weight - lr * gradient
            # For Adam: more sophisticated update with momentum and adaptive lr
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * inputs.size(0)

            # Track accuracy for classification tasks
            # outputs shape: (batch, num_classes) — take argmax for predicted class
            if outputs.dim() > 1 and outputs.size(-1) > 1:
                _, predicted = outputs.max(dim=-1)
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        metrics: dict[str, float] = {"loss": avg_loss}
        if total > 0:
            metrics["accuracy"] = correct / total

        return metrics

    @torch.no_grad()  # Disable gradient computation — saves memory + speed
    def validate_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one validation epoch (no gradient computation).

        WHY torch.no_grad()?
          During validation, we don't update weights, so we don't need
          gradients. Disabling gradient computation:
          - Saves memory (no computation graph stored)
          - Speeds up computation (~30% faster)

        WHY model.eval()?
          Switches batch normalization and dropout to evaluation mode:
          - BatchNorm uses running statistics instead of batch statistics
          - Dropout is disabled (all neurons active)
          Forgetting model.eval() is a common bug that causes validation
          metrics to be unreliable!

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

            # Forward pass only — no backward, no optimizer step
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            if outputs.dim() > 1 and outputs.size(-1) > 1:
                _, predicted = outputs.max(dim=-1)
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        metrics: dict[str, float] = {"loss": avg_loss}
        if total > 0:
            metrics["accuracy"] = correct / total

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        patience: int = 5,
        min_delta: float = 0.001,
        checkpoint_dir: str | None = None,
    ) -> TrainingHistory:
        """Train the model for multiple epochs with early stopping.

        This is the main method you'll call. It orchestrates the entire
        training process:
          1. Train for one epoch
          2. Validate (if val_loader provided)
          3. Adjust learning rate (if scheduler provided)
          4. Check early stopping (if patience > 0)
          5. Save checkpoint (if val loss improved)
          6. Repeat until done or early stopped

        EARLY STOPPING:
          If validation loss doesn't improve for `patience` epochs, training
          stops. This prevents overfitting — the point where the model
          memorizes training data instead of learning general patterns.

          Analogy: Studying for an exam. At some point, more studying leads
          to memorizing specific practice problems rather than understanding
          the concepts. Early stopping catches this inflection point.

        Args:
            train_loader: Training data.
            val_loader: Validation data (optional but recommended!).
                Without validation, you can't detect overfitting.
            epochs: Maximum training epochs.
            patience: Epochs without improvement before stopping.
                Set to 0 to disable early stopping.
            min_delta: Minimum improvement for early stopping to reset patience.
            checkpoint_dir: Directory to save model checkpoints.
                If None, best model is kept in memory only.

        Returns:
            TrainingHistory with metrics for every epoch.
        """
        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) if patience > 0 else None

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "training_started",
            max_epochs=epochs,
            patience=patience,
            train_batches=len(train_loader),
            val_batches=len(val_loader) if val_loader else 0,
        )

        print("\n" + "=" * 75)
        print("TRAINING STARTED")
        print("=" * 75)

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            # --- Training Phase ---
            train_metrics = self.train_epoch(train_loader)
            history.train_losses.append(train_metrics["loss"])
            if "accuracy" in train_metrics:
                history.train_accuracies.append(train_metrics["accuracy"])

            # Record current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history.learning_rates.append(current_lr)

            # --- Validation Phase ---
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                history.val_losses.append(val_metrics["loss"])
                if "accuracy" in val_metrics:
                    history.val_accuracies.append(val_metrics["accuracy"])

                val_loss = val_metrics["loss"]

                # Check for improvement and save best model
                if val_loss < history.best_val_loss:
                    history.best_val_loss = val_loss
                    history.best_epoch = epoch
                    # Save best model state in memory
                    self._best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    # Save checkpoint to disk if directory provided
                    if checkpoint_dir:
                        self.save_checkpoint(
                            f"{checkpoint_dir}/best_model.pt",
                            epoch=epoch,
                            val_loss=val_loss,
                            history=history,
                        )

            # --- Learning Rate Scheduling ---
            if self.scheduler is not None:
                # ReduceLROnPlateau needs the validation loss as input
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # --- Epoch Timing ---
            epoch_time = time.perf_counter() - epoch_start
            history.epoch_times.append(epoch_time)

            # --- Progress Logging ---
            log_parts = [f"Epoch {epoch:3d}/{epochs}"]
            log_parts.append(f"Train Loss: {train_metrics['loss']:.4f}")
            if "accuracy" in train_metrics:
                log_parts.append(f"Train Acc: {train_metrics['accuracy']:.4f}")
            if val_metrics:
                log_parts.append(f"Val Loss: {val_metrics['loss']:.4f}")
                if "accuracy" in val_metrics:
                    log_parts.append(f"Val Acc: {val_metrics['accuracy']:.4f}")
            log_parts.append(f"LR: {current_lr:.6f}")
            log_parts.append(f"Time: {epoch_time:.1f}s")

            print(" | ".join(log_parts))

            logger.info(
                "epoch_completed",
                epoch=epoch,
                train_loss=round(train_metrics["loss"], 6),
                val_loss=round(val_metrics.get("loss", 0.0), 6),
                train_acc=round(train_metrics.get("accuracy", 0.0), 4),
                val_acc=round(val_metrics.get("accuracy", 0.0), 4),
                lr=current_lr,
                epoch_time=round(epoch_time, 2),
            )

            # --- Early Stopping ---
            if early_stopping is not None and val_loader is not None:
                if early_stopping(val_metrics["loss"]):
                    print(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best val loss: {early_stopping.best_loss:.6f} "
                        f"at epoch {history.best_epoch}"
                    )
                    break

        # --- Restore Best Model ---
        # After training (or early stopping), load the best model weights
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            logger.info(
                "best_model_restored",
                best_epoch=history.best_epoch,
                best_val_loss=round(history.best_val_loss, 6),
            )

        # --- Training Summary ---
        print("=" * 75)
        print("TRAINING COMPLETED")
        print(f"  Total epochs:    {len(history.train_losses)}")
        print(f"  Best val loss:   {history.best_val_loss:.6f} (epoch {history.best_epoch})")
        if history.val_accuracies:
            best_val_acc = max(history.val_accuracies)
            print(f"  Best val acc:    {best_val_acc:.4f}")
        total_time = sum(history.epoch_times)
        print(f"  Total time:      {total_time:.1f}s")
        print("=" * 75)

        return history

    def save_checkpoint(
        self,
        path: str,
        epoch: int | None = None,
        val_loss: float | None = None,
        history: TrainingHistory | None = None,
    ) -> None:
        """Save model and optimizer state to disk.

        WHY SAVE BOTH MODEL AND OPTIMIZER?
          The optimizer state (momentum buffers, adaptive learning rates)
          is needed to resume training. Without it, training would
          effectively restart from scratch with a warm model.

        WHAT'S SAVED:
          - model_state_dict: all model weights
          - optimizer_state_dict: optimizer state (momentum, adaptive LR, etc.)
          - scheduler_state_dict: scheduler state (if applicable)
          - epoch: which epoch we're at
          - val_loss: validation loss at this checkpoint
          - history: full training history (for analysis)

        Args:
            path: File path for the checkpoint (.pt file).
            epoch: Current epoch number.
            val_loss: Validation loss at this checkpoint.
            history: Training history to include.
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if history is not None:
            checkpoint["history"] = history.to_dict()

        torch.save(checkpoint, path)
        logger.info("checkpoint_saved", path=path, epoch=epoch)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load model and optimizer state from disk.

        Use this to resume training from a checkpoint or to load a
        pre-trained model for inference.

        Args:
            path: Path to checkpoint file.

        Returns:
            The checkpoint dictionary (for inspection if needed).

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # map_location ensures GPU checkpoints can be loaded on CPU and vice versa
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            "checkpoint_loaded",
            path=path,
            epoch=checkpoint.get("epoch"),
            val_loss=checkpoint.get("val_loss"),
        )

        return checkpoint

    def get_learning_rate(self) -> float:
        """Get the current learning rate.

        Returns:
            Current learning rate from the first parameter group.
        """
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see the Trainer in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TRAINER DEMO")
    print("Training a simple model on random data")
    print("=" * 70)

    # Create a simple model for demonstration
    class DemoModel(nn.Module):
        """Tiny model for testing the Trainer."""
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # Create random data (simulating a 5-class classification problem)
    from torch.utils.data import TensorDataset

    X_train = torch.randn(500, 20)
    y_train = torch.randint(0, 5, (500,))
    X_val = torch.randn(100, 20)
    y_val = torch.randint(0, 5, (100,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model, optimizer, loss function, and scheduler
    model = DemoModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # CosineAnnealingLR smoothly decreases the LR following a cosine curve:
    #   LR starts at max and decays to eta_min over T_max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )

    # Create Trainer and train!
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        scheduler=scheduler,
        grad_clip_value=1.0,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        patience=5,
        min_delta=0.001,
    )

    # Print summary of last 3 epochs
    print("\nFinal Training History (last 3 epochs):")
    n = min(3, len(history.train_losses))
    for i in range(-n, 0):
        epoch_num = len(history.train_losses) + i + 1
        parts = [f"Epoch {epoch_num}"]
        parts.append(f"Loss={history.train_losses[i]:.4f}")
        if history.val_losses:
            parts.append(f"Val Loss={history.val_losses[i]:.4f}")
        parts.append(f"LR={history.learning_rates[i]:.6f}")
        print("  " + " | ".join(parts))
