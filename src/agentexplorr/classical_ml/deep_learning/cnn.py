"""
Convolutional Neural Network (CNN) for Image Classification
=============================================================

A from-scratch CNN built with PyTorch, designed for CIFAR-10 (32x32 RGB images,
10 classes). Every layer is explained in detail — this is your first real
neural network, and we want you to understand EVERY piece.

WHAT IS A CNN?
    A Convolutional Neural Network is a neural network designed specifically for
    data with spatial structure (images, audio spectrograms, video frames).

    The key insight: nearby pixels are related. A CNN exploits this by:
      1. Using small filters (kernels) that slide across the image
      2. Each filter learns to detect a specific pattern (edge, texture, shape)
      3. Stacking layers builds a hierarchy: edges → textures → parts → objects

ARCHITECTURE DIAGRAM (ASCII):

    Input Image (3 x 32 x 32)  ← 3 channels (RGB), 32x32 pixels
    │
    ├─── Conv2d(3→32, 3x3, pad=1)  ← 32 filters, each 3x3, learn edge detectors
    ├─── BatchNorm2d(32)             ← Normalize activations for stable training
    ├─── ReLU                        ← Non-linearity: f(x) = max(0, x)
    ├─── MaxPool2d(2x2)             ← Downsample 32x32 → 16x16 (take max in each 2x2 block)
    │    Output: 32 x 16 x 16
    │
    ├─── Conv2d(32→64, 3x3, pad=1) ← 64 filters, each 3x3, learn texture detectors
    ├─── BatchNorm2d(64)
    ├─── ReLU
    ├─── MaxPool2d(2x2)             ← Downsample 16x16 → 8x8
    │    Output: 64 x 8 x 8
    │
    ├─── Conv2d(64→128, 3x3, pad=1) ← 128 filters, learn higher-level features
    ├─── BatchNorm2d(128)
    ├─── ReLU
    ├─── MaxPool2d(2x2)              ← Downsample 8x8 → 4x4
    │    Output: 128 x 4 x 4
    │
    ├─── Flatten                     ← Reshape 128x4x4 = 2048 → flat vector
    │    Output: 2048
    │
    ├─── Linear(2048→256)            ← Fully connected layer
    ├─── ReLU
    ├─── Dropout(0.5)                ← Randomly zero 50% of neurons (regularization)
    │
    ├─── Linear(256→128)             ← Another FC layer
    ├─── ReLU
    ├─── Dropout(0.3)
    │
    └─── Linear(128→10)              ← Output layer: 10 classes (CIFAR-10)
         Output: 10 logits (raw scores, NOT probabilities yet)

    Total parameters: ~550K (small enough to train on CPU in minutes)

KEY CONCEPTS EXPLAINED:

    CONVOLUTION:
        A convolution slides a small filter (e.g., 3x3) across the image,
        computing a dot product at each position. If the filter matches the
        local pattern, it produces a high activation.

        Example 3x3 edge-detection filter:
          [-1  0  1]
          [-1  0  1]     This detects vertical edges (bright-to-dark transitions)
          [-1  0  1]

        The network LEARNS these filters through backpropagation — we don't
        hand-design them. Early layers learn edges, middle layers learn textures,
        deep layers learn object parts.

    POOLING (MaxPool):
        MaxPool2d(2x2) takes the maximum value in each 2x2 region, reducing
        spatial dimensions by half. This provides:
          1. Translation invariance (small shifts don't change the output)
          2. Dimensionality reduction (fewer parameters downstream)
          3. Larger receptive field (each neuron "sees" more of the image)

    ACTIVATION (ReLU):
        ReLU(x) = max(0, x)
        Without activation functions, stacking linear layers just gives
        another linear layer (matrix multiplication is associative).
        ReLU introduces non-linearity, letting the network learn complex
        functions. It's simple, fast, and avoids the vanishing gradient
        problem that plagued sigmoid/tanh.

    BATCH NORMALIZATION:
        Normalizes activations within each mini-batch to have mean=0, std=1.
        Benefits:
          1. Stabilizes training (prevents activation explosion/collapse)
          2. Allows higher learning rates
          3. Acts as light regularization
          4. Reduces sensitivity to weight initialization

    DROPOUT:
        During training, randomly sets a fraction of neurons to zero.
        This prevents co-adaptation (neurons relying on specific other neurons)
        and acts as a form of ensemble learning (each forward pass uses a
        different sub-network). During evaluation, all neurons are active
        but outputs are scaled down.

CIFAR-10 DATASET:
    - 60,000 32x32 color images in 10 classes
    - 50,000 training + 10,000 test
    - Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - This is the "Hello World" of image classification (after MNIST)

LEARNING RESOURCES:
    - CS231n CNN Lecture Notes: https://cs231n.github.io/convolutional-networks/
    - PyTorch CNN Tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    - VIDEO: "But what is a convolution?" (3Blue1Brown) — https://www.youtube.com/watch?v=KuXjwB4LzSA
    - VIDEO: "CNN Explainer" (interactive visualization) — https://poloclub.github.io/cnn-explainer/
    - VIDEO: "Convolutional Neural Networks Explained" — https://www.youtube.com/watch?v=YRhxdVk_sIs
    - VIDEO: "Batch Norm Explained" — https://www.youtube.com/watch?v=yXOMHOpbon8
    - PAPER: LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition"
    - PAPER: Krizhevsky et al. (2012) "ImageNet Classification with Deep CNNs" (AlexNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# CIFAR-10 class names — useful for converting model outputs to human-readable labels
CIFAR10_CLASSES: list[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 image classification.

    This network processes 32x32 RGB images through three convolutional blocks
    (each: Conv → BatchNorm → ReLU → MaxPool), then two fully connected layers
    with dropout regularization, producing 10 class logits.

    Architecture:
        3x32x32 → [Conv Block 1] → 32x16x16
                 → [Conv Block 2] → 64x8x8
                 → [Conv Block 3] → 128x4x4
                 → Flatten → 2048
                 → FC(256) → FC(128) → FC(10)

    Args:
        num_classes: Number of output classes (10 for CIFAR-10).
        dropout_rate: Dropout probability for FC layers.

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> x = torch.randn(8, 3, 32, 32)  # batch of 8 images
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([8, 10])
        >>> probs = torch.softmax(logits, dim=1)  # convert to probabilities
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initialize the CNN architecture.

        HOW nn.Module WORKS:
            All PyTorch models inherit from nn.Module. You must:
              1. Call super().__init__() (registers submodules)
              2. Define layers as attributes (so PyTorch can track their parameters)
              3. Implement forward() (defines the computation)

            PyTorch automatically:
              - Tracks all parameters (model.parameters())
              - Handles device placement (model.to("cuda"))
              - Switches between train/eval modes (model.train() / model.eval())
              - Computes gradients via autograd (loss.backward())
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # =====================================================================
        # CONVOLUTIONAL BLOCK 1: Input processing
        # =====================================================================
        # Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        #
        # in_channels=3: RGB image has 3 color channels
        # out_channels=32: we learn 32 different 3x3 filters
        #   Each filter detects a different pattern (edges, colors, textures)
        # kernel_size=3: each filter is 3x3 pixels
        #   3x3 is the most common choice (captures local patterns efficiently)
        # padding=1: add 1 pixel of zeros around the border
        #   This preserves the spatial dimensions: 32x32 in → 32x32 out
        #   Without padding, the output would be 30x30 (loses 2 pixels per dim)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )
        # BatchNorm normalizes across the batch dimension for each channel.
        # Input: (batch, 32, H, W) → normalize each of the 32 channels
        self.bn1 = nn.BatchNorm2d(32)

        # =====================================================================
        # CONVOLUTIONAL BLOCK 2: Feature extraction
        # =====================================================================
        # After Block 1 + MaxPool: 32 channels, 16x16 spatial
        # We increase channels (32 → 64) as spatial size decreases.
        # WHY? Deeper layers need more filters to capture more complex patterns.
        # Think of it as: fewer pixels to process, but more types of features.
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # =====================================================================
        # CONVOLUTIONAL BLOCK 3: High-level features
        # =====================================================================
        # After Block 2 + MaxPool: 64 channels, 8x8 spatial
        # 128 filters to detect high-level combinations of textures and shapes
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # =====================================================================
        # POOLING LAYER (shared across all blocks)
        # =====================================================================
        # MaxPool2d(kernel_size=2, stride=2):
        # - Takes the MAX value in each 2x2 window
        # - stride=2 means non-overlapping windows
        # - Reduces spatial dimensions by 2x each time:
        #   32x32 → 16x16 → 8x8 → 4x4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # =====================================================================
        # FULLY CONNECTED (CLASSIFICATION HEAD)
        # =====================================================================
        # After conv blocks: 128 channels x 4x4 spatial = 2048 features
        # We flatten this into a 1D vector and pass through linear layers.

        # FC1: 2048 → 256 (compress the feature representation)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

        # FC2: 256 → 128 (further compress)
        self.fc2 = nn.Linear(256, 128)

        # FC3 (output): 128 → num_classes (one score per class)
        # These scores are called "logits" — raw, unnormalized predictions.
        # We apply softmax OUTSIDE the model (CrossEntropyLoss does it internally).
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout layers with different rates
        # Higher dropout in earlier FC layers (more parameters to regularize)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=max(0.1, dropout_rate - 0.2))

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            "cnn_initialized",
            num_classes=num_classes,
            total_params=total_params,
            trainable_params=trainable_params,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: process an image through the CNN.

        THE FORWARD PASS EXPLAINED:
            Data flows through the network in one direction (hence "forward"):
              Input → Conv1 → BN1 → ReLU → Pool → Conv2 → ... → FC3 → Logits

            Each layer transforms the data:
              - Conv: detects spatial patterns → (batch, channels, H, W)
              - BN: normalizes activations → same shape
              - ReLU: introduces non-linearity → same shape
              - Pool: reduces spatial dimensions → (batch, channels, H/2, W/2)
              - FC: combines all features → (batch, num_classes)

        TENSOR SHAPES (for a batch of 8 images):
            Input:           (8,   3, 32, 32)  ← batch_size=8, RGB, 32x32
            After conv1+pool: (8,  32, 16, 16)
            After conv2+pool: (8,  64,  8,  8)
            After conv3+pool: (8, 128,  4,  4)
            After flatten:    (8, 2048)
            After fc1:        (8,  256)
            After fc2:        (8,  128)
            Output:           (8,   10)         ← 10 class logits

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).
               Values should be normalized (e.g., mean=0, std=1 per channel).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
            Apply softmax to get probabilities: probs = torch.softmax(logits, dim=1)
        """
        # --- Convolutional Block 1 ---
        # Conv → BatchNorm → ReLU → MaxPool
        # Input: (batch, 3, 32, 32) → Output: (batch, 32, 16, 16)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # --- Convolutional Block 2 ---
        # Input: (batch, 32, 16, 16) → Output: (batch, 64, 8, 8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # --- Convolutional Block 3 ---
        # Input: (batch, 64, 8, 8) → Output: (batch, 128, 4, 4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # --- Flatten ---
        # Reshape from (batch, 128, 4, 4) to (batch, 128*4*4) = (batch, 2048)
        # The -1 means "infer this dimension automatically"
        # x.size(0) is the batch size (we keep it, flatten everything else)
        x = x.view(x.size(0), -1)

        # --- Fully Connected Layers ---
        # FC1: 2048 → 256 with ReLU and dropout
        x = self.dropout1(F.relu(self.fc1(x)))

        # FC2: 256 → 128 with ReLU and dropout
        x = self.dropout2(F.relu(self.fc2(x)))

        # FC3 (output): 128 → 10 (raw logits, NO activation)
        # WHY NO SOFTMAX HERE?
        #   PyTorch's CrossEntropyLoss internally applies log_softmax + NLLLoss.
        #   Applying softmax here AND in the loss would be double-softmax (wrong!).
        #   For inference, apply softmax manually: probs = torch.softmax(logits, dim=1)
        x = self.fc3(x)

        return x

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with probabilities and class labels.

        This is a convenience method that applies softmax and argmax to get
        human-readable predictions. Use this for inference, NOT training.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Tuple of (probabilities, predicted_classes):
              - probabilities: (batch_size, num_classes) — softmax probabilities
              - predicted_classes: (batch_size,) — integer class labels
        """
        # Set model to eval mode (disables dropout, uses batch norm running stats)
        self.eval()

        # torch.no_grad() disables gradient computation, saving memory and time.
        # We don't need gradients during inference (no backpropagation).
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

        return probabilities, predicted_classes

    def get_num_parameters(self) -> dict[str, int]:
        """Count model parameters by layer.

        Understanding parameter counts helps you:
          1. Estimate memory requirements (each float32 param = 4 bytes)
          2. Identify bottlenecks (which layer has the most params?)
          3. Compare model complexity across architectures

        Returns:
            Dictionary mapping layer names to parameter counts.
        """
        param_counts: dict[str, int] = {}
        total = 0

        for name, param in self.named_parameters():
            count = param.numel()  # Number of elements in the tensor
            param_counts[name] = count
            total += count

        param_counts["TOTAL"] = total

        return param_counts


def create_cifar10_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create CIFAR-10 train and test DataLoaders with data augmentation.

    DATA AUGMENTATION:
        Artificially increases the effective size of the training set by
        applying random transformations (flip, crop, rotation, color jitter).
        This teaches the model to be invariant to these transformations.

        A horizontally flipped cat is still a cat. By training on flipped
        images, the model learns this invariance without needing more data.

        We ONLY augment the TRAINING data. Test data should be clean
        (only normalized) to give a fair evaluation.

    NORMALIZATION:
        We normalize each channel (R, G, B) to have mean=0, std=1 using
        the dataset statistics:
          mean = [0.4914, 0.4822, 0.4465]  (per-channel means across CIFAR-10)
          std  = [0.2470, 0.2435, 0.2616]  (per-channel stds across CIFAR-10)

        This is the same idea as StandardScaler in sklearn — models train
        faster and more stably when inputs are normalized.

    Args:
        batch_size: Number of images per training batch.
            Larger batches = more stable gradients, faster training (with enough GPU)
            Smaller batches = more noise in gradients (acts as regularization)
            64 is a good default for CIFAR-10 on a single GPU.
        data_dir: Directory to download/store CIFAR-10 data.
        num_workers: Number of parallel data loading processes.
            Set to 0 for debugging (single-process, easier to get error messages).

    Returns:
        Tuple of (train_loader, test_loader).
    """
    from torchvision import datasets, transforms

    # --- Training transforms (with data augmentation) ---
    train_transform = transforms.Compose([
        # RandomCrop: randomly crop a 32x32 patch from a padded 36x36 image.
        # This simulates small translations of the object in the image.
        transforms.RandomCrop(32, padding=4),

        # RandomHorizontalFlip: 50% chance of flipping the image horizontally.
        # A flipped car is still a car. (We don't flip vertically because
        # upside-down objects are rare in CIFAR-10.)
        transforms.RandomHorizontalFlip(p=0.5),

        # Convert PIL Image to PyTorch tensor (scales pixels from [0,255] to [0,1])
        transforms.ToTensor(),

        # Normalize to zero mean and unit variance (per channel)
        # These values are the dataset statistics, computed once and hardcoded.
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    # --- Test transforms (NO augmentation, only normalization) ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    # Download CIFAR-10 if not already present
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data each epoch (important!)
        num_workers=num_workers,
        pin_memory=True,    # Speeds up CPU→GPU transfer
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # Don't shuffle test data (reproducible evaluation)
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "cifar10_dataloaders_created",
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        batch_size=batch_size,
        n_batches_train=len(train_loader),
        n_batches_test=len(test_loader),
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see the CNN architecture
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SimpleCNN for CIFAR-10")
    print("=" * 60)

    # Create model
    model = SimpleCNN(num_classes=10, dropout_rate=0.5)

    # Print architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    print("\nParameter Counts:")
    params = model.get_num_parameters()
    for name, count in params.items():
        if name == "TOTAL":
            print(f"  {'─' * 40}")
        print(f"  {name}: {count:,}")

    # Test forward pass with random data
    print("\nForward Pass Test:")
    dummy_input = torch.randn(8, 3, 32, 32)  # Batch of 8 random "images"
    logits = model(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")

    # Test prediction
    probs, classes = model.predict(dummy_input)
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Predicted classes:   {classes.tolist()}")
    print(f"  Class names:         {[CIFAR10_CLASSES[c] for c in classes.tolist()]}")

    print("\nModel is ready for training! Use the Trainer class from training_loop.py.")
