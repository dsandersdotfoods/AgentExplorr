"""
LoRA Fine-Tuning for Large Language Models
============================================

WHAT IS LoRA? (Low-Rank Adaptation)
  LoRA is the most popular parameter-efficient fine-tuning method. Instead
  of updating ALL parameters in a large model, LoRA freezes the original
  weights and injects small trainable matrices into specific layers.

  THE MATH (simplified):
    In a standard linear layer:  y = W * x       (W is a huge matrix)
    With LoRA:                   y = W * x + B * A * x

    Where:
      W = original frozen weights (e.g., 4096 x 4096 = 16.7M parameters)
      A = small matrix of shape (4096 x r) where r << 4096 (e.g., r=16)
      B = small matrix of shape (r x 4096)
      B * A = low-rank update (only r * 4096 * 2 = 131K parameters!)

    So instead of updating 16.7M parameters per layer, we update only 131K.
    That's a 128x reduction! And it works almost as well as full fine-tuning.

  ┌───────────────────────────────────────────────────────────────┐
  │  WHY "LOW-RANK"?                                              │
  │                                                               │
  │  The key insight from the LoRA paper is that the weight       │
  │  updates during fine-tuning have LOW INTRINSIC RANK.          │
  │  Translation: the actual "information" in the weight update   │
  │  can be captured by a much smaller matrix.                    │
  │                                                               │
  │  Think of it like image compression: a 4K photo can be        │
  │  compressed to 1/10th its size because most of the            │
  │  "information" lives in a low-dimensional subspace.           │
  │  LoRA does the same thing for weight updates.                 │
  └───────────────────────────────────────────────────────────────┘

KEY HYPERPARAMETERS:
  r (rank): The rank of the low-rank matrices A and B.
    - Higher r = more parameters = more capacity = slower training
    - Lower r = fewer parameters = less capacity = faster training
    - Typical values: 8, 16, 32, 64
    - Start with r=16 for most tasks

  lora_alpha: A scaling factor for the LoRA update.
    - The actual scaling is: alpha / r
    - Higher alpha = stronger LoRA effect = model changes more
    - Common practice: alpha = 2 * r (e.g., r=16, alpha=32)
    - Some practitioners set alpha = r for a scaling factor of 1.0

  target_modules: Which layers to apply LoRA to.
    - In transformers, the main matrix multiplications happen in:
      - q_proj, k_proj, v_proj (attention query/key/value projections)
      - o_proj (attention output projection)
      - gate_proj, up_proj, down_proj (MLP/feed-forward layers)
    - The LoRA paper focused on attention layers (q_proj, v_proj)
    - Empirically, applying to ALL linear layers often works better
    - More target modules = more trainable parameters = more memory

  lora_dropout: Dropout probability for LoRA layers.
    - Regularization to prevent overfitting
    - Typical values: 0.0 to 0.1
    - Use higher dropout for smaller datasets

LEARNING RESOURCES:
  - LoRA paper: https://arxiv.org/abs/2106.09685
  - PEFT library docs: https://huggingface.co/docs/peft
  - TRL (Trainer for RL) docs: https://huggingface.co/docs/trl
  - VIDEO: "LoRA Explained in 10 Minutes" (Umar Jamil):
    https://www.youtube.com/watch?v=PXWYUTMt-AU
  - VIDEO: "Fine-Tune ANY LLM with LoRA" (Matt Williams):
    https://www.youtube.com/watch?v=eC6Hd1hFvos
  - VIDEO: "SFTTrainer Deep Dive" (Hugging Face):
    https://www.youtube.com/watch?v=lQCJL0YEk2U
  - BLOG: "Practical Tips for Finetuning LLMs Using LoRA"
    (Sebastian Raschka): https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from agentexplorr.core import get_logger, load_yaml_config

# ─── Module Logger ──────────────────────────────────────────────────────
logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.

    This dataclass holds ALL the knobs you can turn when fine-tuning with
    LoRA. Each parameter is documented with what it does, why it matters,
    and what values to try.

    QUICK START:
      For most instruction-tuning tasks, the defaults here work well.
      The most impactful parameters to tune (in order of importance):
        1. learning_rate (start with 2e-4, try 1e-4 and 5e-4)
        2. num_train_epochs (start with 3, try 1-5)
        3. lora_r (start with 16, try 8 and 32)
        4. per_device_train_batch_size (as high as VRAM allows)
    """

    # ─── Model Configuration ────────────────────────────────────────
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    """Hugging Face model to fine-tune.

    TinyLlama is the DEFAULT because:
      - It's only 1.1B parameters (fits in ~4.5 GB VRAM in FP32)
      - It's surprisingly capable for its size
      - Fast to train (minutes, not hours)
      - Perfect for LEARNING the fine-tuning process

    For production:
      - "mistralai/Mistral-7B-v0.1" (7B, best quality/size ratio)
      - "meta-llama/Llama-2-7b-hf" (7B, Meta, needs HF token)
      - "microsoft/phi-2" (2.7B, great quality for size)
    """

    # ─── LoRA Hyperparameters ───────────────────────────────────────
    lora_r: int = 16
    """Rank of the LoRA decomposition matrices.

    This is the MOST IMPORTANT LoRA hyperparameter. It controls how many
    parameters are added (and thus how much the model can change).

    Rule of thumb:
      - r=8:  Minimal adaptation. Good for simple tasks like classification.
      - r=16: Good default. Works well for instruction following.
      - r=32: More capacity. Better for complex tasks or larger datasets.
      - r=64: Maximum common value. Approaches full fine-tuning quality.
      - r>64: Diminishing returns. You're spending parameters inefficiently.

    MEMORY IMPACT: Each doubling of r roughly doubles the trainable parameters
    and increases memory by ~10-20% (the base model still dominates memory).
    """

    lora_alpha: int = 32
    """Scaling factor for LoRA updates.

    The LoRA update is scaled by (alpha / r). So with r=16 and alpha=32,
    the scaling factor is 2.0 — the LoRA update is amplified 2x.

    Common strategies:
      - alpha = 2 * r (scaling factor = 2.0) — our default, slightly aggressive
      - alpha = r (scaling factor = 1.0) — conservative, stable training
      - alpha = 1 (scaling factor = 1/r) — very conservative

    If training is unstable (loss spikes or NaN), try reducing alpha.
    If the model isn't changing enough, try increasing alpha.
    """

    lora_dropout: float = 0.05
    """Dropout probability applied to LoRA layers.

    Dropout randomly zeroes out some LoRA activations during training.
    This prevents the model from relying too heavily on specific LoRA
    neurons, which reduces overfitting.

    Guidelines:
      - 0.0:  No dropout. Fine for large datasets (>10K examples).
      - 0.05: Light dropout. Good default for medium datasets.
      - 0.1:  Moderate dropout. Good for small datasets (<1K examples).
      - 0.2:  Heavy dropout. Only if you're severely overfitting.
    """

    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",  # Query projection in self-attention
            "k_proj",  # Key projection in self-attention
            "v_proj",  # Value projection in self-attention
            "o_proj",  # Output projection in self-attention
            "gate_proj",  # Gate in the MLP (for gated architectures like Llama)
            "up_proj",  # Up projection in the MLP
            "down_proj",  # Down projection in the MLP
        ]
    )
    """Which linear layers to apply LoRA to.

    WHAT ARE THESE LAYERS?
      In a Transformer, each layer has two main sub-modules:

      1. Self-Attention (how tokens attend to each other):
         - q_proj: Computes query vectors (what am I looking for?)
         - k_proj: Computes key vectors (what do I contain?)
         - v_proj: Computes value vectors (what information do I carry?)
         - o_proj: Projects attention output back to model dimension

      2. MLP/Feed-Forward (non-linear transformation):
         - gate_proj: Gating mechanism (Llama-style architectures)
         - up_proj: Project up to intermediate dimension
         - down_proj: Project back down to model dimension

    The LoRA paper originally only targeted q_proj and v_proj. Later work
    showed that targeting ALL linear layers usually gives better results
    (the QLoRA paper demonstrated this convincingly).

    TRADE-OFF: More target modules = more trainable parameters = better
    quality but more memory and slower training.
    """

    # ─── Training Hyperparameters ───────────────────────────────────
    num_train_epochs: int = 3
    """Number of passes through the entire training dataset.

    - 1 epoch: The model sees every example once. Often undertrained.
    - 3 epochs: Good default. Model sees each example 3 times.
    - 5+ epochs: Risk of overfitting, especially on small datasets.

    Watch val loss: if it starts increasing while train loss decreases,
    you're overfitting and should use fewer epochs (or more dropout).
    """

    per_device_train_batch_size: int = 4
    """Number of examples processed per GPU per training step.

    Larger batch sizes:
      + More stable gradient estimates (less noisy updates)
      + Faster training (better GPU utilization)
      - More VRAM required
      - May generalize slightly worse (the "large batch training" debate)

    If you get CUDA OOM (Out Of Memory):
      1. Reduce batch_size to 2 or 1
      2. Increase gradient_accumulation_steps to compensate
      3. Effective batch size = batch_size * gradient_accumulation_steps
    """

    gradient_accumulation_steps: int = 4
    """Accumulate gradients over this many steps before updating weights.

    This is a MEMORY-SAVING TRICK. Instead of doing one big forward pass
    with batch_size=16, we do 4 forward passes with batch_size=4 and
    accumulate (sum) the gradients before doing the weight update.

    Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    With our defaults: 4 * 4 = 16 effective batch size.

    Larger effective batch sizes generally lead to smoother training.
    """

    learning_rate: float = 2e-4
    """Learning rate for the optimizer (AdamW).

    THE most important training hyperparameter. It controls how big each
    weight update step is.

    - Too high (>1e-3): Training is unstable, loss oscillates or diverges
    - Too low (<1e-5):  Training is very slow, model barely changes
    - Sweet spot (1e-4 to 3e-4): Where most LoRA fine-tuning works best

    We use a warmup schedule: the learning rate starts at 0 and linearly
    increases to this value over the first warmup_ratio fraction of training.
    This prevents early instability when the model hasn't adapted yet.
    """

    warmup_ratio: float = 0.03
    """Fraction of training steps for learning rate warmup.

    During warmup, the learning rate linearly increases from 0 to
    learning_rate. This prevents large, destructive gradient updates
    at the start of training when the LoRA weights are randomly initialized.

    - 0.03: Warm up for 3% of training steps (good default)
    - 0.05-0.10: More conservative warmup (helps with unstable training)
    """

    max_seq_length: int = 512
    """Maximum sequence length for training.

    Must match the value used during data preparation!
    VRAM usage scales quadratically with sequence length (because of
    self-attention), so doubling this roughly quadruples memory usage.
    """

    weight_decay: float = 0.01
    """L2 regularization factor.

    Weight decay shrinks the magnitude of model weights, which acts as
    regularization to prevent overfitting. Applied to all parameters
    EXCEPT biases and LayerNorm weights (which should not be regularized).

    - 0.0:  No regularization (fine for large datasets)
    - 0.01: Light regularization (good default)
    - 0.1:  Strong regularization (if overfitting badly)
    """

    output_dir: str = "./outputs/lora_finetuned"
    """Directory to save checkpoints and final adapter weights."""

    logging_steps: int = 10
    """Log training metrics (loss, learning rate) every N steps."""

    save_strategy: str = "epoch"
    """When to save checkpoints: "epoch", "steps", or "no"."""

    fp16: bool = False
    """Use FP16 mixed precision training.

    Mixed precision uses FP16 for forward/backward passes and FP32 for
    weight updates. This halves memory usage and can be 2x faster on
    GPUs with Tensor Cores (RTX 3000+, A100, etc.).

    Set to True if your GPU supports it (most modern GPUs do).
    Set to False if you're on CPU or have issues with NaN losses.
    """

    bf16: bool = False
    """Use BF16 mixed precision training.

    BF16 (Brain Float 16) has the same exponent range as FP32 but with
    reduced precision. It's more numerically stable than FP16 (no need
    for loss scaling) but requires Ampere+ GPUs (RTX 3000+, A100).

    PREFERENCE ORDER:
      1. bf16=True (if available — check torch.cuda.is_bf16_supported())
      2. fp16=True (if bf16 not available)
      3. Neither (CPU or debugging)
    """

    seed: int = 42
    """Random seed for reproducibility."""


class LoRAFineTuner:
    """Fine-tune a language model using LoRA (Low-Rank Adaptation).

    This class encapsulates the complete LoRA fine-tuning pipeline:
      1. Load a pre-trained base model from Hugging Face
      2. Apply LoRA adapters to specified layers
      3. Train using SFTTrainer (Supervised Fine-Tuning Trainer from TRL)
      4. Save the trained adapter weights
      5. Optionally merge adapters back into the base model

    TYPICAL USAGE:
        >>> from agentexplorr.llm_training.fine_tune_lora import (
        ...     LoRAFineTuner, LoRAConfig
        ... )
        >>> from agentexplorr.llm_training.data_preparation import (
        ...     DatasetPreparer, DatasetConfig
        ... )
        >>>
        >>> # Prepare data
        >>> data_config = DatasetConfig(max_samples=1000)
        >>> preparer = DatasetPreparer(data_config)
        >>> processed = preparer.prepare("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>>
        >>> # Configure and train with LoRA
        >>> config = LoRAConfig(num_train_epochs=3, learning_rate=2e-4)
        >>> tuner = LoRAFineTuner(config)
        >>> tuner.train(processed.train_dataset, processed.val_dataset)
        >>>
        >>> # Save adapter (small, ~10-50 MB)
        >>> tuner.save_adapter("./my_adapter")
        >>>
        >>> # Merge adapter into base model for easy deployment
        >>> tuner.merge_and_save("./my_merged_model")

    WHAT GETS SAVED?
      - Adapter-only save: Just the LoRA weights (~10-50 MB). The base model
        must be loaded separately and the adapter applied at inference time.
        Pros: Tiny file size, can share adapters easily, swap between tasks.

      - Merged save: The LoRA weights are merged INTO the base model weights.
        You get a single, standalone model. Pros: Simpler inference (just
        load one model). Cons: Large file (~2-14 GB), can't swap adapters.

    LEARNING RESOURCES:
      - PEFT LoRA guide: https://huggingface.co/docs/peft/conceptual_guides/lora
      - TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
      - VIDEO: "LoRA From Scratch" (Chris Olah style):
        https://www.youtube.com/watch?v=DhRoTONcyZE
    """

    def __init__(self, config: LoRAConfig | None = None) -> None:
        """Initialize the LoRA fine-tuner.

        Args:
            config: LoRA configuration. Uses defaults if not provided.
        """
        self.config = config or LoRAConfig()
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.trainer: SFTTrainer | None = None

        # Detect hardware capabilities for automatic precision selection.
        # This makes the code "just work" on different hardware.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            logger.info(
                "gpu_detected",
                device_name=torch.cuda.get_device_name(0),
                vram_gb=round(
                    torch.cuda.get_device_properties(0).total_mem / 1e9, 1
                ),
            )
        else:
            logger.warning(
                "no_gpu_detected",
                message="Training will be VERY slow on CPU. "
                "Consider using Google Colab (free GPU).",
            )

        logger.info(
            "lora_fine_tuner_initialized",
            base_model=self.config.base_model_name,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> LoRAFineTuner:
        """Create a LoRAFineTuner from a YAML config file.

        This is the RECOMMENDED way to create a tuner for experiments.
        YAML configs are:
          - Version controllable (git track your experiments)
          - Shareable (send a config file, not a code snippet)
          - Readable (comments explain each parameter)

        Args:
            config_path: Path to a YAML file with LoRA config.
                See configs/lora_config.yaml for the expected format.

        Returns:
            Configured LoRAFineTuner.

        Example:
            >>> tuner = LoRAFineTuner.from_yaml("configs/lora_config.yaml")
        """
        raw = load_yaml_config(config_path)

        # Merge YAML config with defaults. YAML values override defaults.
        # This means you only need to specify the parameters you want to
        # change — everything else uses the sensible defaults from LoRAConfig.
        config = LoRAConfig(**raw)
        return cls(config)

    def load_model(self) -> None:
        """Load the base model and tokenizer from Hugging Face Hub.

        HOW MODEL LOADING WORKS:
          AutoModelForCausalLM.from_pretrained() does several things:
            1. Downloads model weights from HF Hub (cached locally)
            2. Instantiates the correct model architecture
            3. Loads the pre-trained weights into the model
            4. Moves everything to the specified device (CPU/GPU)

          The model is loaded in FULL PRECISION (FP32) by default.
          For a 1.1B model like TinyLlama, this is ~4.4 GB in VRAM.
          For QLoRA (see fine_tune_qlora.py), we load in 4-bit instead.

        DEVICE MAPPING:
          device_map="auto" uses Hugging Face Accelerate to automatically
          distribute the model across available devices. If you have 1 GPU,
          it all goes there. If you have multiple GPUs, it shards the model.
          If the model is too big for GPU, it offloads layers to CPU/disk.
        """
        logger.info("loading_base_model", model=self.config.base_model_name)

        # Load the tokenizer first — it's always needed and is lightweight
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
        )

        # Ensure pad token exists (see data_preparation.py for explanation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load the model
        # torch_dtype: Use float16 or bfloat16 to save memory on GPU.
        # On CPU, we must use float32 (CPU doesn't support float16 well).
        if self._device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=dtype,
            # device_map="auto" requires the `accelerate` package.
            # It intelligently places model layers across available devices.
            device_map="auto",
            trust_remote_code=True,
        )

        # IMPORTANT: Disable caching during training.
        # KV-cache speeds up inference by caching intermediate attention
        # computations. But during training, it's incompatible with
        # gradient checkpointing and wastes memory. Always disable it.
        self.model.config.use_cache = False

        # Print model size information for educational purposes
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "base_model_loaded",
            total_parameters=f"{total_params:,}",
            total_parameters_gb=f"{total_params * 4 / 1e9:.2f} GB (FP32)",
            dtype=str(dtype),
        )

    def apply_lora(self) -> None:
        """Apply LoRA adapters to the loaded base model.

        WHAT HAPPENS HERE:
          1. We create a LoraConfig that specifies which layers to modify
             and how many parameters to add.
          2. get_peft_model() wraps the original model, freezing all
             original weights and injecting trainable LoRA matrices.
          3. After this call, only the LoRA matrices are trainable.

        Before apply_lora():
          model.parameters() → ALL parameters are trainable
          Trainable params: 1,100,000,000 (1.1B for TinyLlama)

        After apply_lora():
          model.parameters() → Only LoRA params are trainable
          Trainable params: ~5,000,000 (0.45% of total!)

        This massive reduction is WHY LoRA works on consumer hardware.

        Raises:
            RuntimeError: If model hasn't been loaded yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        logger.info(
            "applying_lora_adapters",
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )

        # Create the LoRA configuration.
        # TaskType.CAUSAL_LM tells PEFT this is a decoder-only language model
        # (GPT-style, left-to-right generation). Other task types include
        # SEQ_2_SEQ_LM (encoder-decoder like T5) and SEQ_CLS (classification).
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            # bias="none" means we don't train bias parameters.
            # The LoRA paper found that training biases gives minimal benefit
            # for decoder-only models. Other options: "all", "lora_only".
            bias="none",
        )

        # Wrap the model with LoRA adapters.
        # This operation is IN-PLACE — self.model is now a PeftModel.
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameter statistics — this is one of the most
        # satisfying outputs in all of ML. You'll see something like:
        #   "trainable params: 4,194,304 || all params: 1,100,048,384 || trainable%: 0.38"
        self.model.print_trainable_parameters()

        # Log the same info programmatically
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / all_params

        logger.info(
            "lora_applied",
            trainable_params=f"{trainable_params:,}",
            all_params=f"{all_params:,}",
            trainable_pct=f"{trainable_pct:.2f}%",
            # The adapter size is approximately: trainable_params * 2 bytes (FP16)
            estimated_adapter_size_mb=f"{trainable_params * 2 / 1e6:.1f} MB",
        )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Run the training loop.

        WHAT HAPPENS DURING TRAINING:
          For each batch of examples:
            1. FORWARD PASS: Feed tokens through model → get predicted next tokens
            2. LOSS COMPUTATION: Compare predictions to actual next tokens
               (cross-entropy loss, ignoring padding tokens marked as -100)
            3. BACKWARD PASS: Compute gradients of loss w.r.t. LoRA parameters
               (original frozen parameters get zero gradients)
            4. OPTIMIZER STEP: Update LoRA parameters using AdamW optimizer
            5. LEARNING RATE SCHEDULE: Adjust learning rate (warmup + cosine decay)

          This repeats for every batch, for every epoch.

        SFTTrainer (from TRL):
          SFTTrainer is a specialized version of Hugging Face's Trainer
          designed for Supervised Fine-Tuning of language models. It adds:
            - Automatic prompt formatting
            - Support for packing multiple short examples into one sequence
            - Integration with PEFT (LoRA/QLoRA)
            - Response-only training (mask the prompt, only train on response)

        Args:
            train_dataset: Tokenized training dataset.
            val_dataset: Optional tokenized validation dataset for monitoring.

        Returns:
            Dictionary of training metrics (train_loss, val_loss, etc.).

        Raises:
            RuntimeError: If model hasn't been loaded or LoRA not applied.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() and apply_lora() first."
            )

        logger.info(
            "training_starting",
            train_size=len(train_dataset),
            val_size=len(val_dataset) if val_dataset else 0,
            epochs=self.config.num_train_epochs,
            batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation=self.config.gradient_accumulation_steps,
            effective_batch_size=(
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
            ),
            learning_rate=self.config.learning_rate,
        )

        # Set up training arguments.
        # TrainingArguments is the Hugging Face mega-config that controls
        # EVERYTHING about the training loop.
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy="epoch" if val_dataset is not None else "no",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            # Gradient checkpointing trades compute for memory:
            # Instead of storing all intermediate activations (for backward
            # pass), it recomputes them as needed. Uses ~60% less memory
            # at the cost of ~20% more computation. Almost always worth it.
            gradient_checkpointing=True,
            # The "cosine" scheduler gradually reduces the learning rate
            # following a cosine curve. This is standard for LLM training
            # and usually works better than constant or linear decay.
            lr_scheduler_type="cosine",
            # Optim: "adamw_torch" is the standard. For 8-bit optimizers
            # (saves memory), use "adamw_8bit" (requires bitsandbytes).
            optim="adamw_torch",
            # Report metrics to the console (and optionally to W&B/MLflow)
            report_to="none",
            # Don't load the best model at end — we'll save it explicitly
            load_best_model_at_end=False,
            # Max gradient norm for gradient clipping. Prevents exploding
            # gradients from destabilizing training.
            max_grad_norm=1.0,
        )

        # Create the SFT Trainer.
        # We pass the already-tokenized dataset, so we set
        # dataset_text_field=None and max_seq_length to match our data.
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            # max_seq_length caps the sequence length.
            # Must match what we used during tokenization.
            max_seq_length=self.config.max_seq_length,
        )

        # ─── THE TRAINING LOOP ──────────────────────────────────────
        # trainer.train() handles EVERYTHING:
        #   - Batching and data loading
        #   - Forward pass, loss computation, backward pass
        #   - Gradient accumulation and clipping
        #   - Optimizer step and learning rate scheduling
        #   - Checkpointing and logging
        #   - Mixed precision (if enabled)
        #   - Multi-GPU distribution (if available)
        logger.info("training_loop_starting")
        train_result = self.trainer.train()

        # Extract and log training metrics
        metrics = train_result.metrics
        logger.info(
            "training_complete",
            train_loss=round(metrics.get("train_loss", 0.0), 4),
            train_runtime=round(metrics.get("train_runtime", 0.0), 1),
            train_samples_per_second=round(
                metrics.get("train_samples_per_second", 0.0), 1
            ),
        )

        # Run final evaluation on the validation set
        if val_dataset is not None:
            eval_metrics = self.trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info(
                "evaluation_complete",
                eval_loss=round(eval_metrics.get("eval_loss", 0.0), 4),
            )

        return metrics

    def save_adapter(self, output_dir: str | Path | None = None) -> Path:
        """Save only the LoRA adapter weights (NOT the full model).

        WHY SAVE ADAPTERS SEPARATELY?
          The adapter weights are typically 10-100 MB, while the base model
          is 2-14+ GB. Saving just the adapter means:
            - Fast saves and loads
            - Easy to share (attach to an email or HF Hub)
            - Multiple adapters can share one base model
            - You can A/B test different adapters

        WHAT'S SAVED:
          - adapter_model.safetensors: The LoRA weight matrices
          - adapter_config.json: LoRA hyperparameters (r, alpha, etc.)

        To load later:
          >>> from peft import PeftModel
          >>> base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/...")
          >>> model = PeftModel.from_pretrained(base_model, "./my_adapter")

        Args:
            output_dir: Where to save. Defaults to config.output_dir.

        Returns:
            Path to the saved adapter directory.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        save_path = Path(output_dir or self.config.output_dir) / "adapter"
        save_path.mkdir(parents=True, exist_ok=True)

        # save_pretrained saves ONLY the adapter weights, not the base model.
        # This is possible because PeftModel tracks which parameters are
        # "original" vs "LoRA-added".
        self.model.save_pretrained(str(save_path))

        # Also save the tokenizer alongside the adapter
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(
            "adapter_saved",
            path=str(save_path),
            # Report the size of saved files
            note="To load: PeftModel.from_pretrained(base_model, path)",
        )

        return save_path

    def merge_and_save(self, output_dir: str | Path | None = None) -> Path:
        """Merge LoRA adapters into the base model and save.

        WHAT IS MERGING?
          Remember the LoRA formula: y = W*x + B*A*x
          Merging computes W_new = W + B*A and saves W_new.

          After merging, the model is a standard model with NO adapter —
          the LoRA modifications are baked into the weights permanently.

        WHEN TO MERGE:
          - For deployment: Merged model has no PEFT dependency, simpler inference
          - For sharing: Upload a single model to HF Hub
          - For further training: Use the merged model as a new base

        WHEN NOT TO MERGE:
          - During experimentation: Keep adapters separate for flexibility
          - Multi-task: Different adapters for different tasks, same base

        Args:
            output_dir: Where to save the merged model.

        Returns:
            Path to the saved merged model.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise RuntimeError("No model to merge. Train first.")

        save_path = Path(output_dir or self.config.output_dir) / "merged"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("merging_adapter_with_base_model")

        # merge_and_unload() does two things:
        #   1. merge: Computes W_new = W + B*A for every LoRA layer
        #   2. unload: Removes the PEFT wrapper, returning a vanilla model
        merged_model = self.model.merge_and_unload()

        # Save the merged model — this is now a standard HF model
        merged_model.save_pretrained(str(save_path))

        # Save tokenizer too — you always want them together
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(
            "merged_model_saved",
            path=str(save_path),
            note="This is a standalone model. Load with AutoModelForCausalLM.",
        )

        return save_path

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from the fine-tuned model.

        GENERATION PARAMETERS:
          - max_new_tokens: Maximum number of tokens to generate.
            This is DIFFERENT from max_length (total length including prompt).
            Use max_new_tokens to control response length independently.

          - temperature: Controls randomness of generation.
            - 0.0: Deterministic (always pick the most likely token)
            - 0.7: Moderate randomness (good for creative tasks)
            - 1.0: Full randomness from the model's distribution
            - >1.0: Extra random (usually too chaotic)

          - top_p (nucleus sampling): Only sample from the top tokens
            whose cumulative probability exceeds p.
            - 0.9: Sample from tokens covering 90% probability mass
            - 1.0: No filtering (consider all tokens)
            Lower top_p = more focused, higher top_p = more diverse.

        Args:
            prompt: The input prompt (use DatasetPreparer.format_for_inference()).
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text (response only, prompt stripped).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return PyTorch tensors
        ).to(self.model.device)

        # Generate
        with torch.no_grad():  # No gradient computation during inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # do_sample=True enables sampling (probabilistic generation).
                # Without it, the model uses greedy decoding (always picks
                # the most likely token), which is deterministic but bland.
                do_sample=temperature > 0.0,
                # Repetition penalty discourages the model from repeating
                # the same tokens. 1.0 = no penalty, >1.0 = less repetition.
                repetition_penalty=1.15,
            )

        # Decode only the NEW tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()
