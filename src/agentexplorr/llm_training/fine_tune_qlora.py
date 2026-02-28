"""
QLoRA Fine-Tuning for Large Language Models
=============================================

WHAT IS QLoRA? (Quantized Low-Rank Adaptation)
  QLoRA = LoRA + 4-bit Quantization of the base model.

  With standard LoRA, the base model is loaded in FP16/BF16 (2 bytes per param).
  A 7B model needs ~14 GB VRAM just for the frozen weights.

  QLoRA loads the base model in 4-bit precision (0.5 bytes per param).
  That same 7B model now needs only ~3.5 GB for frozen weights.

  This is a GAME-CHANGER for accessibility:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Model Size  │  FP32 (Full)  │  FP16 (LoRA)  │  4-bit (QLoRA) │
    │──────────────│───────────────│───────────────│────────────────│
    │  1.1B        │  4.4 GB       │  2.2 GB       │  ~0.7 GB       │
    │  7B          │  28 GB        │  14 GB        │  ~3.5 GB       │
    │  13B         │  52 GB        │  26 GB        │  ~6.5 GB       │
    │  70B         │  280 GB       │  140 GB       │  ~35 GB        │
    └─────────────────────────────────────────────────────────────────┘

  With QLoRA, you can fine-tune a 7B model on a FREE Google Colab GPU
  (T4 with 15 GB VRAM)! Without QLoRA, you'd need a $10,000+ A100.

WHAT IS QUANTIZATION?
  Quantization reduces the precision of model weights to save memory.

  PRECISION FORMATS:
    FP32 (32 bits, 4 bytes):
      - Full IEEE 754 float: 1 sign + 8 exponent + 23 mantissa bits
      - Range: +-3.4 x 10^38, ~7 decimal digits of precision
      - This is what your CPU uses for math by default

    FP16 (16 bits, 2 bytes):
      - Half precision: 1 sign + 5 exponent + 10 mantissa bits
      - Range: +-65504, ~3 decimal digits of precision
      - 2x memory savings over FP32, minimal quality loss for inference

    BF16 (16 bits, 2 bytes):
      - Brain Float: 1 sign + 8 exponent + 7 mantissa bits
      - Same range as FP32 (wider than FP16) but less precision
      - Best of both worlds for training stability

    INT8 (8 bits, 1 byte):
      - Integer quantization: 256 discrete values
      - 4x memory savings over FP32
      - Some quality loss, especially for outlier weights

    INT4 / NF4 (4 bits, 0.5 bytes):
      - Only 16 discrete values per weight!
      - 8x memory savings over FP32
      - NF4 (NormalFloat4) assumes weights follow a normal distribution
        and places the 16 quantization levels optimally for that distribution
      - This is what QLoRA uses — surprisingly good quality!

  ┌─────────────────────────────────────────────────────────────────┐
  │  INTUITION: Think of quantization like choosing a color palette  │
  │                                                                 │
  │  FP32: 16 million colors (true color photo)                     │
  │  FP16: 65,536 colors (still looks great to human eyes)          │
  │  INT8: 256 colors (like a GIF — noticeable artifacts on close   │
  │        inspection but overall picture is preserved)              │
  │  NF4:  16 colors (like pixel art — amazingly, ML models handle  │
  │        this well because the "important" weights keep their      │
  │        relative magnitudes)                                      │
  └─────────────────────────────────────────────────────────────────┘

DOUBLE QUANTIZATION:
  QLoRA introduces "double quantization" — quantizing the quantization
  constants themselves. Here's what that means:

  When you quantize a block of weights to NF4, you need to store a
  scaling factor (FP32) for each block. With block_size=64, that's one
  FP32 constant per 64 weights = 0.5 bits overhead per weight.

  Double quantization quantizes THOSE scaling factors to INT8, reducing
  the overhead from 0.5 bits to ~0.125 bits per weight. This saves an
  additional ~3 GB on a 65B model.

  It's "quantization all the way down" — and it works!

LEARNING RESOURCES:
  - QLoRA paper (READ THIS — it's well-written and accessible):
    https://arxiv.org/abs/2305.14314
  - BitsAndBytes library: https://github.com/TimDettmers/bitsandbytes
  - HF Quantization docs: https://huggingface.co/docs/transformers/quantization
  - VIDEO: "QLoRA — Fine-tune LLMs on Your GPU" (Trelis Research):
    https://www.youtube.com/watch?v=XpoKB3usmKc
  - VIDEO: "Quantization Explained Simply" (Efficient NLP):
    https://www.youtube.com/watch?v=MvKwi2Q6d6o
  - VIDEO: "QLoRA and 4-Bit Quantization" (Yannic Kilcher):
    https://www.youtube.com/watch?v=TPcXVJ1VSRI
  - BLOG: "Making LLMs even more accessible with bitsandbytes"
    (Hugging Face): https://huggingface.co/blog/4bit-transformers-bitsandbytes
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
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from agentexplorr.core import get_logger, load_yaml_config

# ─── Module Logger ──────────────────────────────────────────────────────
logger = get_logger(__name__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning.

    QLoRA extends LoRA with 4-bit quantization of the base model.
    This config covers BOTH the LoRA parameters AND the quantization settings.

    The key difference from LoRAConfig is the quantization section.
    Training hyperparameters are largely the same, with some adjustments
    to account for the quantized base model.

    QUICK REFERENCE — MEMORY REQUIREMENTS (approximate):
      TinyLlama 1.1B + QLoRA: ~2-3 GB VRAM (fits on ANY modern GPU)
      Mistral 7B + QLoRA:     ~6-8 GB VRAM (fits on Colab T4)
      Llama 2 13B + QLoRA:    ~12-15 GB VRAM (needs Colab A100 or L4)
      Llama 2 70B + QLoRA:    ~40-48 GB VRAM (needs multiple GPUs or A100-80G)
    """

    # ─── Model Configuration ────────────────────────────────────────
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    """See LoRAConfig.base_model_name for model recommendations."""

    # ─── Quantization Configuration ─────────────────────────────────
    # These settings control how the BASE MODEL weights are quantized.
    # The LoRA adapter weights are always in FP16/BF16 (NOT quantized),
    # because they need full precision for gradient updates.

    load_in_4bit: bool = True
    """Load the base model in 4-bit precision.

    This is the main QLoRA feature. It reduces memory by ~4x compared
    to FP16 loading. The 4-bit weights are dequantized to BF16/FP16
    on-the-fly during the forward pass, so the computation is still
    done in higher precision.
    """

    bnb_4bit_quant_type: str = "nf4"
    """Which 4-bit quantization scheme to use.

    Options:
      - "nf4" (NormalFloat4): Assumes weights follow a normal distribution.
        Places quantization levels optimally for normally-distributed values.
        This is what the QLoRA paper recommends and uses.

      - "fp4" (Float4): A more general 4-bit float format.
        Slightly less memory-efficient than NF4.

    USE NF4. The QLoRA paper showed it consistently outperforms FP4.
    Neural network weights empirically follow a normal distribution
    (thanks to common initialization schemes like Kaiming/Xavier),
    so NF4's assumption is well-justified.
    """

    bnb_4bit_compute_dtype: str = "bfloat16"
    """Dtype for computation during the forward/backward pass.

    Even though weights are STORED in 4-bit, actual matrix multiplications
    happen in this dtype. The 4-bit weights are dequantized on-the-fly.

    Options:
      - "bfloat16": Best if your GPU supports it (Ampere+, i.e. RTX 3000+)
      - "float16": Universal GPU support, but can have numerical issues
      - "float32": Most precise, but defeats the purpose of quantization

    BF16 > FP16 > FP32 in terms of speed, and BF16 >= FP32 > FP16 in
    terms of training stability (BF16 has FP32's exponent range).
    """

    bnb_4bit_use_double_quant: bool = True
    """Enable double quantization (quantize the quantization constants).

    This is a QLoRA innovation that further reduces memory:
      - Without: Each 64-weight block has one FP32 scaling constant = 0.5 bits/param overhead
      - With:    Scaling constants are quantized to INT8 = ~0.125 bits/param overhead

    Memory savings: ~0.4 GB for a 7B model, ~3 GB for a 65B model.
    Quality impact: Negligible. Always enable this.
    """

    # ─── LoRA Hyperparameters ───────────────────────────────────────
    # Same as LoRA — see LoRAConfig for detailed explanations.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # ─── Training Hyperparameters ───────────────────────────────────
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    weight_decay: float = 0.01
    output_dir: str = "./outputs/qlora_finetuned"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42

    # QLoRA training is usually done in BF16 for the compute dtype
    # (even though the base model is in 4-bit). If BF16 isn't available,
    # fall back to FP16.
    fp16: bool = False
    bf16: bool = False


class QLoRAFineTuner:
    """Fine-tune a language model using QLoRA (Quantized LoRA).

    QLoRA is the MOST MEMORY-EFFICIENT fine-tuning method available.
    It combines:
      1. 4-bit NF4 quantization of the base model (8x memory reduction)
      2. LoRA adapters in BF16/FP16 (only ~0.5% of parameters trained)
      3. Paged optimizers to handle memory spikes (uses CPU RAM as overflow)
      4. Double quantization (further reduces quantization overhead)

    The result: You can fine-tune a 7B model on a single consumer GPU
    with 8 GB of VRAM. The QLoRA paper showed this matches full 16-bit
    fine-tuning quality on many benchmarks.

    TYPICAL USAGE:
        >>> from agentexplorr.llm_training.fine_tune_qlora import (
        ...     QLoRAFineTuner, QLoRAConfig
        ... )
        >>> from agentexplorr.llm_training.data_preparation import (
        ...     DatasetPreparer, DatasetConfig
        ... )
        >>>
        >>> # Prepare data
        >>> preparer = DatasetPreparer(DatasetConfig(max_samples=1000))
        >>> processed = preparer.prepare("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>>
        >>> # Configure QLoRA (note: almost identical to LoRA!)
        >>> config = QLoRAConfig(
        ...     base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     load_in_4bit=True,               # <-- THE key difference
        ...     bnb_4bit_quant_type="nf4",        # NormalFloat4
        ...     bnb_4bit_use_double_quant=True,   # Quantize the quant constants
        ... )
        >>> tuner = QLoRAFineTuner(config)
        >>> tuner.train(processed.train_dataset, processed.val_dataset)
        >>> tuner.save_adapter("./my_qlora_adapter")

    DIFFERENCES FROM LoRA:
      1. Model loading: 4-bit quantization via BitsAndBytesConfig
      2. Model preparation: prepare_model_for_kbit_training() is required
      3. Optimizer: "paged_adamw_8bit" handles OOM gracefully
      4. Memory: ~3-4x less than standard LoRA

    LEARNING RESOURCES:
      - QLoRA paper: https://arxiv.org/abs/2305.14314
      - BitsAndBytes docs: https://huggingface.co/docs/bitsandbytes
      - VIDEO: "QLoRA Explained Simply" (AI Coffee Break):
        https://www.youtube.com/watch?v=y9PHWGOa8HA
      - VIDEO: "Fine-tune Llama 2 with QLoRA on Colab" (Maxime Labonne):
        https://www.youtube.com/watch?v=eeM6V5aPjhk
    """

    def __init__(self, config: QLoRAConfig | None = None) -> None:
        """Initialize the QLoRA fine-tuner.

        Args:
            config: QLoRA configuration. Uses defaults if not provided.
        """
        self.config = config or QLoRAConfig()
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.trainer: SFTTrainer | None = None

        # Hardware detection
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
            logger.info(
                "gpu_detected",
                device_name=gpu_name,
                vram_gb=vram_gb,
            )

            # Auto-detect BF16 support and set precision accordingly.
            # This makes the code "just work" on different GPUs.
            if torch.cuda.is_bf16_supported():
                self.config.bf16 = True
                self.config.fp16 = False
                logger.info("auto_enabled_bf16", reason="GPU supports BF16")
            else:
                self.config.bf16 = False
                self.config.fp16 = True
                logger.info("auto_enabled_fp16", reason="GPU does not support BF16")
        else:
            logger.warning(
                "no_gpu_detected",
                message="QLoRA REQUIRES a CUDA GPU. BitsAndBytes 4-bit "
                "quantization does not work on CPU. Please use Google "
                "Colab or another GPU environment.",
            )

        logger.info(
            "qlora_fine_tuner_initialized",
            base_model=self.config.base_model_name,
            load_in_4bit=self.config.load_in_4bit,
            quant_type=self.config.bnb_4bit_quant_type,
            double_quant=self.config.bnb_4bit_use_double_quant,
            lora_r=self.config.lora_r,
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> QLoRAFineTuner:
        """Create a QLoRAFineTuner from a YAML config file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Configured QLoRAFineTuner.
        """
        raw = load_yaml_config(config_path)
        config = QLoRAConfig(**raw)
        return cls(config)

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create the BitsAndBytes quantization configuration.

        THIS IS THE CORE OF QLoRA — this config tells Hugging Face HOW
        to quantize the base model weights from FP16/FP32 to 4-bit NF4.

        BitsAndBytesConfig is the bridge between the Hugging Face
        Transformers library and Tim Dettmers' bitsandbytes CUDA kernels
        that perform the actual quantization math on the GPU.

        HOW NF4 QUANTIZATION WORKS (simplified):
          1. Take a block of 64 weights (e.g., [0.123, -0.456, 0.789, ...])
          2. Compute the absolute max of the block (the scaling factor)
          3. Normalize all values to [-1, 1] range
          4. Map each value to the nearest NF4 quantization level
             (16 levels, optimally placed for a normal distribution)
          5. Store: 64 x 4-bit indices + 1 x FP32 scaling factor

          During forward pass: dequantize by reversing this process.
          The dequantized values won't be EXACTLY the originals, but
          they'll be close enough for the model to still work well.

        Returns:
            BitsAndBytesConfig for 4-bit quantization.
        """
        # Map string dtype names to torch dtypes.
        # We use a string in the config (YAML-friendly) and convert here.
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            self.config.bnb_4bit_compute_dtype,
            torch.bfloat16,  # Default to BF16 if unrecognized
        )

        bnb_config = BitsAndBytesConfig(
            # ─── Core quantization settings ─────────────────────────
            load_in_4bit=self.config.load_in_4bit,
            # NF4 quantization type — the star of the QLoRA paper.
            # Places quantization levels based on the assumption that
            # weights follow a normal distribution (which they empirically do).
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            # Compute dtype for forward/backward passes.
            # Weights are STORED in 4-bit but COMPUTED in this dtype.
            # The dequantization happens on-the-fly in the CUDA kernel.
            bnb_4bit_compute_dtype=compute_dtype,
            # Double quantization: quantize the quantization scaling factors.
            # Saves ~0.4 bits per parameter with negligible quality loss.
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        logger.info(
            "quantization_config_created",
            load_in_4bit=self.config.load_in_4bit,
            quant_type=self.config.bnb_4bit_quant_type,
            compute_dtype=str(compute_dtype),
            double_quant=self.config.bnb_4bit_use_double_quant,
        )

        return bnb_config

    def load_model(self) -> None:
        """Load the base model in 4-bit quantized precision.

        DIFFERENCE FROM LoRA:
          In standard LoRA (fine_tune_lora.py), we load the model in FP16/BF16.
          Here, we pass a BitsAndBytesConfig that tells Transformers to quantize
          the weights to 4-bit NF4 during loading.

          The quantization happens ON THE GPU — the weights are downloaded in
          their original precision and quantized as they're loaded onto the GPU.

        AFTER LOADING:
          We call prepare_model_for_kbit_training() which does several things:
            1. Casts LayerNorm weights to FP32 (they're sensitive to quantization)
            2. Enables gradient computation for the input embeddings
            3. Enables gradient checkpointing to save memory
            4. Sets up the model for stable 4-bit training

          Without this step, training would likely produce garbage results
          or crash with numerical errors.
        """
        logger.info(
            "loading_quantized_model",
            model=self.config.base_model_name,
            quant_type=self.config.bnb_4bit_quant_type,
        )

        # Load tokenizer (same as LoRA — tokenizer is not quantized)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Create the 4-bit quantization config
        bnb_config = self._create_quantization_config()

        # Load model WITH quantization.
        # The quantization_config parameter triggers BitsAndBytes to intercept
        # the weight loading process and quantize each linear layer's weights
        # to the specified format (NF4 in our case).
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # Note: We do NOT set torch_dtype here when using quantization.
            # The quantization_config handles the dtype management.
        )

        # Disable KV-cache for training (same as LoRA)
        self.model.config.use_cache = False

        # ─── CRITICAL STEP: Prepare for k-bit training ──────────────
        # This function from PEFT does essential setup for stable training
        # with quantized base models:
        #
        # 1. Upcasts LayerNorm layers to FP32:
        #    LayerNorm computes statistics (mean, variance) that are very
        #    sensitive to precision. In 4-bit, these statistics would be
        #    garbage, causing training instability. FP32 keeps them accurate.
        #
        # 2. Enables gradient computation for input embeddings:
        #    Even though we freeze most parameters, the input embeddings
        #    need gradients to flow through them to reach the LoRA layers.
        #
        # 3. Sets requires_grad=False on the quantized parameters:
        #    We don't want to update the 4-bit weights (they can't hold
        #    gradient-based updates anyway).
        self.model = prepare_model_for_kbit_training(self.model)

        # Report memory usage — this is where you see the QLoRA magic.
        # A 7B model that needs 14 GB in FP16 will use only ~4 GB here.
        total_params = sum(p.numel() for p in self.model.parameters())
        if self._device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "quantized_model_loaded",
                total_parameters=f"{total_params:,}",
                gpu_memory_allocated_gb=f"{memory_allocated:.2f}",
                gpu_memory_reserved_gb=f"{memory_reserved:.2f}",
                note="4-bit quantization reduces memory by ~4x vs FP16",
            )
        else:
            logger.info(
                "quantized_model_loaded",
                total_parameters=f"{total_params:,}",
            )

    def apply_qlora(self) -> None:
        """Apply LoRA adapters on top of the 4-bit quantized model.

        THE QLoRA ARCHITECTURE:
          ┌──────────────────────────────────────────────────────────┐
          │                                                          │
          │  Base Model Weights: 4-bit NF4 (FROZEN, NOT trainable)   │
          │       │                                                  │
          │       ├── LoRA-A matrix: BF16 (TRAINABLE)                │
          │       │       │                                          │
          │       │       └── LoRA-B matrix: BF16 (TRAINABLE)        │
          │       │               │                                  │
          │       └───────────────┘                                  │
          │               │                                          │
          │         y = W_4bit(x) + B(A(x))                          │
          │                                                          │
          │  Forward pass computation: BF16 / FP16                   │
          │  Gradients: Only for A and B matrices                    │
          │  Optimizer states: Only for A and B (small!)             │
          └──────────────────────────────────────────────────────────┘

          This is why QLoRA is so memory-efficient:
            - Base weights: 4-bit (vs 16-bit in LoRA) → ~4x savings
            - Adapter weights: Same as LoRA (~0.5% of model)
            - Optimizer states: Same as LoRA (only for adapter params)
            - Activations: Reduced by gradient checkpointing

        Raises:
            RuntimeError: If model hasn't been loaded yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        logger.info(
            "applying_qlora_adapters",
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )

        # Create LoRA config — IDENTICAL to standard LoRA.
        # The only difference is that the base model underneath is quantized.
        # PEFT handles the interaction between LoRA and quantization transparently.
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

        # Apply LoRA adapters to the quantized model
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameters — should be ~0.3-1% of total
        self.model.print_trainable_parameters()

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / all_params

        logger.info(
            "qlora_applied",
            trainable_params=f"{trainable_params:,}",
            all_params=f"{all_params:,}",
            trainable_pct=f"{trainable_pct:.2f}%",
            estimated_adapter_size_mb=f"{trainable_params * 2 / 1e6:.1f} MB",
        )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Run the QLoRA training loop.

        DIFFERENCES FROM LoRA TRAINING:
          1. Optimizer: We use "paged_adamw_8bit" instead of "adamw_torch".
             "paged" means the optimizer uses CPU RAM as overflow when GPU
             VRAM runs out (like virtual memory / swap). This prevents OOM
             crashes at the cost of slightly slower training.
             "8bit" means optimizer states (momentum, variance) are stored
             in INT8 instead of FP32, halving their memory usage.

          2. Gradient checkpointing: ESSENTIAL for QLoRA. Without it,
             storing activations for the backward pass would negate the
             memory savings from quantization.

          3. Mixed precision: The compute happens in BF16/FP16 (via
             bnb_4bit_compute_dtype), even though weights are stored in 4-bit.

        Args:
            train_dataset: Tokenized training dataset.
            val_dataset: Optional tokenized validation dataset.

        Returns:
            Dictionary of training metrics.

        Raises:
            RuntimeError: If model hasn't been loaded or QLoRA not applied.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() and apply_qlora() first."
            )

        effective_batch = (
            self.config.per_device_train_batch_size
            * self.config.gradient_accumulation_steps
        )

        logger.info(
            "qlora_training_starting",
            train_size=len(train_dataset),
            val_size=len(val_dataset) if val_dataset else 0,
            epochs=self.config.num_train_epochs,
            effective_batch_size=effective_batch,
            learning_rate=self.config.learning_rate,
            quantization=f"4-bit {self.config.bnb_4bit_quant_type}",
        )

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
            # Gradient checkpointing is CRITICAL for QLoRA.
            # It reduces activation memory from O(n_layers) to O(sqrt(n_layers))
            # by recomputing activations during the backward pass instead of
            # storing them all. The ~20% speed penalty is well worth the
            # ~60% memory savings.
            gradient_checkpointing=True,
            lr_scheduler_type="cosine",
            # "paged_adamw_8bit" is the RECOMMENDED optimizer for QLoRA:
            #   - "paged": Uses CPU RAM as overflow when GPU OOM occurs
            #     (like OS virtual memory). This means training might slow
            #     down instead of crashing when memory is tight.
            #   - "8bit": Optimizer states use INT8 instead of FP32.
            #     AdamW normally stores 2 FP32 values per parameter (momentum
            #     and variance). With 8-bit, these use ~1/4 the memory.
            #   Together, this reduces optimizer memory by ~75%.
            optim="paged_adamw_8bit",
            report_to="none",
            load_best_model_at_end=False,
            max_grad_norm=1.0,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        # Log GPU memory before training starts
        if self._device == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                "pre_training_gpu_memory",
                allocated_gb=f"{mem_gb:.2f}",
            )

        # ─── THE TRAINING LOOP ──────────────────────────────────────
        logger.info("qlora_training_loop_starting")
        train_result = self.trainer.train()

        metrics = train_result.metrics

        # Log GPU memory after training
        if self._device == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                "training_complete",
                train_loss=round(metrics.get("train_loss", 0.0), 4),
                train_runtime_sec=round(metrics.get("train_runtime", 0.0), 1),
                peak_gpu_memory_gb=f"{peak_mem_gb:.2f}",
            )
        else:
            logger.info(
                "training_complete",
                train_loss=round(metrics.get("train_loss", 0.0), 4),
            )

        # Evaluate on validation set
        if val_dataset is not None:
            eval_metrics = self.trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info(
                "evaluation_complete",
                eval_loss=round(eval_metrics.get("eval_loss", 0.0), 4),
            )

        return metrics

    def save_adapter(self, output_dir: str | Path | None = None) -> Path:
        """Save only the QLoRA adapter weights.

        IMPORTANT NOTE:
          QLoRA adapters are saved in the SAME format as LoRA adapters.
          The quantization config is NOT saved with the adapter — it's
          only needed for the base model. When loading the adapter for
          inference, you can choose to load the base model in whatever
          precision you want (4-bit for memory savings, FP16 for quality).

        What's saved:
          - adapter_model.safetensors: LoRA weight matrices (BF16/FP16)
          - adapter_config.json: LoRA hyperparameters

        To load for inference:
          >>> # Option A: Load base in 4-bit (memory-efficient inference)
          >>> model = AutoModelForCausalLM.from_pretrained(
          ...     "TinyLlama/...", quantization_config=bnb_config)
          >>> model = PeftModel.from_pretrained(model, "./my_qlora_adapter")
          >>>
          >>> # Option B: Load base in FP16 (higher quality inference)
          >>> model = AutoModelForCausalLM.from_pretrained(
          ...     "TinyLlama/...", torch_dtype=torch.float16)
          >>> model = PeftModel.from_pretrained(model, "./my_qlora_adapter")

        Args:
            output_dir: Where to save. Defaults to config.output_dir.

        Returns:
            Path to the saved adapter directory.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        save_path = Path(output_dir or self.config.output_dir) / "adapter"
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(
            "qlora_adapter_saved",
            path=str(save_path),
        )

        return save_path

    def merge_and_save(self, output_dir: str | Path | None = None) -> Path:
        """Merge QLoRA adapters into the base model and save.

        IMPORTANT CAVEAT FOR QLoRA:
          When merging a QLoRA adapter, the base model weights are
          DEQUANTIZED back to FP16/BF16 before merging. This means
          the merged model is in FP16 precision, NOT 4-bit.

          The merged model will be LARGER than the quantized version:
            - 4-bit TinyLlama: ~0.7 GB
            - Merged FP16 TinyLlama: ~2.2 GB

          This is fine — the merged model is meant for deployment, where
          you might want FP16 quality. You can always re-quantize the
          merged model later using tools like GPTQ, AWQ, or llama.cpp.

        Args:
            output_dir: Where to save the merged model.

        Returns:
            Path to the saved merged model.
        """
        if self.model is None:
            raise RuntimeError("No model to merge. Train first.")

        save_path = Path(output_dir or self.config.output_dir) / "merged"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "merging_qlora_adapter",
            note="Dequantizing base model to FP16 and merging adapter weights",
        )

        # merge_and_unload() dequantizes the 4-bit weights to BF16/FP16,
        # adds the LoRA update (B*A), and returns a standard model.
        merged_model = self.model.merge_and_unload()

        merged_model.save_pretrained(str(save_path))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(
            "qlora_merged_model_saved",
            path=str(save_path),
            note="Merged model is in FP16 (not 4-bit). "
            "Re-quantize with GPTQ/AWQ for deployment.",
        )

        return save_path

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from the QLoRA fine-tuned model.

        Generation works identically to LoRA (see fine_tune_lora.py).
        The only difference is that the forward pass uses 4-bit weights
        with on-the-fly dequantization, which is handled transparently
        by BitsAndBytes CUDA kernels.

        PERFORMANCE NOTE:
          4-bit inference is ~10-20% slower than FP16 inference due to
          the dequantization overhead. But it uses ~4x less memory, so
          you can run larger models or larger batch sizes.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0=deterministic, 1=random).
            top_p: Nucleus sampling threshold.

        Returns:
            Generated response text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                repetition_penalty=1.15,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()
