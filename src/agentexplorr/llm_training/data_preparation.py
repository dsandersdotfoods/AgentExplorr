"""
Dataset Preparation for LLM Fine-Tuning
=========================================

WHY DATA PREPARATION MATTERS:
  "Garbage in, garbage out" is the cardinal rule of ML. For LLM fine-tuning,
  data quality is even MORE important than data quantity. 500 high-quality,
  well-formatted examples will outperform 50,000 noisy ones.

  This module handles the full data pipeline:
    Raw Dataset → Instruction Formatting → Tokenization → Train/Val Split → Disk

THE INSTRUCTION FORMAT:
  Modern instruction-following LLMs are trained on data in this format:

    ### Instruction:
    Summarize the following article.

    ### Input:
    [Article text here...]

    ### Response:
    [Summary here...]

  The "Input" field is optional — some tasks (like "Write a poem about dogs")
  don't need one. This format was popularized by Stanford's Alpaca dataset and
  is now the de facto standard for instruction tuning.

WHY TOKENIZATION?
  LLMs don't see text — they see token IDs. The tokenizer converts:
    "Hello world" → [15496, 995]

  Each model has its OWN tokenizer with its own vocabulary. You MUST use the
  matching tokenizer for your base model. Using the wrong tokenizer would be
  like feeding Spanish text to a Japanese dictionary.

  Key tokenization concepts:
    - Padding: Adding [PAD] tokens so all sequences are the same length
      (required for batched GPU processing)
    - Truncation: Cutting sequences that exceed max_length
      (prevents OOM errors; GPUs have finite memory)
    - Attention Mask: Binary mask (1=real token, 0=padding) so the model
      ignores padding tokens during self-attention

LEARNING RESOURCES:
  - Hugging Face Datasets docs: https://huggingface.co/docs/datasets
  - Tokenizers deep-dive: https://huggingface.co/docs/transformers/tokenizer_summary
  - Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
  - VIDEO: "Tokenization Explained" (Andrej Karpathy):
    https://www.youtube.com/watch?v=zduSFxRajkE
  - VIDEO: "Prepare Data for Fine-Tuning" (Trelis Research):
    https://www.youtube.com/watch?v=XYsz6wDOdlY
  - VIDEO: "Hugging Face Datasets Tutorial" (James Briggs):
    https://www.youtube.com/watch?v=_BZearw7f0w
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from agentexplorr.core import get_logger

# ─── Module Logger ──────────────────────────────────────────────────────
logger = get_logger(__name__)


# ─── Data Classes ───────────────────────────────────────────────────────
# Using dataclasses for configuration is a clean pattern in ML code.
# They're like lightweight Pydantic models — less validation but simpler.


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing.

    Attributes:
        dataset_name: Hugging Face Hub dataset identifier.
            Examples: "tatsu-lab/alpaca", "databricks/databricks-dolly-15k",
                      "OpenAssistant/oasst1"
        dataset_split: Which split to load. Most HF datasets have "train".
        max_samples: Limit the number of samples (useful for testing/debugging).
            Set to None to use the full dataset.
        val_split_ratio: Fraction of data to hold out for validation.
            Typical values: 0.05-0.15 (5-15%). Too low = noisy val loss,
            too high = wasted training data.
        max_seq_length: Maximum number of tokens per example after tokenization.
            - 512: Good for short Q&A, classification
            - 1024: Good for instruction following, summarization
            - 2048: Good for longer generation tasks
            Rule of thumb: Set to the 95th percentile of your data's token lengths.
        seed: Random seed for reproducible train/val splits.
    """

    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    max_samples: int | None = None
    val_split_ratio: float = 0.1
    max_seq_length: int = 512
    seed: int = 42


@dataclass
class ProcessedDataset:
    """Container for a processed dataset ready for training.

    WHY A CONTAINER?
      Instead of returning a raw dict or tuple, we bundle everything the
      trainer needs into a single, well-documented object. This makes it
      impossible to mix up which dataset is train vs validation, or to
      forget the tokenizer.

    Attributes:
        train_dataset: Tokenized training examples.
        val_dataset: Tokenized validation examples.
        tokenizer: The tokenizer used (needed by the trainer for padding).
        raw_dataset: The unprocessed dataset (useful for inspection/debugging).
        config: The configuration used to create this dataset.
    """

    train_dataset: Dataset
    val_dataset: Dataset
    tokenizer: PreTrainedTokenizer
    raw_dataset: Dataset
    config: DatasetConfig


class DatasetPreparer:
    """End-to-end dataset preparation pipeline for LLM fine-tuning.

    This class handles the complete journey from a raw Hugging Face dataset
    to tokenized, split, GPU-ready data. It's designed for instruction-tuning
    datasets but can be adapted for other formats.

    TYPICAL USAGE:
        >>> from agentexplorr.llm_training.data_preparation import (
        ...     DatasetPreparer, DatasetConfig
        ... )
        >>>
        >>> # Step 1: Configure
        >>> config = DatasetConfig(
        ...     dataset_name="tatsu-lab/alpaca",
        ...     max_samples=1000,  # Start small for testing!
        ...     max_seq_length=512,
        ... )
        >>>
        >>> # Step 2: Prepare
        >>> preparer = DatasetPreparer(config)
        >>> processed = preparer.prepare(
        ...     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ... )
        >>>
        >>> # Step 3: Inspect
        >>> print(f"Training examples: {len(processed.train_dataset)}")
        >>> print(f"Validation examples: {len(processed.val_dataset)}")
        >>> print(processed.train_dataset[0])  # See a tokenized example

    ARCHITECTURE:
      ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
      │ Load from HF │ -> │ Format into  │ -> │ Tokenize   │ -> │ Train/   │
      │ Hub          │    │ Instruction  │    │ (pad+trunc) │    │ Val Split│
      └─────────────┘    │ Template     │    └────────────┘    └──────────┘
                         └──────────────┘

    LEARNING RESOURCES:
      - Alpaca format: https://github.com/tatsu-lab/stanford_alpaca#data-release
      - HF load_dataset: https://huggingface.co/docs/datasets/loading
      - VIDEO: "Fine-Tuning Data Prep" (Weights & Biases):
        https://www.youtube.com/watch?v=wPdq95ZiYjU
    """

    # ─── Prompt Templates ───────────────────────────────────────────
    # These templates match the Alpaca format. Most open-source instruction
    # datasets follow this convention, so you can swap datasets easily.
    #
    # The "with input" template is for tasks that have both an instruction
    # AND supporting context (e.g., "Summarize this: [text]").
    # The "without input" template is for standalone tasks
    # (e.g., "Write a haiku about rain").

    PROMPT_WITH_INPUT: str = (
        "Below is an instruction that describes a task, paired with further "
        "input that provides additional context. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    PROMPT_WITHOUT_INPUT: str = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    # Used at inference time (no output provided — the model generates it)
    INFERENCE_WITH_INPUT: str = (
        "Below is an instruction that describes a task, paired with further "
        "input that provides additional context. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    INFERENCE_WITHOUT_INPUT: str = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Initialize the DatasetPreparer.

        Args:
            config: Dataset configuration. If None, uses sensible defaults.
        """
        self.config = config or DatasetConfig()
        logger.info(
            "dataset_preparer_initialized",
            dataset=self.config.dataset_name,
            max_samples=self.config.max_samples,
            max_seq_length=self.config.max_seq_length,
        )

    # ─── Public API ─────────────────────────────────────────────────

    def format_instruction(self, example: dict[str, str]) -> str:
        """Format a single instruction example into the Alpaca template.

        Args:
            example: Dictionary with "instruction", optional "input", and "output" keys.

        Returns:
            Formatted prompt string with ### Instruction:, optional ### Input:,
            and ### Response: sections.
        """
        if example.get("input", "").strip():
            return self.PROMPT_WITH_INPUT.format(
                instruction=example["instruction"].strip(),
                input=example["input"].strip(),
                output=example["output"].strip(),
            )
        return self.PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"].strip(),
            output=example["output"].strip(),
        )

    def prepare(self, model_name: str) -> ProcessedDataset:
        """Run the full data preparation pipeline.

        This is the main entry point. It orchestrates:
          1. Loading the raw dataset from Hugging Face Hub
          2. Formatting each example into the instruction template
          3. Tokenizing with the model's tokenizer
          4. Splitting into train/validation sets

        Args:
            model_name: Hugging Face model identifier. The tokenizer for this
                model will be loaded automatically. Must match the model you
                plan to fine-tune — using a mismatched tokenizer will produce
                garbage results.
                Examples:
                  - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B, great for learning)
                  - "mistralai/Mistral-7B-v0.1" (7B, production quality)
                  - "meta-llama/Llama-2-7b-hf" (7B, requires HF token)

        Returns:
            ProcessedDataset containing train/val splits, tokenizer, and raw data.

        Raises:
            ValueError: If the dataset doesn't have expected columns.
            ConnectionError: If Hugging Face Hub is unreachable.
        """
        logger.info("prepare_pipeline_starting", model_name=model_name)

        # Step 1: Load raw data
        raw_dataset = self.load_dataset()

        # Step 2: Format into instruction template
        formatted_dataset = self.format_dataset(raw_dataset)

        # Step 3: Load the tokenizer for the target model
        tokenizer = self._load_tokenizer(model_name)

        # Step 4: Tokenize
        tokenized_dataset = self.tokenize_dataset(formatted_dataset, tokenizer)

        # Step 5: Train/validation split
        splits = self.split_dataset(tokenized_dataset)

        logger.info(
            "prepare_pipeline_complete",
            train_size=len(splits["train"]),
            val_size=len(splits["validation"]),
        )

        return ProcessedDataset(
            train_dataset=splits["train"],
            val_dataset=splits["validation"],
            tokenizer=tokenizer,
            raw_dataset=raw_dataset,
            config=self.config,
        )

    def load_dataset(self) -> Dataset:
        """Load a dataset from the Hugging Face Hub.

        HOW THIS WORKS:
          The `load_dataset` function from the `datasets` library downloads
          (and caches!) datasets from https://huggingface.co/datasets.
          After the first download, subsequent calls load from disk cache.

          The cache is typically stored in ~/.cache/huggingface/datasets/.
          You can change this with the HF_DATASETS_CACHE env var.

        POPULAR INSTRUCTION DATASETS (all free and open):
          - tatsu-lab/alpaca       (52K examples, Stanford)
          - databricks/dolly-15k   (15K examples, Databricks)
          - OpenAssistant/oasst1   (Multi-turn conversations)
          - yahma/alpaca-cleaned   (Cleaned version of Alpaca)
          - timdettmers/openassistant-guanaco (Subset of OASST, used in QLoRA paper)

        Returns:
            The loaded dataset (before formatting/tokenization).

        Raises:
            ConnectionError: If the dataset can't be downloaded.
        """
        logger.info(
            "loading_dataset",
            name=self.config.dataset_name,
            split=self.config.dataset_split,
        )

        # load_dataset returns a DatasetDict (keyed by split) unless you specify
        # a split. We specify the split to get a flat Dataset object.
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        )

        # Subsample if max_samples is set — essential for rapid prototyping.
        # ALWAYS test your pipeline with a small subset first (e.g., 100-500
        # examples) before running on the full dataset. This catches bugs
        # in minutes instead of hours.
        if self.config.max_samples is not None:
            # .select() picks specific indices — much more efficient than
            # slicing because it doesn't copy data.
            n_samples = min(self.config.max_samples, len(dataset))
            dataset = dataset.select(range(n_samples))
            logger.info("dataset_subsampled", n_samples=n_samples)

        logger.info(
            "dataset_loaded",
            num_examples=len(dataset),
            columns=dataset.column_names,
        )

        return dataset

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Format each example into the instruction-following template.

        WHY FORMAT?
          Raw datasets have different column names and structures. The Alpaca
          dataset has columns: instruction, input, output. Other datasets might
          have: prompt, context, response. This function normalizes everything
          into a single "text" column with a consistent prompt template.

          The "text" column is what the tokenizer and trainer expect.

        IMPORTANT: The template includes BOTH the prompt AND the response.
          During training, the model learns to predict the response tokens
          given the instruction tokens. At inference time, you provide only
          the instruction and let the model generate the response.

        Args:
            dataset: Raw dataset with instruction/input/output columns.

        Returns:
            Dataset with an added "text" column containing formatted prompts.
        """
        logger.info("formatting_dataset", num_examples=len(dataset))

        def _format_example(example: dict[str, Any]) -> dict[str, str]:
            """Format a single example into the instruction template.

            The Alpaca format has three fields:
              - instruction: The task description (always present)
              - input: Optional additional context
              - output: The expected response (always present)

            We choose the template based on whether 'input' is present and
            non-empty. This is critical — using the wrong template would
            confuse the model about when to expect input context.
            """
            # Check if this example has additional input context.
            # Some examples have input="" (empty string), which we treat
            # as "no input" — otherwise the prompt would have an empty
            # "### Input:\n\n" section, which looks broken.
            if example.get("input", "").strip():
                text = DatasetPreparer.PROMPT_WITH_INPUT.format(
                    instruction=example["instruction"].strip(),
                    input=example["input"].strip(),
                    output=example["output"].strip(),
                )
            else:
                text = DatasetPreparer.PROMPT_WITHOUT_INPUT.format(
                    instruction=example["instruction"].strip(),
                    output=example["output"].strip(),
                )
            return {"text": text}

        # .map() applies a function to every example in the dataset.
        # It's lazy and memory-efficient — it doesn't load the whole
        # dataset into memory at once. Think of it like Python's map()
        # but optimized for tabular ML data.
        formatted = dataset.map(
            _format_example,
            # remove_columns removes the original columns after mapping.
            # We keep them for now so users can inspect the raw data.
            desc="Formatting examples into instruction template",
        )

        # Log a sample so users can verify the formatting looks correct
        if len(formatted) > 0:
            sample_text = formatted[0]["text"]
            logger.info(
                "format_sample",
                first_200_chars=sample_text[:200] + "...",
            )

        return formatted

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
    ) -> Dataset:
        """Tokenize the formatted dataset for model consumption.

        WHAT TOKENIZATION DOES:
          Text (strings) → Token IDs (integers) → Model Input

          "Hello world" → tokenizer → {"input_ids": [15496, 995],
                                        "attention_mask": [1, 1]}

        WHY PADDING AND TRUNCATION?
          GPUs process data in batches. All sequences in a batch must have
          the same length. We achieve this by:
            - Padding short sequences with [PAD] tokens (usually id=0)
            - Truncating long sequences to max_seq_length

          The attention_mask tells the model which tokens are real (1) vs
          padding (0), so padding doesn't corrupt the model's attention.

        LABELS:
          For causal language modeling (next-token prediction), labels are
          identical to input_ids but shifted right by 1 position. The
          Hugging Face Trainer handles this shift internally — we just need
          to provide the labels field.

          input_ids:  [Hello] [world] [how]  [are] [you]
          labels:     [world] [how]   [are]  [you] [EOS]  ← shifted by 1

          The model learns: "Given [Hello], predict [world]", etc.

        Args:
            dataset: Formatted dataset with a "text" column.
            tokenizer: The pretrained tokenizer matching your base model.

        Returns:
            Tokenized dataset ready for the Trainer.
        """
        logger.info(
            "tokenizing_dataset",
            max_seq_length=self.config.max_seq_length,
            tokenizer_vocab_size=tokenizer.vocab_size,
        )

        def _tokenize(examples: dict[str, list[str]]) -> dict[str, list[Any]]:
            """Tokenize a batch of examples.

            We use batched=True in .map() below, so `examples["text"]` is a
            LIST of strings, not a single string. This is much more efficient
            because the tokenizer can process multiple texts in parallel.

            Key tokenizer arguments:
              - truncation=True: Cut sequences longer than max_length
              - max_length: Maximum number of tokens (not characters!)
              - padding="max_length": Pad all sequences to max_length
                (alternative: padding="longest" pads to the longest in
                the batch, which wastes less memory but requires dynamic
                batch padding in the DataLoader)
            """
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                # return_tensors is NOT set here. When using .map(), we
                # return Python lists. The Trainer converts to tensors later.
                # Setting return_tensors="pt" here would cause issues with
                # the datasets library's Arrow format.
            )

            # For causal LM fine-tuning, labels = input_ids.
            # The model internally shifts labels by 1 to create the
            # next-token prediction task. We also set padding token
            # labels to -100, which tells PyTorch's CrossEntropyLoss
            # to IGNORE those positions (don't penalize the model for
            # failing to predict padding tokens).
            labels = []
            for input_ids, attention_mask in zip(
                tokenized["input_ids"],
                tokenized["attention_mask"],
            ):
                # Replace padding token IDs with -100 in labels.
                # -100 is the magic "ignore" value in PyTorch's
                # cross_entropy loss function. Any label set to -100
                # is excluded from the loss computation.
                label = [
                    token_id if mask == 1 else -100
                    for token_id, mask in zip(input_ids, attention_mask)
                ]
                labels.append(label)

            tokenized["labels"] = labels
            return tokenized

        # batched=True means _tokenize receives lists of examples
        # instead of one example at a time. This is ~10x faster.
        tokenized = dataset.map(
            _tokenize,
            batched=True,
            # Remove the text column — the model only needs token IDs.
            # Keeping it would waste memory during training.
            remove_columns=["text"],
            desc="Tokenizing examples",
        )

        logger.info(
            "tokenization_complete",
            num_examples=len(tokenized),
            columns=tokenized.column_names,
        )

        return tokenized

    def split_dataset(
        self,
        dataset: Dataset,
    ) -> DatasetDict:
        """Split the dataset into training and validation sets.

        WHY SPLIT?
          You need a held-out validation set to detect overfitting. If your
          training loss keeps going down but validation loss starts going UP,
          you're overfitting — the model is memorizing training examples
          instead of learning generalizable patterns.

          ┌────────────────────────────────────────────────────────────┐
          │  Training Loss ↓↓↓   Val Loss ↓↓↑↑↑  → OVERFITTING!     │
          │  Training Loss ↓↓↓   Val Loss ↓↓↓↓↓  → Still learning   │
          │  Training Loss →→→   Val Loss →→→→→  → Converged        │
          └────────────────────────────────────────────────────────────┘

        CHOOSING SPLIT RATIO:
          - 0.05 (5%): Large datasets (>50K examples)
          - 0.10 (10%): Medium datasets (5K-50K examples)
          - 0.15-0.20 (15-20%): Small datasets (<5K examples)

        Args:
            dataset: The tokenized dataset to split.

        Returns:
            DatasetDict with "train" and "validation" keys.
        """
        logger.info(
            "splitting_dataset",
            total_examples=len(dataset),
            val_ratio=self.config.val_split_ratio,
        )

        # train_test_split is deterministic with the same seed, so your
        # experiments are reproducible. ALWAYS set a seed for ML experiments!
        split = dataset.train_test_split(
            test_size=self.config.val_split_ratio,
            seed=self.config.seed,
        )

        # Rename "test" to "validation" for clarity.
        # HF's train_test_split calls the held-out set "test", but in the
        # context of fine-tuning, this is really our validation set (we
        # monitor it during training). A true "test" set would be completely
        # separate data you never look at until final evaluation.
        result = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

        logger.info(
            "split_complete",
            train_size=len(result["train"]),
            val_size=len(result["validation"]),
        )

        return result

    def save_processed_dataset(
        self,
        processed: ProcessedDataset,
        output_dir: str | Path,
    ) -> Path:
        """Save the processed dataset to disk for reuse.

        WHY SAVE TO DISK?
          Tokenization is CPU-intensive and can take minutes for large
          datasets. Saving the processed dataset means you only pay this
          cost once. On subsequent training runs, you can load the
          pre-tokenized data instantly.

          This is especially important when experimenting with different
          training hyperparameters — you want fast iteration on the
          training loop, not re-tokenization every time.

        The saved format is Apache Arrow, which is:
          - Memory-mapped (loads instantly regardless of size)
          - Columnar (efficient for ML operations)
          - Portable (works across Python, R, Rust, etc.)

        Args:
            processed: The ProcessedDataset to save.
            output_dir: Directory to save into. Will be created if needed.

        Returns:
            Path to the saved dataset directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save train and validation splits separately
        train_path = output_path / "train"
        val_path = output_path / "validation"

        processed.train_dataset.save_to_disk(str(train_path))
        processed.val_dataset.save_to_disk(str(val_path))

        # Also save the tokenizer so it travels with the data.
        # This ensures you always use the correct tokenizer when
        # loading the processed dataset later.
        tokenizer_path = output_path / "tokenizer"
        processed.tokenizer.save_pretrained(str(tokenizer_path))

        logger.info(
            "dataset_saved_to_disk",
            output_dir=str(output_path),
            train_size=len(processed.train_dataset),
            val_size=len(processed.val_dataset),
        )

        return output_path

    @staticmethod
    def load_processed_dataset(
        dataset_dir: str | Path,
    ) -> tuple[Dataset, Dataset, PreTrainedTokenizer]:
        """Load a previously saved processed dataset from disk.

        This is the counterpart to save_processed_dataset(). Use it to
        skip the tokenization step on subsequent training runs.

        Args:
            dataset_dir: Path to the directory saved by save_processed_dataset.

        Returns:
            Tuple of (train_dataset, val_dataset, tokenizer).

        Raises:
            FileNotFoundError: If the dataset directory doesn't exist.
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at: {dataset_path}")

        train_dataset = Dataset.load_from_disk(str(dataset_path / "train"))
        val_dataset = Dataset.load_from_disk(str(dataset_path / "validation"))
        tokenizer = AutoTokenizer.from_pretrained(str(dataset_path / "tokenizer"))

        logger.info(
            "processed_dataset_loaded",
            train_size=len(train_dataset),
            val_size=len(val_dataset),
        )

        return train_dataset, val_dataset, tokenizer

    @staticmethod
    def format_for_inference(
        instruction: str,
        input_text: str = "",
    ) -> str:
        """Format a single instruction for inference (no output template).

        Use this when generating predictions with your fine-tuned model.
        The template includes everything UP TO "### Response:\\n" — the
        model generates everything after that.

        Args:
            instruction: The task instruction.
            input_text: Optional additional context.

        Returns:
            Formatted prompt string ready for model.generate().

        Example:
            >>> prompt = DatasetPreparer.format_for_inference(
            ...     instruction="Translate to French",
            ...     input_text="Hello, how are you?"
            ... )
            >>> outputs = model.generate(tokenizer(prompt, return_tensors="pt"))
        """
        if input_text.strip():
            return DatasetPreparer.INFERENCE_WITH_INPUT.format(
                instruction=instruction.strip(),
                input=input_text.strip(),
            )
        return DatasetPreparer.INFERENCE_WITHOUT_INPUT.format(
            instruction=instruction.strip(),
        )

    # ─── Private Helpers ────────────────────────────────────────────

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load the tokenizer for the specified model.

        IMPORTANT TOKENIZER DETAILS:
          - AutoTokenizer automatically selects the correct tokenizer class
            based on the model config. You don't need to know if it's a
            LlamaTokenizer, GPT2Tokenizer, etc.

          - pad_token: Many decoder-only models (GPT, Llama) don't have a
            padding token because they were pre-trained without padding.
            We set pad_token = eos_token as a common workaround.
            This is safe because the attention mask ensures padding is ignored.

          - padding_side="right": For decoder-only models (causal LMs), we
            pad on the RIGHT. This is because the model generates tokens
            left-to-right, and left-padding would shift all positions.
            (Note: For batch generation/inference, LEFT padding is preferred,
            but for training RIGHT padding is standard.)

        Args:
            model_name: Hugging Face model identifier.

        Returns:
            The configured tokenizer.
        """
        logger.info("loading_tokenizer", model_name=model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # trust_remote_code=True allows loading tokenizers with custom
            # code. Some models (e.g., Phi, Qwen) need this. It's a security
            # consideration — only use with models from trusted sources.
            trust_remote_code=True,
        )

        # Many causal LM tokenizers don't have a pad token.
        # Without one, the tokenizer can't pad sequences, and batched
        # training fails. Using eos_token as pad_token is standard practice.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(
                "pad_token_set_to_eos",
                pad_token=tokenizer.pad_token,
            )

        # For training, pad on the right side.
        # For generation/inference, you'd want padding_side="left".
        tokenizer.padding_side = "right"

        logger.info(
            "tokenizer_loaded",
            vocab_size=tokenizer.vocab_size,
            model_max_length=tokenizer.model_max_length,
            pad_token=tokenizer.pad_token,
        )

        return tokenizer
