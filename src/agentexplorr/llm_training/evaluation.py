"""
Model Evaluation for Fine-Tuned LLMs
======================================

WHY EVALUATION MATTERS:
  Training loss going down doesn't mean your model is actually good.
  You need to evaluate on HELD-OUT data using MULTIPLE metrics to get
  a holistic picture of model quality.

  Think of it like grading a student:
    - Training loss = homework score (they might be memorizing)
    - Perplexity = multiple-choice exam (tests knowledge breadth)
    - BLEU/ROUGE = essay grading (tests generation quality)
    - Qualitative eval = oral examination (tests practical ability)

EVALUATION METRICS EXPLAINED:

  1. PERPLEXITY (lower is better)
     ─────────────────────────────
     "How surprised is the model by the test data?"

     Perplexity = exp(average cross-entropy loss)

     Intuition: If the model assigns high probability to the correct next
     token, cross-entropy is low, and perplexity is low. A perplexity of 10
     means the model is "as confused as if it had to choose uniformly among
     10 equally likely tokens at each step."

     Typical ranges:
       - 5-15:  Excellent (the model knows this domain well)
       - 15-50: Good (reasonable performance)
       - 50-100: Mediocre (significant uncertainty)
       - >100:  Poor (the model is very confused)

  2. BLEU SCORE (higher is better, 0-1)
     ────────────────────────────────────
     "How much does the generated text overlap with reference text?"

     BLEU (Bilingual Evaluation Understudy) counts matching n-grams
     between the generated text and reference text. Originally designed
     for machine translation, it's useful for any task with expected outputs.

     How it works:
       Reference: "The cat sat on the mat"
       Generated: "The cat is on the mat"
       Unigram matches: "The", "cat", "on", "the", "mat" (5/6 = 0.83)
       Bigram matches: "The cat", "on the", "the mat" (3/5 = 0.60)
       BLEU-4 = geometric mean of 1-gram through 4-gram precision

     Limitations:
       - Doesn't capture synonyms ("happy" vs "glad" = 0 match)
       - Doesn't capture meaning ("I love cats" vs "Felines are my passion")
       - Can be gamed by generating many common words

  3. ROUGE SCORE (higher is better, 0-1)
     ─────────────────────────────────────
     "How well does the generated text RECALL the reference content?"

     ROUGE (Recall-Oriented Understudy for Gisting Evaluation) focuses on
     RECALL: what fraction of the reference n-grams appear in the output?
     (BLEU focuses on PRECISION: what fraction of output n-grams match?)

     Variants:
       - ROUGE-1: Unigram overlap (individual words)
       - ROUGE-2: Bigram overlap (word pairs — captures local structure)
       - ROUGE-L: Longest Common Subsequence (captures sentence structure)

     ROUGE is the standard metric for summarization tasks.

  4. QUALITATIVE EVALUATION
     ───────────────────────
     Numbers don't tell the whole story. ALWAYS generate sample outputs
     and READ them. A model with perfect BLEU but incoherent responses
     is worse than one with lower BLEU but clear, useful answers.

LEARNING RESOURCES:
  - Perplexity explained: https://huggingface.co/docs/transformers/perplexity
  - BLEU paper (Papineni et al., 2002): https://aclanthology.org/P02-1040/
  - ROUGE paper (Lin, 2004): https://aclanthology.org/W04-1013/
  - VIDEO: "NLP Evaluation Metrics Explained" (Weights & Biases):
    https://www.youtube.com/watch?v=TMshhnrEXlg
  - VIDEO: "Perplexity in 5 Minutes" (ritvikmath):
    https://www.youtube.com/watch?v=NURcDHhYe98
  - VIDEO: "BLEU Score — How to Evaluate Machine Translation" (StatQuest):
    https://www.youtube.com/watch?v=M05L1DhFqcw
  - BLOG: "How to evaluate LLMs" (Hugging Face):
    https://huggingface.co/blog/evaluating-llm-models
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from agentexplorr.core import get_logger

# ─── Module Logger ──────────────────────────────────────────────────────
logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results across all metrics.

    Having a structured result object makes it easy to:
      - Log results consistently
      - Compare multiple models
      - Save results to disk (JSON/YAML)
      - Display in a dashboard

    Attributes:
        perplexity: Model perplexity on the test set (lower = better).
        bleu_scores: BLEU scores at different n-gram levels.
        rouge_scores: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        sample_outputs: Generated text samples for qualitative review.
        metadata: Additional information (model name, dataset, etc.).
    """

    perplexity: float | None = None
    bleu_scores: dict[str, float] = field(default_factory=dict)
    rouge_scores: dict[str, float] = field(default_factory=dict)
    sample_outputs: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of all evaluation results.

        Returns:
            Formatted string with all metrics and sample outputs.
        """
        lines = [
            "=" * 70,
            "MODEL EVALUATION RESULTS",
            "=" * 70,
        ]

        # Metadata
        if self.metadata:
            lines.append("\n--- Metadata ---")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        # Perplexity
        if self.perplexity is not None:
            lines.append("\n--- Perplexity ---")
            lines.append(f"  Perplexity: {self.perplexity:.2f}")
            # Add an interpretive guide
            if self.perplexity < 15:
                lines.append("  Interpretation: Excellent")
            elif self.perplexity < 50:
                lines.append("  Interpretation: Good")
            elif self.perplexity < 100:
                lines.append("  Interpretation: Mediocre")
            else:
                lines.append("  Interpretation: Poor")

        # BLEU Scores
        if self.bleu_scores:
            lines.append("\n--- BLEU Scores ---")
            for key, value in self.bleu_scores.items():
                lines.append(f"  {key}: {value:.4f}")

        # ROUGE Scores
        if self.rouge_scores:
            lines.append("\n--- ROUGE Scores ---")
            for key, value in self.rouge_scores.items():
                lines.append(f"  {key}: {value:.4f}")

        # Sample Outputs
        if self.sample_outputs:
            lines.append(f"\n--- Sample Outputs ({len(self.sample_outputs)}) ---")
            for i, sample in enumerate(self.sample_outputs, 1):
                lines.append(f"\n  Sample {i}:")
                lines.append(f"  Instruction: {sample.get('instruction', 'N/A')}")
                if sample.get("input"):
                    lines.append(f"  Input: {sample['input'][:100]}...")
                lines.append(f"  Expected: {sample.get('expected', 'N/A')[:200]}")
                lines.append(f"  Generated: {sample.get('generated', 'N/A')[:200]}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluate fine-tuned language models using multiple metrics.

    This class provides a comprehensive evaluation suite for comparing
    base models vs fine-tuned models. It supports:
      - Perplexity (intrinsic quality metric)
      - BLEU score (n-gram precision for translation/generation)
      - ROUGE score (recall-oriented for summarization)
      - Qualitative sample generation

    EVALUATION WORKFLOW:
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │ Load Base     │    │ Load Fine-   │    │ Compare      │
      │ Model         │    │ Tuned Model  │    │ Results      │
      └──────┬───────┘    └──────┬───────┘    └──────────────┘
             │                    │                     ↑
             └────── Evaluate ────┘─────── Results ────┘

    TYPICAL USAGE:
        >>> from agentexplorr.llm_training.evaluation import ModelEvaluator
        >>>
        >>> evaluator = ModelEvaluator()
        >>>
        >>> # Load models
        >>> evaluator.load_model(
        ...     model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     label="base"
        ... )
        >>> evaluator.load_model(
        ...     model_name_or_path="./outputs/lora_finetuned/merged",
        ...     label="finetuned"
        ... )
        >>>
        >>> # Run evaluation
        >>> base_result = evaluator.evaluate(test_dataset, model_label="base")
        >>> ft_result = evaluator.evaluate(test_dataset, model_label="finetuned")
        >>>
        >>> # Compare
        >>> comparison = evaluator.compare(base_result, ft_result)
        >>> print(comparison)

    LEARNING RESOURCES:
      - HF evaluate library: https://huggingface.co/docs/evaluate
      - LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
      - VIDEO: "How to Evaluate LLMs" (Hugging Face):
        https://www.youtube.com/watch?v=_gOquJoSeLY
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        # Store multiple models for comparison
        self._models: dict[str, PreTrainedModel] = {}
        self._tokenizers: dict[str, PreTrainedTokenizer] = {}

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "model_evaluator_initialized",
            device=self._device,
        )

    def load_model(
        self,
        model_name_or_path: str,
        label: str = "default",
        load_in_4bit: bool = False,
    ) -> None:
        """Load a model for evaluation.

        You can load multiple models with different labels for comparison.
        This is the standard workflow for comparing base vs fine-tuned models.

        Args:
            model_name_or_path: Hugging Face model ID or local path.
                Examples:
                  - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (base model)
                  - "./outputs/lora_finetuned/merged" (your fine-tuned model)
            label: A human-readable label for this model (e.g., "base", "finetuned").
                Used to identify models in comparison results.
            load_in_4bit: Whether to load in 4-bit quantization (saves memory
                for evaluation of large models).
        """
        logger.info(
            "loading_model_for_evaluation",
            model=model_name_or_path,
            label=label,
            load_in_4bit=load_in_4bit,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model with optional quantization
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if self._device == "cuda" else None,
        }

        if load_in_4bit:
            # Import here to avoid requiring bitsandbytes for non-quantized eval
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            if self._device == "cuda":
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                dtype = torch.float32
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )

        # Set to evaluation mode (disables dropout, etc.)
        model.eval()

        self._models[label] = model
        self._tokenizers[label] = tokenizer

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "model_loaded_for_evaluation",
            label=label,
            parameters=f"{total_params:,}",
        )

    def calculate_perplexity(
        self,
        dataset: Dataset,
        model_label: str = "default",
        max_samples: int | None = 100,
        max_length: int = 512,
    ) -> float:
        """Calculate perplexity of a model on a dataset.

        PERPLEXITY DEEP DIVE:
          Perplexity = exp(H), where H is the average cross-entropy loss.

          Cross-entropy loss measures how well the model's predicted
          probability distribution matches the actual next token.

          If the model assigns probability p to the correct next token:
            - p = 1.0 → loss = 0, perplexity = 1 (perfect prediction!)
            - p = 0.5 → loss = 0.69, perplexity = 2
            - p = 0.1 → loss = 2.30, perplexity = 10
            - p = 0.01 → loss = 4.61, perplexity = 100

          Perplexity literally means "how many equally likely options the
          model thinks there are." A perplexity of 10 means the model is
          as uncertain as if it always had 10 equally likely next tokens.

        IMPORTANT CAVEATS:
          - Perplexity depends on the tokenizer. You CAN'T compare perplexity
            between models with different tokenizers (different vocabularies).
          - Perplexity on the training data is NOT useful (will be artificially
            low due to memorization). Always use held-out data.
          - Perplexity doesn't capture everything. A model can have low
            perplexity but still generate incoherent or harmful text.

        Args:
            dataset: Dataset with a "text" column (or "input_ids" if tokenized).
            model_label: Which loaded model to evaluate.
            max_samples: Limit evaluation to this many samples (for speed).
                Set to None to evaluate on the full dataset.
            max_length: Maximum sequence length for evaluation.

        Returns:
            Perplexity score (float). Lower is better.

        Raises:
            KeyError: If model_label doesn't match any loaded model.
        """
        if model_label not in self._models:
            raise KeyError(
                f"Model '{model_label}' not loaded. Available: "
                f"{list(self._models.keys())}. Call load_model() first."
            )

        model = self._models[model_label]
        tokenizer = self._tokenizers[model_label]

        logger.info(
            "calculating_perplexity",
            model_label=model_label,
            dataset_size=len(dataset),
            max_samples=max_samples,
        )

        # Subsample if needed
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        total_loss = 0.0
        total_tokens = 0

        # Process each example individually to handle variable lengths.
        # For production, you'd batch this for efficiency, but for
        # educational clarity we process one at a time.
        for i in range(len(dataset)):
            example = dataset[i]

            # Handle both tokenized (has input_ids) and raw (has text) datasets.
            if "input_ids" in example:
                # Already tokenized — use directly
                input_ids = torch.tensor(
                    example["input_ids"][:max_length],
                    dtype=torch.long,
                ).unsqueeze(0)
            elif "text" in example:
                # Raw text — tokenize it
                tokens = tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"]
            else:
                # Skip examples without text or input_ids
                continue

            # Move to the model's device
            input_ids = input_ids.to(model.device)

            # Skip very short sequences (< 2 tokens can't compute next-token loss)
            if input_ids.shape[1] < 2:
                continue

            # Compute loss WITHOUT gradient computation (saves memory + speed)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,  # For causal LM, labels = input_ids
                )
                # outputs.loss is the average cross-entropy loss over all
                # positions in this sequence (excluding padding).
                loss = outputs.loss

            # Accumulate for computing average loss across the whole dataset.
            # We weight by the number of tokens in this sequence so that
            # longer sequences contribute more to the average (proportionally
            # to how much information they contain).
            n_tokens = input_ids.shape[1] - 1  # -1 because loss is computed on shifted tokens
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        if total_tokens == 0:
            logger.warning("no_valid_tokens_for_perplexity")
            return float("inf")

        # Average cross-entropy loss across all tokens
        avg_loss = total_loss / total_tokens

        # Perplexity = exp(average cross-entropy loss)
        # We clamp to prevent overflow for very bad models
        perplexity = math.exp(min(avg_loss, 100.0))

        logger.info(
            "perplexity_calculated",
            model_label=model_label,
            perplexity=round(perplexity, 2),
            avg_loss=round(avg_loss, 4),
            total_tokens=total_tokens,
        )

        return perplexity

    def calculate_bleu(
        self,
        references: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Calculate BLEU scores between references and predictions.

        BLEU SCORE COMPUTATION:
          BLEU computes precision at different n-gram levels and combines
          them with a brevity penalty.

          For each n (1, 2, 3, 4):
            precision_n = (matching n-grams in prediction) / (total n-grams in prediction)

          BLEU = BP * exp(sum of log(precision_n) for n=1..4) / 4)

          Where BP (Brevity Penalty) penalizes translations shorter than
          the reference. Without it, a model could game BLEU by outputting
          a single high-confidence word.

        We use the `evaluate` library (Hugging Face) which implements
        the standard BLEU calculation from Papineni et al. (2002).

        Args:
            references: List of reference/expected texts.
            predictions: List of generated/predicted texts.
                Must be the same length as references.

        Returns:
            Dictionary with BLEU scores:
              - "bleu": Overall BLEU-4 score (0-1)
              - "bleu_1": Unigram precision
              - "bleu_2": Bigram precision
              - "bleu_3": Trigram precision
              - "bleu_4": 4-gram precision
              - "brevity_penalty": The brevity penalty factor

        Raises:
            ValueError: If references and predictions have different lengths.
        """
        if len(references) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs "
                f"{len(predictions)} predictions"
            )

        logger.info(
            "calculating_bleu",
            num_examples=len(references),
        )

        # We use the `evaluate` library for standardized metric computation.
        # It implements the exact same algorithm as the original BLEU paper.
        # If evaluate is not installed, fall back to a simple implementation.
        try:
            import evaluate

            bleu_metric = evaluate.load("bleu")

            # The evaluate library expects references as a list of lists
            # (each reference can have multiple valid translations).
            # For simplicity, we have one reference per example.
            results = bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references],
            )

            scores = {
                "bleu": results["bleu"],
                "bleu_1": results["precisions"][0],
                "bleu_2": results["precisions"][1],
                "bleu_3": results["precisions"][2],
                "bleu_4": results["precisions"][3],
                "brevity_penalty": results["brevity_penalty"],
            }

        except ImportError:
            logger.warning(
                "evaluate_not_installed",
                message="Install with: pip install evaluate. "
                "Falling back to simple BLEU approximation.",
            )
            scores = self._simple_bleu(references, predictions)

        logger.info("bleu_calculated", **{k: round(v, 4) for k, v in scores.items()})
        return scores

    def calculate_rouge(
        self,
        references: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Calculate ROUGE scores between references and predictions.

        ROUGE VARIANTS:
          ROUGE-1: Unigram overlap. Measures if the output uses the same
            individual words as the reference. High ROUGE-1 = good vocabulary
            coverage, but doesn't guarantee grammatical coherence.

          ROUGE-2: Bigram overlap. Measures if the output preserves word PAIRS
            from the reference. Higher bar than ROUGE-1 because word order
            matters. "The big red dog" and "red big the dog" have the same
            ROUGE-1 but very different ROUGE-2.

          ROUGE-L: Longest Common Subsequence. Measures the longest sequence
            of words that appears in both texts (not necessarily contiguous).
            Captures sentence-level structure. Best single ROUGE metric.

          Each variant reports:
            - Precision: What fraction of the output's n-grams are in the reference?
            - Recall: What fraction of the reference's n-grams are in the output?
            - F1: Harmonic mean of precision and recall (the standard number reported)

        Args:
            references: List of reference/expected texts.
            predictions: List of generated/predicted texts.

        Returns:
            Dictionary with ROUGE scores:
              - "rouge1": ROUGE-1 F1 score
              - "rouge2": ROUGE-2 F1 score
              - "rougeL": ROUGE-L F1 score
              - "rougeLsum": ROUGE-L computed over sentence pairs

        Raises:
            ValueError: If references and predictions have different lengths.
        """
        if len(references) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs "
                f"{len(predictions)} predictions"
            )

        logger.info(
            "calculating_rouge",
            num_examples=len(references),
        )

        try:
            import evaluate

            rouge_metric = evaluate.load("rouge")

            results = rouge_metric.compute(
                predictions=predictions,
                references=references,
                # use_stemmer reduces words to their root form, making
                # matching more lenient. "running" and "ran" would both
                # become "run" and count as a match.
                use_stemmer=True,
            )

            scores = {
                "rouge1": results["rouge1"],
                "rouge2": results["rouge2"],
                "rougeL": results["rougeL"],
                "rougeLsum": results["rougeLsum"],
            }

        except ImportError:
            logger.warning(
                "evaluate_not_installed",
                message="Install with: pip install evaluate rouge_score. "
                "Falling back to simple ROUGE approximation.",
            )
            scores = self._simple_rouge(references, predictions)

        logger.info("rouge_calculated", **{k: round(v, 4) for k, v in scores.items()})
        return scores

    def generate_samples(
        self,
        prompts: list[dict[str, str]],
        model_label: str = "default",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict[str, str]]:
        """Generate sample outputs for qualitative evaluation.

        WHY QUALITATIVE EVALUATION?
          Automated metrics (perplexity, BLEU, ROUGE) are useful but imperfect.
          They can't capture:
            - Factual correctness (is the answer actually right?)
            - Coherence (does the response make logical sense?)
            - Helpfulness (does it actually answer the question?)
            - Safety (is the response harmful or biased?)
            - Style (does it match the desired tone?)

          The BEST evaluation is a human reading the outputs. Generate a
          diverse set of test prompts and manually review the responses.

        TIPS FOR GOOD QUALITATIVE EVALUATION:
          - Include a variety of prompt types (simple, complex, edge cases)
          - Test prompts that are DIFFERENT from training data
          - Include prompts where the correct answer is ambiguous
          - Look for hallucination (confident but wrong answers)
          - Test the model's ability to say "I don't know"

        Args:
            prompts: List of dicts with "instruction" and optional "input"
                and "expected" keys.
            model_label: Which loaded model to use for generation.
            max_new_tokens: Maximum tokens per response.
            temperature: Sampling temperature.

        Returns:
            List of dicts with the original prompt info plus "generated" key.
        """
        if model_label not in self._models:
            raise KeyError(
                f"Model '{model_label}' not loaded. Available: "
                f"{list(self._models.keys())}"
            )

        model = self._models[model_label]
        tokenizer = self._tokenizers[model_label]

        logger.info(
            "generating_evaluation_samples",
            model_label=model_label,
            num_prompts=len(prompts),
        )

        results: list[dict[str, str]] = []

        for prompt_info in prompts:
            instruction = prompt_info.get("instruction", "")
            input_text = prompt_info.get("input", "")

            # Format using the Alpaca-style inference template
            if input_text.strip():
                formatted_prompt = (
                    "Below is an instruction that describes a task, paired with "
                    "further input that provides additional context. Write a "
                    "response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    "### Response:\n"
                )
            else:
                formatted_prompt = (
                    "Below is an instruction that describes a task. Write a "
                    "response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    "### Response:\n"
                )

            # Tokenize and generate
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=temperature > 0.0,
                    repetition_penalty=1.15,
                    # eos_token_id tells the model when to stop generating.
                    # Without it, the model might keep going until max_new_tokens.
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            result = {
                "instruction": instruction,
                "input": input_text,
                "expected": prompt_info.get("expected", ""),
                "generated": generated.strip(),
                "model": model_label,
            }
            results.append(result)

            logger.debug(
                "sample_generated",
                instruction=instruction[:50],
                generated_length=len(generated),
            )

        logger.info(
            "samples_generated",
            num_samples=len(results),
            model_label=model_label,
        )

        return results

    def evaluate(
        self,
        test_dataset: Dataset,
        model_label: str = "default",
        max_perplexity_samples: int = 100,
        max_generation_samples: int = 10,
        max_new_tokens: int = 256,
    ) -> EvaluationResult:
        """Run a comprehensive evaluation across all metrics.

        This is the main entry point for evaluation. It runs:
          1. Perplexity computation on the test set
          2. Sample generation for qualitative review
          3. BLEU and ROUGE scores (if reference outputs are available)

        Args:
            test_dataset: The held-out test/validation dataset.
                Should have "text" or "input_ids" columns for perplexity.
                Should have "instruction", "input", "output" columns for
                generation and metric computation.
            model_label: Which loaded model to evaluate.
            max_perplexity_samples: Max samples for perplexity computation.
            max_generation_samples: Number of samples to generate.
            max_new_tokens: Max tokens per generated response.

        Returns:
            EvaluationResult with all computed metrics.
        """
        logger.info(
            "comprehensive_evaluation_starting",
            model_label=model_label,
            dataset_size=len(test_dataset),
        )

        result = EvaluationResult(
            metadata={
                "model_label": model_label,
                "dataset_size": len(test_dataset),
                "device": self._device,
            }
        )

        # ─── Step 1: Perplexity ─────────────────────────────────────
        try:
            result.perplexity = self.calculate_perplexity(
                test_dataset,
                model_label=model_label,
                max_samples=max_perplexity_samples,
            )
        except Exception as e:
            logger.error("perplexity_calculation_failed", error=str(e))
            result.perplexity = None

        # ─── Step 2: Sample Generation ──────────────────────────────
        # Check if the dataset has the right columns for generation
        has_generation_columns = all(
            col in test_dataset.column_names
            for col in ["instruction", "output"]
        )

        if has_generation_columns:
            # Build prompts for generation
            n_gen = min(max_generation_samples, len(test_dataset))
            prompts = []
            for i in range(n_gen):
                example = test_dataset[i]
                prompts.append(
                    {
                        "instruction": example["instruction"],
                        "input": example.get("input", ""),
                        "expected": example["output"],
                    }
                )

            try:
                result.sample_outputs = self.generate_samples(
                    prompts,
                    model_label=model_label,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error("sample_generation_failed", error=str(e))

            # ─── Step 3: BLEU and ROUGE ─────────────────────────────
            # Only compute if we have both references and predictions
            if result.sample_outputs:
                references = [s["expected"] for s in result.sample_outputs]
                predictions = [s["generated"] for s in result.sample_outputs]

                # Filter out empty strings (would break metric computation)
                valid_pairs = [
                    (ref, pred)
                    for ref, pred in zip(references, predictions)
                    if ref.strip() and pred.strip()
                ]

                if valid_pairs:
                    valid_refs, valid_preds = zip(*valid_pairs)

                    try:
                        result.bleu_scores = self.calculate_bleu(
                            list(valid_refs),
                            list(valid_preds),
                        )
                    except Exception as e:
                        logger.error("bleu_calculation_failed", error=str(e))

                    try:
                        result.rouge_scores = self.calculate_rouge(
                            list(valid_refs),
                            list(valid_preds),
                        )
                    except Exception as e:
                        logger.error("rouge_calculation_failed", error=str(e))
        else:
            logger.info(
                "skipping_generation_metrics",
                reason="Dataset doesn't have instruction/output columns. "
                "Only perplexity is computed.",
                available_columns=test_dataset.column_names,
            )

        logger.info(
            "comprehensive_evaluation_complete",
            model_label=model_label,
            perplexity=result.perplexity,
        )

        return result

    def compare(
        self,
        base_result: EvaluationResult,
        finetuned_result: EvaluationResult,
    ) -> str:
        """Compare evaluation results between base and fine-tuned models.

        Generates a side-by-side comparison with improvement percentages.
        This is the "money shot" of fine-tuning — seeing measurable
        improvement on your task.

        Args:
            base_result: Evaluation results for the base model.
            finetuned_result: Evaluation results for the fine-tuned model.

        Returns:
            Formatted comparison string.
        """
        lines = [
            "=" * 70,
            "MODEL COMPARISON: Base vs Fine-Tuned",
            "=" * 70,
        ]

        # Perplexity comparison
        if (
            base_result.perplexity is not None
            and finetuned_result.perplexity is not None
        ):
            ppl_change = (
                (finetuned_result.perplexity - base_result.perplexity)
                / base_result.perplexity
                * 100
            )
            # For perplexity, LOWER is better, so negative change is good
            direction = "improved" if ppl_change < 0 else "worsened"
            lines.append("\n--- Perplexity (lower is better) ---")
            lines.append(f"  Base:       {base_result.perplexity:.2f}")
            lines.append(f"  Fine-tuned: {finetuned_result.perplexity:.2f}")
            lines.append(f"  Change:     {ppl_change:+.1f}% ({direction})")

        # BLEU comparison
        if base_result.bleu_scores and finetuned_result.bleu_scores:
            lines.append("\n--- BLEU Score (higher is better) ---")
            for key in ["bleu", "bleu_1", "bleu_4"]:
                if key in base_result.bleu_scores and key in finetuned_result.bleu_scores:
                    base_val = base_result.bleu_scores[key]
                    ft_val = finetuned_result.bleu_scores[key]
                    if base_val > 0:
                        change = (ft_val - base_val) / base_val * 100
                    else:
                        change = float("inf") if ft_val > 0 else 0.0
                    lines.append(
                        f"  {key:>8s}: Base={base_val:.4f}  "
                        f"FT={ft_val:.4f}  Change={change:+.1f}%"
                    )

        # ROUGE comparison
        if base_result.rouge_scores and finetuned_result.rouge_scores:
            lines.append("\n--- ROUGE Score (higher is better) ---")
            for key in ["rouge1", "rouge2", "rougeL"]:
                if key in base_result.rouge_scores and key in finetuned_result.rouge_scores:
                    base_val = base_result.rouge_scores[key]
                    ft_val = finetuned_result.rouge_scores[key]
                    if base_val > 0:
                        change = (ft_val - base_val) / base_val * 100
                    else:
                        change = float("inf") if ft_val > 0 else 0.0
                    lines.append(
                        f"  {key:>8s}: Base={base_val:.4f}  "
                        f"FT={ft_val:.4f}  Change={change:+.1f}%"
                    )

        # Sample output comparison (show side by side)
        base_samples = base_result.sample_outputs
        ft_samples = finetuned_result.sample_outputs
        if base_samples and ft_samples:
            n_compare = min(3, len(base_samples), len(ft_samples))
            lines.append(f"\n--- Sample Output Comparison ({n_compare} examples) ---")

            for i in range(n_compare):
                lines.append(f"\n  Example {i + 1}:")
                lines.append(f"  Instruction: {base_samples[i]['instruction'][:80]}")
                if base_samples[i].get("expected"):
                    lines.append(
                        f"  Expected:    {base_samples[i]['expected'][:120]}"
                    )
                lines.append(
                    f"  Base:        {base_samples[i]['generated'][:120]}"
                )
                lines.append(
                    f"  Fine-tuned:  {ft_samples[i]['generated'][:120]}"
                )

        lines.append("\n" + "=" * 70)

        comparison = "\n".join(lines)
        logger.info("comparison_generated")
        return comparison

    # ─── Fallback Metric Implementations ────────────────────────────
    # These simple implementations are used if the `evaluate` library
    # is not installed. They're less precise but educational.

    @staticmethod
    def _simple_bleu(
        references: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Simple BLEU approximation without the `evaluate` library.

        This is a simplified implementation for educational purposes.
        For production evaluation, install the `evaluate` library.

        HOW THIS WORKS (simplified BLEU):
          1. Tokenize both reference and prediction into words
          2. Count how many prediction words appear in the reference
          3. Divide by total prediction words = unigram precision
          4. Repeat for bigrams, trigrams, 4-grams
          5. Take the geometric mean
        """
        from collections import Counter

        def _get_ngrams(tokens: list[str], n: int) -> Counter:
            """Extract n-grams from a token list."""
            return Counter(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )

        precisions = [0.0, 0.0, 0.0, 0.0]  # 1-gram through 4-gram

        for n in range(1, 5):
            total_matches = 0
            total_predicted = 0

            for ref, pred in zip(references, predictions):
                ref_tokens = ref.lower().split()
                pred_tokens = pred.lower().split()

                if len(pred_tokens) < n:
                    continue

                ref_ngrams = _get_ngrams(ref_tokens, n)
                pred_ngrams = _get_ngrams(pred_tokens, n)

                # Count matches (clipped to reference count)
                for ngram, count in pred_ngrams.items():
                    total_matches += min(count, ref_ngrams.get(ngram, 0))

                total_predicted += max(len(pred_tokens) - n + 1, 0)

            if total_predicted > 0:
                precisions[n - 1] = total_matches / total_predicted

        # Geometric mean of precisions (add epsilon to avoid log(0))
        eps = 1e-10
        log_avg = sum(math.log(max(p, eps)) for p in precisions) / 4
        bleu = math.exp(log_avg)

        return {
            "bleu": bleu,
            "bleu_1": precisions[0],
            "bleu_2": precisions[1],
            "bleu_3": precisions[2],
            "bleu_4": precisions[3],
            "brevity_penalty": 1.0,  # Simplified: no brevity penalty
        }

    @staticmethod
    def _simple_rouge(
        references: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Simple ROUGE approximation without the `evaluate` library.

        Computes ROUGE-1, ROUGE-2, and ROUGE-L (F1 scores).
        This is a simplified implementation for educational purposes.
        """
        from collections import Counter

        def _lcs_length(x: list[str], y: list[str]) -> int:
            """Compute the length of the Longest Common Subsequence.

            LCS is a classic dynamic programming problem:
              - Build a 2D table where table[i][j] = LCS length of x[:i] and y[:j]
              - If x[i] == y[j], extend the previous LCS: table[i][j] = table[i-1][j-1] + 1
              - Otherwise, take the max of skipping from either side
            """
            m, n = len(x), len(y)
            # dp[i][j] = LCS length of x[:i] and y[:j]
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        rouge1_scores: list[float] = []
        rouge2_scores: list[float] = []
        rougeL_scores: list[float] = []

        for ref, pred in zip(references, predictions):
            ref_tokens = ref.lower().split()
            pred_tokens = pred.lower().split()

            if not ref_tokens or not pred_tokens:
                continue

            # ROUGE-1 (unigram F1)
            ref_counts = Counter(ref_tokens)
            pred_counts = Counter(pred_tokens)
            overlap = sum(
                min(ref_counts[t], pred_counts[t]) for t in ref_counts
            )
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(ref_tokens) if ref_tokens else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            rouge1_scores.append(f1)

            # ROUGE-2 (bigram F1)
            ref_bigrams = Counter(
                tuple(ref_tokens[i : i + 2])
                for i in range(len(ref_tokens) - 1)
            )
            pred_bigrams = Counter(
                tuple(pred_tokens[i : i + 2])
                for i in range(len(pred_tokens) - 1)
            )
            if ref_bigrams and pred_bigrams:
                overlap_2 = sum(
                    min(ref_bigrams[b], pred_bigrams[b]) for b in ref_bigrams
                )
                p2 = overlap_2 / sum(pred_bigrams.values())
                r2 = overlap_2 / sum(ref_bigrams.values())
                f1_2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0
                rouge2_scores.append(f1_2)
            else:
                rouge2_scores.append(0.0)

            # ROUGE-L (LCS-based F1)
            lcs = _lcs_length(ref_tokens, pred_tokens)
            p_l = lcs / len(pred_tokens) if pred_tokens else 0
            r_l = lcs / len(ref_tokens) if ref_tokens else 0
            f1_l = (
                2 * p_l * r_l / (p_l + r_l) if (p_l + r_l) > 0 else 0
            )
            rougeL_scores.append(f1_l)

        # Average across all examples
        def safe_mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return {
            "rouge1": safe_mean(rouge1_scores),
            "rouge2": safe_mean(rouge2_scores),
            "rougeL": safe_mean(rougeL_scores),
            "rougeLsum": safe_mean(rougeL_scores),  # Simplified: same as rougeL
        }
