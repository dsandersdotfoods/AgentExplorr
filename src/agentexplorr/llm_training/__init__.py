"""
LLM Fine-Tuning & Training Module
===================================

WHAT IS THIS MODULE?
  This module provides a complete, educational pipeline for fine-tuning
  Large Language Models (LLMs) using parameter-efficient methods. Everything
  here is open-source and designed to run on consumer hardware or free
  cloud GPUs (Google Colab, Lambda Labs, Kaggle).

  You will learn to take a pre-trained model (like TinyLlama) and teach it
  new behaviors — following instructions, answering domain-specific questions,
  writing in a particular style — without training from scratch.

FINE-TUNING vs PRE-TRAINING — THE KEY DISTINCTION:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PRE-TRAINING (you probably won't do this)                         │
  │  ─────────────────────────────────────────                         │
  │  • Train a model from RANDOM WEIGHTS on TRILLIONS of tokens        │
  │  • Teaches general language understanding (grammar, facts, logic)   │
  │  • Requires 100s-1000s of GPUs, weeks-months, $1M-$100M+           │
  │  • Examples: GPT-4, Llama 3, Mistral were pre-trained              │
  │                                                                    │
  │  FINE-TUNING (what we do here)                                     │
  │  ────────────────────────────                                      │
  │  • Start from a PRE-TRAINED model (already understands language)    │
  │  • Train on a SMALL, TASK-SPECIFIC dataset (100s to 100K examples)  │
  │  • Updates only a small fraction of parameters (LoRA/QLoRA)         │
  │  • Requires 1 GPU, hours-days, $0-$50                              │
  │  • Result: A model specialized for YOUR use case                   │
  └──────────────────────────────────────────────────────────────────────┘

WHEN SHOULD YOU FINE-TUNE? (Decision Framework)
  Use PROMPT ENGINEERING when:
    - Your task can be solved with better instructions / few-shot examples
    - You need flexibility (change behavior without retraining)
    - You have < 100 examples

  Use RAG when:
    - You need the model to access external or frequently changing knowledge
    - You need source attribution / citations
    - Your data changes often

  Use FINE-TUNING when:
    - You need consistent behavior that prompting can't achieve
    - You have domain-specific jargon or style requirements
    - You want to reduce inference cost (shorter prompts after training)
    - You have 500+ high-quality labeled examples
    - You need the model to internalize patterns, not just reference docs

WHY LoRA AND QLoRA? (Parameter-Efficient Fine-Tuning)
  Full fine-tuning updates ALL model parameters. For a 7B model, that's
  7 billion float32 values = ~28 GB just for model weights, plus optimizer
  states (another ~56 GB). You'd need multiple A100 GPUs.

  LoRA (Low-Rank Adaptation) freezes the original model and injects small
  trainable matrices into each layer. Instead of updating 7B parameters,
  you train ~0.1-1% of them. This means:
    - 1 consumer GPU (16 GB VRAM) is enough
    - Training is 2-3x faster
    - Adapter weights are tiny (~10-100 MB vs 14+ GB for full model)

  QLoRA goes further by loading the base model in 4-bit precision:
    - FP32 (4 bytes) → INT4 (0.5 bytes) = ~8x memory reduction
    - A 7B model fits in ~4 GB VRAM instead of ~28 GB
    - Quality loss is minimal thanks to NormalFloat4 (NF4) quantization

MODULE STRUCTURE:
  llm_training/
  ├── __init__.py              ← You are here
  ├── data_preparation.py      ← Load, format, tokenize datasets
  ├── fine_tune_lora.py        ← LoRA fine-tuning (full precision base)
  ├── fine_tune_qlora.py       ← QLoRA fine-tuning (4-bit quantized base)
  ├── evaluation.py            ← Evaluate models (perplexity, BLEU, ROUGE)
  ├── configs/
  │   ├── lora_config.yaml     ← LoRA hyperparameters
  │   └── training_config.yaml ← Training hyperparameters
  └── README.md                ← Comprehensive learning guide

LEARNING RESOURCES:
  Papers (READ THESE — they're approachable and foundational):
    - "LoRA: Low-Rank Adaptation of Large Language Models"
      (Hu et al., 2021) — https://arxiv.org/abs/2106.09685
    - "QLoRA: Efficient Finetuning of Quantized Language Models"
      (Dettmers et al., 2023) — https://arxiv.org/abs/2305.14314
    - "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning"
      (Lialin et al., 2023) — https://arxiv.org/abs/2303.15647

  Documentation:
    - Hugging Face PEFT: https://huggingface.co/docs/peft
    - Hugging Face TRL (Transformer Reinforcement Learning):
      https://huggingface.co/docs/trl
    - BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

  Video Resources (HIGHLY recommended for visual learners):
    - "LoRA Explained" (Umar Jamil, excellent visuals):
      https://www.youtube.com/watch?v=PXWYUTMt-AU
    - "Fine-tune LLMs with QLoRA" (Trelis Research):
      https://www.youtube.com/watch?v=XpoKB3usmKc
    - "Fine-tuning LLMs with Hugging Face" (freeCodeCamp, full course):
      https://www.youtube.com/watch?v=iOdFUJiB0Zc
    - "What is Quantization?" (IBM Technology):
      https://www.youtube.com/watch?v=WFi0Rmj5VIc
    - "Train Your Own LLM" (Andrej Karpathy):
      https://www.youtube.com/watch?v=kCc8FmEb1nY

ALL OPEN SOURCE — NO COMMERCIAL APIs REQUIRED:
  - Models      : Hugging Face Hub (TinyLlama, Llama, Mistral, etc.)
  - Training    : Hugging Face Transformers + PEFT + TRL
  - Quantization: BitsAndBytes (Tim Dettmers)
  - Datasets    : Hugging Face Datasets (Alpaca, Dolly, OASST, etc.)
  - Tracking    : MLflow / Weights & Biases (optional, free tiers)
"""

from __future__ import annotations

from agentexplorr.llm_training.data_preparation import DatasetPreparer
from agentexplorr.llm_training.evaluation import ModelEvaluator
from agentexplorr.llm_training.fine_tune_lora import LoRAFineTuner
from agentexplorr.llm_training.fine_tune_qlora import QLoRAFineTuner

__all__ = [
    # Data pipeline
    "DatasetPreparer",
    # Fine-tuning methods
    "LoRAFineTuner",
    "QLoRAFineTuner",
    # Evaluation
    "ModelEvaluator",
]
