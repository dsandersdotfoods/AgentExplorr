# LLM Fine-Tuning

> Train language models on your own data — no massive GPU cluster required.

## What You'll Learn

| File | Concept | Difficulty |
|------|---------|-----------|
| `data_preparation.py` | Loading & formatting datasets for instruction tuning | Beginner |
| `fine_tune_lora.py` | LoRA: Low-Rank Adaptation (parameter-efficient fine-tuning) | Intermediate |
| `fine_tune_qlora.py` | QLoRA: 4-bit quantized LoRA (fits on consumer GPUs) | Advanced |
| `evaluation.py` | Perplexity, ROUGE scores, qualitative evaluation | Intermediate |
| `configs/*.yaml` | Training hyperparameters with detailed explanations | Beginner |

## Fine-Tuning vs Pretraining

| | Pretraining | Fine-Tuning |
|---|---|---|
| **Data** | Trillions of tokens (internet-scale) | Thousands of examples (task-specific) |
| **Cost** | $1M+ (thousands of GPUs) | $1-100 (single GPU) |
| **Time** | Weeks/months | Hours |
| **Goal** | General language understanding | Task-specific performance |
| **Who** | Companies (Meta, Google, Mistral) | You! |

## How LoRA Works

LoRA (Low-Rank Adaptation) is the key innovation that makes fine-tuning affordable.

**The Problem**: A 7B parameter model has billions of weights. Updating ALL of them requires:
- Massive GPU memory (>60GB VRAM)
- Long training times
- Risk of catastrophic forgetting

**The Solution**: LoRA freezes the original weights and adds tiny trainable matrices:

```
Original weight matrix W (4096 × 4096 = 16.7M params) — FROZEN
                    ↓
            LoRA adds: W + ΔW
            where ΔW = A × B
            A: (4096 × 16) = 65K params  ← TRAINABLE
            B: (16 × 4096) = 65K params  ← TRAINABLE
                    ↓
        Result: 130K trainable params instead of 16.7M (99.2% reduction!)
```

The `r` parameter (rank) controls the size of A and B. Higher r = more capacity but more parameters.

## How QLoRA Works

QLoRA = LoRA + 4-bit Quantization

**Normal**: Model weights stored as FP32 (32 bits per weight) → 28GB for 7B model
**QLoRA**: Weights stored as INT4 (4 bits per weight) → ~3.5GB for 7B model

```
FP32 (32-bit):  3.14159265... → full precision, 28GB for 7B model
FP16 (16-bit):  3.14...       → half precision, 14GB
INT8 (8-bit):   3             → integer approx, 7GB
INT4 (4-bit):   ≈3            → minimal precision, 3.5GB ← QLoRA
```

**NF4 (NormalFloat4)**: A special 4-bit format optimized for neural network weights,
which follow a normal distribution. This is more accurate than naive INT4.

## Quick Start

### 1. Prepare Data
```python
from agentexplorr.llm_training.data_preparation import DatasetPreparer

preparer = DatasetPreparer()
dataset = preparer.load_alpaca(max_samples=1000)
```

### 2. Fine-Tune with LoRA
```python
from agentexplorr.llm_training.fine_tune_lora import LoRAFineTuner

tuner = LoRAFineTuner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,
    lora_alpha=32,
)
tuner.train(dataset, output_dir="outputs/lora")
```

### 3. Evaluate
```python
from agentexplorr.llm_training.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
perplexity = evaluator.calculate_perplexity(model, tokenizer, test_texts)
```

## When to Fine-Tune vs Prompt Engineer

| Use Prompt Engineering When | Use Fine-Tuning When |
|---|---|
| Quick experiments | Consistent style/format needed |
| General tasks | Domain-specific knowledge |
| Limited data (<100 examples) | Abundant data (1000+ examples) |
| Budget is tight | Performance is critical |
| Rapid iteration needed | Deployment optimization needed |

## Learning Resources

### Papers
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Alpaca](https://arxiv.org/abs/2303.16200) — Stanford, 2023

### Videos
- [Fine-Tune LLMs with LoRA](https://www.youtube.com/watch?v=YVU5wAA6Txo) — Hands-on tutorial
- [Tokenization Explained](https://www.youtube.com/watch?v=zduSFxRajkE) — Andrej Karpathy
- [QLoRA Paper Walkthrough](https://www.youtube.com/watch?v=y9PHWGOa8HA)

### Docs
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
