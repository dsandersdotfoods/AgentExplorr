# LLM Fine-Tuning & Training Module

> **Your complete guide to teaching language models new tricks.**
>
> Everything here is open-source. No commercial API keys required.
> Designed to run on free cloud GPUs (Google Colab) or consumer hardware.

---

## Table of Contents

1. [What is Fine-Tuning?](#what-is-fine-tuning)
2. [Fine-Tuning vs Pre-Training](#fine-tuning-vs-pre-training)
3. [When to Fine-Tune vs Prompt Engineer vs RAG](#when-to-fine-tune)
4. [LoRA Explained](#lora-explained)
5. [QLoRA Explained](#qlora-explained)
6. [Module Architecture](#module-architecture)
7. [Quick Start Guide](#quick-start-guide)
8. [Step-by-Step Tutorial](#step-by-step-tutorial)
9. [Troubleshooting](#troubleshooting)
10. [Learning Resources](#learning-resources)

---

## What is Fine-Tuning?

Fine-tuning takes a **pre-trained model** (one that already understands language) and **specializes it** for your specific task by training on a small, task-specific dataset.

**Analogy:** Pre-training is like getting a general education (K-12 + college). Fine-tuning is like getting specialized training for a specific job. You don't start from scratch -- you build on your existing knowledge.

**What fine-tuning can teach a model:**
- Follow specific instruction formats consistently
- Write in a particular style or tone
- Use domain-specific jargon correctly (legal, medical, technical)
- Perform specific tasks better (summarization, translation, code generation)
- Refuse inappropriate requests in a particular way

**What fine-tuning CANNOT do:**
- Add factual knowledge the model never learned (use RAG for this)
- Make a small model as smart as a large one
- Fix fundamental reasoning limitations
- Guarantee factual accuracy

---

## Fine-Tuning vs Pre-Training

```
+----------------------------------------------------------------------+
|                                                                      |
|  PRE-TRAINING                        FINE-TUNING                     |
|  ------------                        -----------                     |
|                                                                      |
|  Goal:    Learn language itself       Goal:    Learn a specific task  |
|  Data:    Trillions of tokens         Data:    100s - 100K examples   |
|  Cost:    $1M - $100M+                Cost:    $0 - $50              |
|  Time:    Weeks - Months              Time:    Minutes - Hours        |
|  Hardware: 100s-1000s of GPUs         Hardware: 1 GPU (even free!)   |
|  Params:  ALL parameters              Params:  0.1% - 1% (LoRA)     |
|  Result:  General language model      Result:  Specialized model     |
|                                                                      |
|  Examples:                            Examples:                       |
|  GPT-4, Llama 3, Mistral             Your custom chatbot, domain    |
|  (someone else did this for you)      expert, code assistant         |
|                                                                      |
+----------------------------------------------------------------------+
```

**Key insight:** You don't need to pre-train. Companies like Meta (Llama), Mistral, and the open-source community have already done the expensive pre-training. You just need to fine-tune their models for your use case.

| Aspect | Pretraining | Fine-Tuning |
|--------|-------------|-------------|
| **Data** | Trillions of tokens (internet-scale) | Thousands of examples (task-specific) |
| **Cost** | $1M+ (thousands of GPUs) | $0-100 (single GPU) |
| **Time** | Weeks/months | Minutes to hours |
| **Goal** | General language understanding | Task-specific performance |
| **Who does it** | Companies (Meta, Google, Mistral) | You! |

---

## When to Fine-Tune?

Use this decision framework:

### Use Prompt Engineering When:
- Your task can be solved with better instructions or few-shot examples
- You need flexibility to change behavior without retraining
- You have fewer than 100 examples
- You are iterating quickly on the task definition
- **Cost: Free, instant changes**

### Use RAG (Retrieval Augmented Generation) When:
- You need the model to access external/changing knowledge
- You need source attribution (citations)
- Your data changes frequently
- You want to reduce hallucination on factual questions
- **Cost: Free (with local embeddings + vector DB)**

### Use Fine-Tuning When:
- Prompting cannot achieve the consistency you need
- You need specific style, format, or behavior patterns
- You have 500+ high-quality labeled examples
- You want to reduce inference cost (shorter prompts)
- You need domain-specific jargon or reasoning
- **Cost: $0-50 on cloud GPUs, hours of time**

### Decision Flowchart:
```
Can you solve it with a better prompt?
  |-- YES --> Use Prompt Engineering
  +-- NO
      |-- Does the model need external knowledge?
      |   |-- YES --> Use RAG
      |   +-- NO
      |       |-- Do you have 500+ labeled examples?
      |       |   |-- YES --> Fine-Tune!
      |       |   +-- NO  --> Collect more data or try few-shot prompting
      |       +-- Is it about style/format/behavior?
      |           |-- YES --> Fine-Tune (even with fewer examples)
      |           +-- NO  --> Consider RAG + Prompt Engineering combo
```

### Summary Table

| Use Prompt Engineering When | Use Fine-Tuning When |
|-----------------------------|----------------------|
| Quick experiments | Consistent style/format needed |
| General tasks | Domain-specific knowledge |
| Limited data (<100 examples) | Abundant data (1000+ examples) |
| Budget is tight | Performance is critical |
| Rapid iteration needed | Deployment optimization needed |

---

## LoRA Explained

### What is LoRA? (Low-Rank Adaptation)

LoRA is the most popular **parameter-efficient fine-tuning** (PEFT) method. It lets you fine-tune large models on consumer hardware by training only a tiny fraction of the parameters.

**Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

### The Problem LoRA Solves

Full fine-tuning updates ALL model parameters:
- 7B model = 7 billion parameters
- Each parameter = 4 bytes (FP32) + 8 bytes optimizer state
- Total memory: **~84 GB** just for model + optimizer
- You would need multiple A100 GPUs ($10,000+ each)

### How LoRA Works

```
                    STANDARD LINEAR LAYER
                    ---------------------
        +-------+
  x --> |   W   | --> y = W*x
        |(d x d)|
        +-------+
        16.7M params (for d=4096)


                    LoRA-MODIFIED LAYER
                    -------------------

        +--------+
  x --> |W(frozen)| --+
        +--------+   |
                      | (add) --> y = W*x + B*A*x
        +---+ +---+   |
  x --> | A |-| B |---+
        |d*r| |r*d|
        +---+ +---+
        131K params (for d=4096, r=16)  <-- 128x fewer!
```

**The key insight:** When you fine-tune a model, the weight updates (delta-W) have **low intrinsic rank**. This means the "important" part of the update can be captured by two small matrices multiplied together, rather than one huge matrix.

**Think of it like SVD/PCA for weight updates** -- most of the "signal" lives in a low-dimensional subspace.

### The Math (Simplified)

In a standard linear layer: `y = W * x` (W is a huge d x d matrix)

With LoRA: `y = W * x + (alpha/r) * B * A * x`

Where:
- `W` = original frozen weights (e.g., 4096 x 4096 = 16.7M parameters)
- `A` = small matrix of shape (d x r), where r << d (e.g., r=16)
- `B` = small matrix of shape (r x d)
- `B * A` = low-rank update (only r * d * 2 = 131K parameters for d=4096, r=16)

So instead of updating 16.7M parameters per layer, we update only 131K. That is a **128x reduction**, and it works almost as well as full fine-tuning.

### LoRA Hyperparameters

| Parameter | What it controls | Typical values | Start with |
|-----------|-----------------|----------------|------------|
| `r` (rank) | Capacity of LoRA update | 8, 16, 32, 64 | **16** |
| `alpha` | Scaling factor | r to 2*r | **32** (2*r) |
| `dropout` | Regularization | 0.0 - 0.1 | **0.05** |
| `target_modules` | Which layers to adapt | Attention, MLP | **All linear** |

**r (rank)** is the most important:
- `r=8`:  Minimal adaptation. Good for simple tasks.
- `r=16`: Good default. Works well for instruction following.
- `r=32`: More capacity. Better for complex tasks.
- `r=64`: Approaches full fine-tuning quality. Diminishing returns beyond this.

### Memory Comparison (7B Model)

| Method | Model Memory | Optimizer Memory | Total VRAM |
|--------|-------------|-----------------|------------|
| Full Fine-Tuning | 28 GB (FP32) | 56 GB | **~84 GB** |
| LoRA (FP16 base) | 14 GB | ~0.1 GB | **~14 GB** |
| QLoRA (4-bit base) | 3.5 GB | ~0.1 GB | **~4 GB** |

---

## QLoRA Explained

### What is QLoRA?

QLoRA = **LoRA + 4-bit Quantization** of the base model.

While LoRA reduces the *trainable* parameters, QLoRA reduces the *memory* needed for the *frozen* base model by loading it in 4-bit precision.

**Paper:** [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

### What is Quantization?

Quantization reduces the precision (number of bits) used to store each weight:

```
FP32 (32 bits, 4 bytes):   ################################
  Full precision. 7 decimal digits. This is the default.

FP16 (16 bits, 2 bytes):   ################
  Half precision. 3 decimal digits. 2x memory savings.

INT8 (8 bits, 1 byte):     ########
  256 possible values. 4x savings. Some quality loss.

NF4 (4 bits, 0.5 bytes):   ####
  Only 16 possible values! 8x savings. Amazingly, works well.
```

### Why NF4 (NormalFloat4)?

Neural network weights typically follow a **normal (Gaussian) distribution** centered around zero. NF4 places its 16 quantization levels optimally for this distribution -- more levels near zero (where most weights are) and fewer in the tails.

```
  Weight Distribution:            NF4 Quantization Levels:

         ##                              |
        ####                         |   |   |
       ######                      | |   |   | |
      ########                   | | |   |   | | |
    ############              |  | | |   |   | | |  |
  ################         |  |  | | |   |   | | |  |  |
  ----------------         -----------------------------
  -3   -1  0  1   3       -3   -1  0  1   3

  Most weights are            More quantization levels
  near zero                   placed near zero = less error
```

### Double Quantization

When you quantize weights, you need to store a **scaling factor** for each block of 64 weights. These scaling factors are in FP32 (4 bytes each).

Double quantization **quantizes the scaling factors themselves** to INT8, saving an additional ~0.4 GB per 7B parameters with negligible quality impact.

```
Without double quant:   64 weights (NF4) + 1 scaling factor (FP32)
                        = 32 bytes + 4 bytes = 36 bytes per block

With double quant:      64 weights (NF4) + 1 scaling factor (INT8) + overhead
                        = 32 bytes + 1 byte + ~0.5 bytes = ~33.5 bytes per block

Savings: ~7% additional memory reduction
```

### QLoRA Memory Requirements

| Model | FP32 | FP16 (LoRA) | 4-bit (QLoRA) | Fits on... |
|-------|------|-------------|---------------|------------|
| TinyLlama 1.1B | 4.4 GB | 2.2 GB | **~0.7 GB** | Any GPU |
| Phi-2 2.7B | 10.8 GB | 5.4 GB | **~1.7 GB** | Any GPU |
| Mistral 7B | 28 GB | 14 GB | **~4 GB** | Colab T4 (free!) |
| Llama 2 13B | 52 GB | 26 GB | **~7 GB** | Colab T4/L4 |
| Llama 2 70B | 280 GB | 140 GB | **~35 GB** | A100 80GB |

---

## Module Architecture

```
llm_training/
|-- __init__.py              # Module entry point, imports, documentation
|-- data_preparation.py      # DatasetPreparer: Load -> Format -> Tokenize -> Split
|-- fine_tune_lora.py        # LoRAFineTuner: Full-precision base + LoRA adapters
|-- fine_tune_qlora.py       # QLoRAFineTuner: 4-bit base + LoRA adapters
|-- evaluation.py            # ModelEvaluator: Perplexity, BLEU, ROUGE, qualitative
|-- configs/
|   |-- lora_config.yaml     # LoRA hyperparameters (heavily commented)
|   +-- training_config.yaml # Full training pipeline config
+-- README.md                # This file
```

### What You Will Learn

| File | Concept | Difficulty |
|------|---------|-----------|
| `data_preparation.py` | Loading and formatting datasets for instruction tuning | Beginner |
| `fine_tune_lora.py` | LoRA: Low-Rank Adaptation (parameter-efficient fine-tuning) | Intermediate |
| `fine_tune_qlora.py` | QLoRA: 4-bit quantized LoRA (fits on consumer GPUs) | Advanced |
| `evaluation.py` | Perplexity, BLEU, ROUGE scores, qualitative evaluation | Intermediate |
| `configs/*.yaml` | Training hyperparameters with detailed explanations | Beginner |

### Data Flow

```
+-------------+   +-------------+   +------------+   +-------------+
| HuggingFace |-->| Format into |-->|  Tokenize  |-->| Train/Val   |
| Hub Dataset |   | Instruction |   | (pad/trunc)|   | Split       |
| (Alpaca)    |   | Template    |   |            |   |             |
+-------------+   +-------------+   +------------+   +------+------+
                                                             |
                  +-------------+   +------------+           |
                  | Save/Merge  |<--| Train      |<----------+
                  | Adapter     |   | (SFTTrainer|
                  +------+------+   | + LoRA)    |
                         |          +------------+
                         |
                  +------v------+
                  |  Evaluate   |
                  | (PPL, BLEU, |
                  |  ROUGE)     |
                  +-------------+
```

---

## Quick Start Guide

### Prerequisites

```bash
# Install required packages
pip install torch transformers datasets peft trl bitsandbytes accelerate

# Optional: for evaluation metrics
pip install evaluate rouge_score

# Optional: for experiment tracking
pip install wandb  # or mlflow
```

### Minimal Example (10 lines to fine-tune)

```python
from agentexplorr.llm_training.data_preparation import DatasetPreparer, DatasetConfig
from agentexplorr.llm_training.fine_tune_lora import LoRAFineTuner, LoRAConfig

# Prepare data (downloads Alpaca dataset, formats, tokenizes)
preparer = DatasetPreparer(DatasetConfig(max_samples=500))
processed = preparer.prepare("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Fine-tune with LoRA
config = LoRAConfig(num_train_epochs=1, per_device_train_batch_size=2)
tuner = LoRAFineTuner(config)
tuner.load_model()
tuner.apply_lora()
metrics = tuner.train(processed.train_dataset, processed.val_dataset)

# Test it!
from agentexplorr.llm_training.data_preparation import DatasetPreparer
prompt = DatasetPreparer.format_for_inference("Explain quantum computing simply.")
print(tuner.generate(prompt))
```

---

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

```python
from agentexplorr.llm_training.data_preparation import DatasetPreparer, DatasetConfig

# Configure -- start small for testing!
config = DatasetConfig(
    dataset_name="tatsu-lab/alpaca",  # 52K instruction examples
    max_samples=1000,                  # Start with 1K for speed
    max_seq_length=512,                # Tokens per example
    val_split_ratio=0.1,               # 10% for validation
)

preparer = DatasetPreparer(config)
processed = preparer.prepare("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Inspect the data
print(f"Training examples: {len(processed.train_dataset)}")
print(f"Validation examples: {len(processed.val_dataset)}")
print(f"Columns: {processed.train_dataset.column_names}")

# Save for reuse (skip re-tokenization on future runs)
preparer.save_processed_dataset(processed, "./data/processed_alpaca")
```

### Step 2a: Fine-Tune with LoRA

```python
from agentexplorr.llm_training.fine_tune_lora import LoRAFineTuner, LoRAConfig

config = LoRAConfig(
    base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_r=16,              # Rank -- 16 is a good default
    lora_alpha=32,          # Scaling -- 2x rank
    lora_dropout=0.05,      # Light regularization
    num_train_epochs=3,     # 3 passes through the data
    learning_rate=2e-4,     # Standard for LoRA
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
)

tuner = LoRAFineTuner(config)
tuner.load_model()           # Download and load base model
tuner.apply_lora()           # Inject LoRA adapters (freezes base)
metrics = tuner.train(       # Train!
    processed.train_dataset,
    processed.val_dataset,
)

# Save just the adapter (~10-50 MB)
tuner.save_adapter("./outputs/my_lora_adapter")

# Or merge into a standalone model (~2 GB for TinyLlama)
tuner.merge_and_save("./outputs/my_merged_model")
```

### Step 2b: Fine-Tune with QLoRA (Lower Memory)

```python
from agentexplorr.llm_training.fine_tune_qlora import QLoRAFineTuner, QLoRAConfig

config = QLoRAConfig(
    base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_4bit=True,                # THE key difference from LoRA
    bnb_4bit_quant_type="nf4",        # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True,   # Extra memory savings
    lora_r=16,
    lora_alpha=32,
    num_train_epochs=3,
    learning_rate=2e-4,
)

tuner = QLoRAFineTuner(config)
tuner.load_model()            # Loads model in 4-bit!
tuner.apply_qlora()           # Apply LoRA on top of 4-bit model
metrics = tuner.train(processed.train_dataset, processed.val_dataset)
tuner.save_adapter("./outputs/my_qlora_adapter")
```

### Step 3: Evaluate Your Model

```python
from agentexplorr.llm_training.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Load both models for comparison
evaluator.load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    label="base"
)
evaluator.load_model(
    "./outputs/my_merged_model",
    label="finetuned"
)

# Run comprehensive evaluation
base_result = evaluator.evaluate(
    processed.val_dataset,
    model_label="base"
)
ft_result = evaluator.evaluate(
    processed.val_dataset,
    model_label="finetuned"
)

# Print comparison
print(evaluator.compare(base_result, ft_result))

# Print detailed results
print(base_result.summary())
print(ft_result.summary())
```

### Step 4: Use Your Fine-Tuned Model

```python
from agentexplorr.llm_training.data_preparation import DatasetPreparer

# Format a prompt using the Alpaca template
prompt = DatasetPreparer.format_for_inference(
    instruction="Explain the theory of relativity in simple terms.",
)
response = tuner.generate(prompt, max_new_tokens=256, temperature=0.7)
print(response)

# With input context
prompt = DatasetPreparer.format_for_inference(
    instruction="Summarize the following text.",
    input_text="Your long text here..."
)
response = tuner.generate(prompt)
print(response)
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

This is the number one issue. Solutions in order of impact:

1. **Reduce batch size** to 2 or 1, increase `gradient_accumulation_steps` to compensate
2. **Use QLoRA** instead of LoRA (4x memory reduction for base model)
3. **Reduce `max_seq_length`** (memory scales quadratically!)
4. **Reduce `lora_r`** (fewer trainable parameters)
5. **Enable gradient checkpointing** (should be on by default)
6. **Use a smaller model** (TinyLlama 1.1B instead of Mistral 7B)

### Training Loss Not Decreasing

1. **Learning rate too low**: Try 5e-4 or 1e-3
2. **Data formatting issue**: Print a few examples and verify they look correct
3. **Too few epochs**: Try 5-10 epochs on small datasets
4. **Wrong tokenizer**: Make sure the tokenizer matches the model

### Training Loss Oscillating Wildly

1. **Learning rate too high**: Try 1e-4 or 5e-5
2. **Batch size too small**: Increase `gradient_accumulation_steps`
3. **Gradient clipping**: Ensure `max_grad_norm=1.0`
4. **Warmup too short**: Try `warmup_ratio=0.1`

### Model Generates Garbage After Fine-Tuning

1. **Overfitting**: Reduce epochs, increase dropout, add more data
2. **Data quality**: Check your training data for errors
3. **Tokenizer mismatch**: Verify you are using the same tokenizer for training and inference
4. **Wrong prompt format**: Use `DatasetPreparer.format_for_inference()` at inference time

### BitsAndBytes Errors

```bash
# "bitsandbytes not found" or "CUDA not available"
pip install bitsandbytes  # Requires CUDA GPU

# On Google Colab, check GPU is enabled:
# Runtime -> Change runtime type -> GPU (T4)
```

---

## Learning Resources

### Papers (Read These!)

| Paper | Year | Why Read It |
|-------|------|-------------|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | The foundational paper. Accessible and well-written. |
| [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) | 2023 | Enabled fine-tuning on consumer GPUs. Revolutionary. |
| [Scaling Down to Scale Up: Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.15647) | 2023 | Survey of ALL PEFT methods. Great overview. |
| [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) | 2023 | Showed you can fine-tune with 52K examples for $100. |
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | 2023 | Only 1,000 examples needed for good fine-tuning! |

### Video Tutorials

| Video | Creator | Duration | Level |
|-------|---------|----------|-------|
| [LoRA Explained (with visuals)](https://www.youtube.com/watch?v=PXWYUTMt-AU) | Umar Jamil | ~30 min | Beginner |
| [Fine-tune LLMs with QLoRA](https://www.youtube.com/watch?v=XpoKB3usmKc) | Trelis Research | ~20 min | Intermediate |
| [Fine-tuning LLMs (Full Course)](https://www.youtube.com/watch?v=iOdFUJiB0Zc) | freeCodeCamp | ~4 hrs | Beginner |
| [What is Quantization?](https://www.youtube.com/watch?v=WFi0Rmj5VIc) | IBM Technology | ~10 min | Beginner |
| [Tokenization Explained](https://www.youtube.com/watch?v=zduSFxRajkE) | Andrej Karpathy | ~2 hrs | Deep Dive |
| [Train Your Own LLM](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Andrej Karpathy | ~2 hrs | Advanced |
| [SFTTrainer Deep Dive](https://www.youtube.com/watch?v=lQCJL0YEk2U) | Hugging Face | ~15 min | Intermediate |
| [NLP Evaluation Metrics Explained](https://www.youtube.com/watch?v=TMshhnrEXlg) | Weights & Biases | ~20 min | Beginner |
| [Perplexity in 5 Minutes](https://www.youtube.com/watch?v=NURcDHhYe98) | ritvikmath | ~5 min | Beginner |
| [QLoRA Paper Walkthrough](https://www.youtube.com/watch?v=y9PHWGOa8HA) | AI Coffee Break | ~15 min | Intermediate |
| [Prepare Data for Fine-Tuning](https://www.youtube.com/watch?v=XYsz6wDOdlY) | Trelis Research | ~15 min | Beginner |

### Documentation

- [Hugging Face PEFT](https://huggingface.co/docs/peft) - LoRA/QLoRA implementation
- [Hugging Face TRL](https://huggingface.co/docs/trl) - SFTTrainer and RLHF
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model loading and training
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) - Dataset loading and processing
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization library
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate) - Evaluation metrics

### Blogs

- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Sebastian Raschka
- [Making LLMs even more accessible with bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - Hugging Face
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) - Andrej Karpathy

---

## Google Colab Quick Setup

If you want to run this on a free GPU:

```python
# In a Colab notebook:

# 1. Enable GPU: Runtime -> Change runtime type -> GPU (T4)

# 2. Install dependencies
!pip install -q torch transformers datasets peft trl bitsandbytes accelerate evaluate rouge_score

# 3. Clone the repo
!git clone https://github.com/YOUR_USERNAME/AgentExplorr.git
!cd AgentExplorr && pip install -e .

# 4. Run the quick start example (copy from Quick Start Guide above)
from agentexplorr.llm_training import DatasetPreparer, LoRAFineTuner
# ... (see Quick Start Guide above)
```

**Colab GPU Memory:**
- T4 (free tier): 15 GB VRAM -- TinyLlama (LoRA), Mistral 7B (QLoRA)
- A100 (Colab Pro): 40/80 GB VRAM -- Llama 2 13B (LoRA), Llama 2 70B (QLoRA)
- L4 (Colab Pro): 24 GB VRAM -- Mistral 7B (LoRA), Llama 2 13B (QLoRA)

---

*Built with care for the AgentExplorr learning playground. Happy fine-tuning!*
