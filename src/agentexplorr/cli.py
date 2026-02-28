"""
AgentExplorr CLI — Command-Line Interface
==========================================

A simple command-line interface for exploring AgentExplorr's capabilities
without writing code. Useful for demos, quick experiments, and getting
an overview of what the project offers.

USAGE:
  python -m agentexplorr info          # Show project info and modules
  python -m agentexplorr calc "2+3"    # Quick calculator
  python -m agentexplorr benchmark     # Run agent benchmark (dry-run)
  python -m agentexplorr foundations    # Demo math foundations
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from agentexplorr import __version__


def _cmd_info(args: argparse.Namespace) -> None:
    """Display project information and available modules."""
    print(
        f"\n  AgentExplorr v{__version__} — AI/ML Playground & Portfolio\n"
        f"  {'=' * 52}\n"
        f"\n"
        f"  Modules:\n"
        f"    1. foundations        Math behind ML (backprop, optimizers, loss, activations)\n"
        f"    2. agents             ReAct, Tool-Calling, Multi-Agent (LangGraph + Ollama)\n"
        f"    3. mcp               Model Context Protocol servers, clients, tools\n"
        f"    4. llm_training      LoRA / QLoRA fine-tuning with Hugging Face\n"
        f"    5. classical_ml      sklearn pipelines, PyTorch CNNs, Transformers, MLflow\n"
        f"    6. rag               Document processing, chunking, vector stores, retrieval\n"
        f"    7. prompt_engineering Templates, few-shot, chain-of-thought, structured output\n"
        f"\n"
        f"  Quick start:\n"
        f"    python -m agentexplorr calc \"sqrt(144) + pi\"\n"
        f"    python -m agentexplorr foundations\n"
        f"    python -m agentexplorr benchmark\n"
        f"\n"
        f"  Docs: See README.md or docs/ folder\n"
    )


def _cmd_calc(args: argparse.Namespace) -> None:
    """Evaluate a math expression using the safe calculator."""
    from agentexplorr.agents.tools.calculator import SafeExpressionError, safe_evaluate

    expression = " ".join(args.expression)
    if not expression:
        print("Usage: python -m agentexplorr calc \"2 + 3 * 4\"")
        sys.exit(1)

    try:
        result = safe_evaluate(expression)
        print(f"  {expression} = {result}")
    except SafeExpressionError as e:
        print(f"  Error: {e}")
        sys.exit(1)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the agent benchmark suite (dry-run mode — no LLM needed)."""
    from agentexplorr.agents.evaluation.benchmarks import (
        get_math_questions,
        get_search_questions,
    )

    math_qs = get_math_questions()
    search_qs = get_search_questions()

    print("\n  AgentExplorr Benchmark Suite")
    print(f"  {'=' * 40}")
    print(f"\n  Math questions:   {len(math_qs)}")
    for q in math_qs:
        print(f"    [{q.difficulty:6s}] {q.question}")
        print(f"             Expected: {q.expected_answer}")

    print(f"\n  Search questions: {len(search_qs)}")
    for q in search_qs:
        print(f"    [{q.difficulty:6s}] {q.question}")
        print(f"             Expected: {q.expected_answer}")

    print(
        f"\n  Total: {len(math_qs) + len(search_qs)} questions"
        f"\n\n  To run with an agent:"
        f"\n    from agentexplorr.agents import ReActAgent"
        f"\n    from agentexplorr.agents.evaluation.benchmarks import AgentBenchmark"
        f"\n    benchmark = AgentBenchmark()"
        f"\n    summary = benchmark.run(ReActAgent(), benchmark.mixed_questions())"
        f"\n    print(f\"Accuracy: {{summary.accuracy:.1%}}\")\n"
    )


def _cmd_foundations(args: argparse.Namespace) -> None:
    """Run interactive demonstrations of the math foundations module."""
    import numpy as np

    from agentexplorr.foundations import (
        SGD,
        Adam,
        Momentum,
        binary_cross_entropy,
        gelu,
        huber_loss,
        leaky_relu,
        mean_squared_error,
        relu,
        sigmoid,
        softmax,
        tanh,
    )

    print("\n  Math Foundations Demo")
    print(f"  {'=' * 40}")

    # Activations
    print("\n  --- Activation Functions ---")
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    print(f"  Input x:     {x}")
    print(f"  ReLU(x):     {relu(x)}")
    print(f"  Sigmoid(x):  {np.round(sigmoid(x), 4)}")
    print(f"  Tanh(x):     {np.round(tanh(x), 4)}")
    print(f"  GELU(x):     {np.round(gelu(x), 4)}")
    print(f"  LeakyReLU:   {leaky_relu(x)}")
    print(f"  Softmax(x):  {np.round(softmax(x), 4)}")

    # Loss functions
    print("\n  --- Loss Functions ---")
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.3])
    print(f"  True:    {y_true}")
    print(f"  Pred:    {y_pred}")
    print(f"  MSE:     {mean_squared_error(y_true, y_pred):.6f}")
    print(f"  BCE:     {binary_cross_entropy(y_true, y_pred):.6f}")
    print(f"  Huber:   {huber_loss(y_true, y_pred):.6f}")

    # Optimizers
    print("\n  --- Optimizers (single step on w=[1.0, -1.0], grad=[0.5, -0.3]) ---")
    w = np.array([1.0, -1.0])
    grad = np.array([0.5, -0.3])

    sgd = SGD(lr=0.1)
    w_sgd = sgd.step(w.copy(), grad)
    print(f"  SGD(lr=0.1):      {np.round(w_sgd, 4)}")

    mom = Momentum(lr=0.1, beta=0.9)
    w_mom = mom.step(w.copy(), grad)
    print(f"  Momentum(0.9):    {np.round(w_mom, 4)}")

    adam = Adam(lr=0.01)
    w_adam = adam.step(w.copy(), grad)
    print(f"  Adam(lr=0.01):    {np.round(w_adam, 4)}")

    print(
        "\n  See src/agentexplorr/foundations/ for full implementations with"
        "\n  detailed math explanations and derivations.\n"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentexplorr",
        description="AgentExplorr — AI/ML Playground CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"agentexplorr {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    sub_info = subparsers.add_parser("info", help="Show project info and modules")
    sub_info.set_defaults(func=_cmd_info)

    # calc
    sub_calc = subparsers.add_parser("calc", help="Evaluate a math expression")
    sub_calc.add_argument("expression", nargs="+", help="Math expression to evaluate")
    sub_calc.set_defaults(func=_cmd_calc)

    # benchmark
    sub_bench = subparsers.add_parser("benchmark", help="Show benchmark question suite")
    sub_bench.set_defaults(func=_cmd_benchmark)

    # foundations
    sub_found = subparsers.add_parser("foundations", help="Demo math foundations")
    sub_found.set_defaults(func=_cmd_foundations)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        _cmd_info(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
