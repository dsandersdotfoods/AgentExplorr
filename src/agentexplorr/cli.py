"""
AgentExplorr CLI — Interactive Console
========================================

An interactive, menu-driven console for exploring AgentExplorr's AI/ML
modules. Uses Rich for beautiful terminal output and questionary for
arrow-key navigation.

USAGE:
  python -m agentexplorr               # Launch interactive mode
  python -m agentexplorr calc "2+3"    # Quick calculator (script mode)
  python -m agentexplorr --version     # Show version
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agentexplorr import __version__

console = Console()

# ── Shared helpers ──────────────────────────────────────────────────────

BACK = "Back"


def _select(message: str, choices: list[str]) -> str | None:
    """Show an arrow-key menu. Returns None on Ctrl-C."""
    try:
        return questionary.select(message, choices=choices).ask()
    except KeyboardInterrupt:
        return None


def _pause() -> None:
    """Wait for Enter before returning to menu."""
    console.print()
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        questionary.press_any_key_to_continue("Press any key to continue...").ask()


# ── Interactive mode: main loop ─────────────────────────────────────────


def interactive_mode() -> None:
    """Main interactive loop — the heart of the console experience."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]AgentExplorr[/bold cyan] v{__version__} — "
            "[dim]AI/ML Playground[/dim]\n\n"
            "[dim]Navigate with arrow keys, Enter to select.[/dim]",
            border_style="cyan",
        )
    )

    menu_items = {
        "Math Foundations    — activations, loss, optimizers, backprop": _explore_foundations,
        "AI Agents           — ReAct, Tool-Calling, Multi-Agent": _explore_agents,
        "RAG Pipeline        — chunking, retrieval, generation": _explore_rag,
        "Prompt Engineering  — CoT, few-shot, templates": _explore_prompts,
        "Classical ML        — classification, regression, clustering": _explore_classical_ml,
        "Calculator          — evaluate math expressions": _interactive_calc,
        "Benchmarks          — agent evaluation questions": _show_benchmarks,
        "Project Info        — modules, tech stack, architecture": _show_project_info,
        "Quit": None,
    }

    while True:
        console.print()
        choice = _select("What would you like to explore?", list(menu_items.keys()))

        if choice is None or choice == "Quit":
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        handler = menu_items.get(choice)
        if handler:
            handler()


# ── Math Foundations ─────────────────────────────────────────────────────


def _explore_foundations() -> None:
    """Interactive explorer for the math foundations module."""
    while True:
        choice = _select(
            "Math Foundations",
            [
                "Activation functions",
                "Loss functions",
                "Optimizer comparison",
                "All at once",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

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
            tanh,
        )

        if choice in ("Activation functions", "All at once"):
            console.print()
            x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
            table = Table(title="Activation Functions", show_header=True, border_style="cyan")
            table.add_column("x", style="bold")
            table.add_column("ReLU")
            table.add_column("Sigmoid")
            table.add_column("Tanh")
            table.add_column("GELU")
            table.add_column("LeakyReLU")

            # Each activation returns (values, gradient) — we only need values
            r = relu(x)[0]
            s = sigmoid(x)[0]
            t = tanh(x)[0]
            g = gelu(x)[0]
            lr_vals = leaky_relu(x)[0]
            for i, xi in enumerate(x):
                table.add_row(
                    f"{float(xi):5.1f}",
                    f"{float(r[i]):7.4f}",
                    f"{float(s[i]):7.4f}",
                    f"{float(t[i]):7.4f}",
                    f"{float(g[i]):7.4f}",
                    f"{float(lr_vals[i]):7.4f}",
                )

            console.print(table)
            console.print(
                Panel(
                    "[bold]Key insight:[/bold] Without activations, stacking linear layers "
                    "just produces another linear function.\n"
                    "ReLU is the default for hidden layers. GELU is used in Transformers (GPT, BERT).\n"
                    "Softmax normalizes to a probability distribution for classification output.",
                    title="Why activations matter",
                    border_style="green",
                )
            )

        if choice in ("Loss functions", "All at once"):
            console.print()
            y_true = np.array([1.0, 0.0, 1.0, 0.0])
            y_pred = np.array([0.9, 0.1, 0.8, 0.3])

            table = Table(title="Loss Functions", border_style="cyan")
            table.add_column("Loss", style="bold")
            table.add_column("Value", justify="right")
            table.add_column("Best for")

            table.add_row(
                "MSE",
                f"{mean_squared_error(y_true, y_pred):.6f}",
                "Regression (continuous values)",
            )
            table.add_row(
                "Binary Cross-Entropy",
                f"{binary_cross_entropy(y_true, y_pred):.6f}",
                "Binary classification",
            )
            table.add_row(
                "Huber",
                f"{huber_loss(y_true, y_pred):.6f}",
                "Regression (outlier-robust)",
            )

            console.print(table)
            console.print(f"  [dim]y_true={y_true}  y_pred={y_pred}[/dim]")

        if choice in ("Optimizer comparison", "All at once"):
            console.print()
            w = np.array([1.0, -1.0])
            grad = np.array([0.5, -0.3])

            table = Table(title="Optimizer Single Step", border_style="cyan")
            table.add_column("Optimizer", style="bold")
            table.add_column("Config")
            table.add_column("w after step", justify="right")

            w_sgd = SGD(lr=0.1).step(w.copy(), grad)
            w_mom = Momentum(lr=0.1, beta=0.9).step(w.copy(), grad)
            w_adam = Adam(lr=0.01).step(w.copy(), grad)

            table.add_row("SGD", "lr=0.1", str(np.round(w_sgd, 4)))
            table.add_row("Momentum", "lr=0.1, beta=0.9", str(np.round(w_mom, 4)))
            table.add_row("Adam", "lr=0.01", str(np.round(w_adam, 4)))

            console.print(table)
            console.print("  [dim]w=[1.0, -1.0]  grad=[0.5, -0.3][/dim]")
            console.print(
                Panel(
                    "[bold]Adam[/bold] adapts per-parameter learning rates and is the standard "
                    "for training Transformers.\n"
                    "[bold]AdamW[/bold] = Adam + weight decay (decoupled regularization).",
                    title="Which optimizer to use?",
                    border_style="green",
                )
            )

        _pause()


# ── AI Agents ───────────────────────────────────────────────────────────


def _explore_agents() -> None:
    """Interactive explorer for the AI agents module."""
    while True:
        choice = _select(
            "AI Agents",
            [
                "Agent overview",
                "Available tools",
                "Architecture diagram",
                "Try calculator tool",
                "Try memory system",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

        if choice == "Agent overview":
            console.print()
            table = Table(title="Agent Types", border_style="cyan")
            table.add_column("Agent", style="bold")
            table.add_column("Pattern")
            table.add_column("Best for")
            table.add_column("Complexity")

            table.add_row(
                "ToolAgent",
                "Direct tool dispatch",
                "Simple queries (1-2 tool calls)",
                "[green]Low[/green]",
            )
            table.add_row(
                "ReActAgent",
                "Think -> Act -> Observe loop",
                "Multi-step research tasks",
                "[yellow]Medium[/yellow]",
            )
            table.add_row(
                "MultiAgentSupervisor",
                "Supervisor + specialists",
                "Complex tasks needing diverse skills",
                "[red]High[/red]",
            )
            console.print(table)

            console.print(
                Panel(
                    "All agents use [bold]Ollama[/bold] for local LLM inference.\n"
                    "No API keys needed. Run [cyan]ollama serve[/cyan] then "
                    "[cyan]ollama pull llama3.2[/cyan] to get started.",
                    title="Getting started",
                    border_style="green",
                )
            )

        elif choice == "Available tools":
            console.print()
            table = Table(title="Agent Tools", border_style="cyan")
            table.add_column("Tool", style="bold")
            table.add_column("Description")
            table.add_column("Requires")

            table.add_row("web_search", "DuckDuckGo search (titles, URLs, snippets)", "Internet")
            table.add_row("calculator", "Safe math evaluation via AST parsing", "Nothing")
            table.add_row("web_scrape", "Extract text from URLs (httpx + BS4)", "Internet")
            table.add_row("get_current_time", "Current date and time (UTC)", "Nothing")
            table.add_row("text_length", "Count chars, words, lines in text", "Nothing")
            table.add_row("unit_converter", "Temperature, distance, weight conversions", "Nothing")
            console.print(table)

        elif choice == "Architecture diagram":
            console.print()
            diagram = """\
ReAct Agent Loop:
                          ┌─────────┐
                          │  START   │
                          └────┬─────┘
                               │
                               ▼
                          ┌─────────┐
                   ┌──────│  THINK  │──────┐
                   │      └─────────┘      │
                   │ "I should search"     │ "I have the answer"
                   ▼                       ▼
              ┌─────────┐            ┌─────────┐
              │   ACT   │            │  FINAL  │
              │ (tools) │            │ ANSWER  │
              └────┬────┘            └─────────┘
                   │
                   ▼
              ┌─────────┐
              │ OBSERVE │──── loop back to THINK
              └─────────┘"""
            console.print(Panel(diagram, title="ReAct Agent Architecture", border_style="cyan"))

        elif choice == "Try calculator tool":
            _try_calculator_tool()

        elif choice == "Try memory system":
            _try_memory_system()

        _pause()


def _try_calculator_tool() -> None:
    """Demonstrate the calculator tool like an agent would use it."""
    from agentexplorr.agents.tools.calculator import safe_evaluate

    console.print()
    queries = [
        ("What is 15% of 280?", "280 * 0.15"),
        ("Square root of 144 plus pi", "sqrt(144) + pi"),
        ("Compound interest: $1000 at 5% for 10 years", "1000 * (1 + 0.05)**10"),
        ("Factorial of 7 divided by 2", "factorial(7) / 2"),
    ]

    table = Table(title="Calculator Tool Demo — Agent-style", border_style="cyan")
    table.add_column("Query", style="bold")
    table.add_column("Tool call")
    table.add_column("Result", style="green", justify="right")

    for question, expr in queries:
        result = safe_evaluate(expr)
        table.add_row(question, f"calculator({expr!r})", str(round(result, 4)))

    console.print(table)
    console.print(
        "[dim]The agent translates natural language into tool calls, "
        "then incorporates results into its response.[/dim]"
    )


def _try_memory_system() -> None:
    """Demonstrate the conversation memory system."""
    from agentexplorr.agents.memory import ConversationMemory

    console.print()
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("What is machine learning?")
    memory.add_assistant_message(
        "Machine learning is a subset of AI where models learn patterns from data."
    )
    memory.add_user_message("How does supervised learning work?")
    memory.add_assistant_message(
        "Supervised learning uses labeled data — input/output pairs — to train "
        "a model to predict outputs for new inputs."
    )
    memory.add_user_message("Give me an example.")
    memory.add_assistant_message(
        "Email spam detection: the model sees thousands of emails labeled "
        "'spam' or 'not spam' and learns to classify new emails."
    )

    console.print(
        Panel(
            memory.get_context_string(),
            title="Conversation Memory (3 turns)",
            border_style="cyan",
        )
    )

    results = memory.search("supervised")
    console.print(f"\n  [bold]Search for 'supervised':[/bold] found {len(results)} messages")
    for msg in results:
        console.print(f"    [{msg.role}] {msg.content[:80]}...")

    console.print(
        Panel(
            "Rolling window  |  Searchable  |  JSON-serializable  |  LangChain-exportable",
            title="Memory features",
            border_style="green",
        )
    )


# ── RAG Pipeline ────────────────────────────────────────────────────────


def _explore_rag() -> None:
    """Interactive explorer for the RAG pipeline."""
    while True:
        choice = _select(
            "RAG Pipeline",
            [
                "Pipeline overview",
                "Chunking strategies",
                "Vector store comparison",
                "Try chunking demo",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

        if choice == "Pipeline overview":
            console.print()
            flow = """\
RAG Flow:
  ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
  │  LOAD    │──>│  CHUNK  │──>│  EMBED  │──>│  STORE  │──>│ RETRIEVE │──>│ GENERATE │
  │ (files)  │   │ (split) │   │(vectors)│   │ (index) │   │ (search) │   │  (LLM)   │
  └──────────┘   └─────────┘   └─────────┘   └─────────┘   └──────────┘   └──────────┘
  PDF, MD, TXT   Recursive     all-MiniLM    ChromaDB      Hybrid         Ollama
                 or semantic    -L6-v2        or FAISS      + re-rank      llama3.2"""
            console.print(Panel(flow, title="End-to-End RAG", border_style="cyan"))

            console.print(
                Panel(
                    'The prompt instructs the LLM: "If the answer is not in the context, '
                    "say 'I don't know.'\" This reduces hallucination by giving the model "
                    "permission to abstain.",
                    title="Key design: anti-hallucination prompt",
                    border_style="green",
                )
            )

        elif choice == "Chunking strategies":
            console.print()
            table = Table(title="Chunking Strategies", border_style="cyan")
            table.add_column("Strategy", style="bold")
            table.add_column("How it works")
            table.add_column("Best for")

            table.add_row(
                "RecursiveChunker",
                "Split by paragraphs, then sentences, then chars",
                "General purpose (default)",
            )
            table.add_row(
                "FixedChunker",
                "Fixed character windows with overlap",
                "Uniform chunk sizes",
            )
            table.add_row(
                "SemanticChunker",
                "Split at topic boundaries using embeddings",
                "Topic-coherent chunks",
            )
            console.print(table)

        elif choice == "Vector store comparison":
            console.print()
            table = Table(title="Vector Stores", border_style="cyan")
            table.add_column("Store", style="bold")
            table.add_column("Index type")
            table.add_column("Persistence")
            table.add_column("Best for")

            table.add_row("ChromaDB", "HNSW", "Built-in (SQLite)", "Ease of use, small-medium data")
            table.add_row("FAISS", "Flat / IVF", "Manual save/load", "Speed, large-scale search")
            console.print(table)

            console.print(
                Panel(
                    "The [bold]HybridRetriever[/bold] queries both stores and re-ranks "
                    "results using Reciprocal Rank Fusion (RRF). This gives higher recall "
                    "than either store alone.",
                    title="Hybrid retrieval",
                    border_style="green",
                )
            )

        elif choice == "Try chunking demo":
            _try_chunking_demo()

        _pause()


def _try_chunking_demo() -> None:
    """Demonstrate text chunking with real content."""
    from agentexplorr.rag.chunking import FixedSizeChunker, RecursiveChunker

    console.print()

    sample_text = (
        "Machine learning is a branch of artificial intelligence that focuses on "
        "building systems that learn from data. Unlike traditional programming where "
        "you write explicit rules, ML algorithms discover patterns automatically.\n\n"
        "There are three main types of machine learning:\n\n"
        "Supervised Learning uses labeled training data — input/output pairs — to "
        "learn a mapping function. Common algorithms include linear regression, "
        "decision trees, and neural networks. Applications include spam detection, "
        "image classification, and medical diagnosis.\n\n"
        "Unsupervised Learning works with unlabeled data to discover hidden structure. "
        "Clustering algorithms like K-Means group similar data points together. "
        "Dimensionality reduction techniques like PCA compress data while preserving "
        "important information.\n\n"
        "Reinforcement Learning trains agents through trial and error. The agent takes "
        "actions in an environment and receives rewards or penalties. This approach "
        "powers game-playing AI, robotics, and recommendation systems."
    )

    console.print(
        Panel(
            f"[dim]{sample_text[:200]}...[/dim]",
            title=f"Sample text ({len(sample_text)} chars)",
            border_style="dim",
        )
    )

    # Fixed-size chunking
    fixed = FixedSizeChunker(chunk_size=200, overlap=30)
    fixed_chunks = fixed.chunk(sample_text)

    # Recursive chunking
    recursive = RecursiveChunker(chunk_size=200, overlap=30)
    recursive_chunks = recursive.chunk(sample_text)

    table = Table(title="Chunking Comparison", border_style="cyan")
    table.add_column("Strategy", style="bold")
    table.add_column("Chunks", justify="center")
    table.add_column("Avg size", justify="center")
    table.add_column("First chunk preview")

    def _avg(chunks: list) -> int:
        return sum(len(c.text) for c in chunks) // max(len(chunks), 1)

    table.add_row(
        "FixedSize (200, overlap=30)",
        str(len(fixed_chunks)),
        f"{_avg(fixed_chunks)} chars",
        fixed_chunks[0].text[:60] + "..." if fixed_chunks else "—",
    )
    table.add_row(
        "Recursive (200, overlap=30)",
        str(len(recursive_chunks)),
        f"{_avg(recursive_chunks)} chars",
        recursive_chunks[0].text[:60] + "..." if recursive_chunks else "—",
    )

    console.print(table)

    # Show all recursive chunks
    console.print("\n  [bold]Recursive chunks detail:[/bold]")
    for i, chunk in enumerate(recursive_chunks):
        preview = chunk.text[:80].replace("\n", " ")
        console.print(f"    [dim]#{i + 1}[/dim] ({len(chunk.text)} chars) {preview}...")

    console.print(
        Panel(
            "[bold]Recursive chunking[/bold] preserves paragraph boundaries, "
            "producing more coherent chunks than fixed-size splitting.\n"
            "This directly improves retrieval quality in RAG pipelines.",
            title="Why recursive chunking wins",
            border_style="green",
        )
    )


# ── Prompt Engineering ──────────────────────────────────────────────────


def _explore_prompts() -> None:
    """Interactive explorer for prompt engineering techniques."""
    while True:
        choice = _select(
            "Prompt Engineering",
            [
                "Chain of Thought demo",
                "Techniques overview",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

        if choice == "Chain of Thought demo":
            console.print()
            from agentexplorr.prompt_engineering.chain_of_thought import (
                ChainOfThoughtPrompt,
                ReasoningStep,
            )

            cot = ChainOfThoughtPrompt(task="Solve the math problem step by step.")
            cot.add_example(
                question="What is 15% of 80?",
                reasoning=[
                    ReasoningStep(1, "Convert percentage", "15% = 0.15"),
                    ReasoningStep(2, "Multiply", "0.15 x 80 = 12"),
                ],
                answer="12",
            )

            prompt = cot.build_few_shot("What is 25% of 200?")
            console.print(
                Panel(
                    Syntax(prompt, "text", theme="monokai", word_wrap=True),
                    title="Generated Few-Shot CoT Prompt",
                    border_style="cyan",
                )
            )
            console.print(
                "[dim]This prompt teaches the LLM to show its reasoning "
                "by providing an example with explicit steps.[/dim]"
            )

        elif choice == "Techniques overview":
            console.print()
            table = Table(title="Prompt Engineering Techniques", border_style="cyan")
            table.add_column("Technique", style="bold")
            table.add_column("Description")
            table.add_column("When to use")

            table.add_row(
                "Zero-shot CoT",
                '"Let\'s think step by step"',
                "Math, reasoning (no examples needed)",
            )
            table.add_row(
                "Few-shot CoT",
                "Examples with reasoning chains",
                "Complex tasks (teach reasoning style)",
            )
            table.add_row(
                "Self-consistency",
                "N paths + majority vote",
                "High-stakes answers",
            )
            table.add_row(
                "Jinja2 Templates",
                "Reusable prompts with variables",
                "Production prompt management",
            )
            table.add_row(
                "Structured Output",
                "Pydantic models for parsing",
                "Extract JSON/data from LLM output",
            )
            console.print(table)

        _pause()


# ── Classical ML ────────────────────────────────────────────────────────


def _explore_classical_ml() -> None:
    """Interactive explorer for classical ML pipelines."""
    while True:
        choice = _select(
            "Classical ML",
            [
                "Pipeline overview",
                "Run classification demo (Wine dataset)",
                "Run clustering demo (Iris dataset)",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

        if choice == "Pipeline overview":
            console.print()
            table = Table(title="ML Pipeline Types", border_style="cyan")
            table.add_column("Pipeline", style="bold")
            table.add_column("Task")
            table.add_column("Algorithm")
            table.add_column("Dataset")

            table.add_row(
                "ClassificationPipeline",
                "Predict discrete labels",
                "Random Forest + GridSearchCV",
                "Wine Quality (178 samples)",
            )
            table.add_row(
                "RegressionPipeline",
                "Predict continuous values",
                "Ridge / Lasso / ElasticNet",
                "California Housing (20K samples)",
            )
            table.add_row(
                "ClusteringPipeline",
                "Discover groups",
                "KMeans / DBSCAN + PCA",
                "Iris (150 samples)",
            )
            console.print(table)

            console.print(
                Panel(
                    "Each pipeline includes: [bold]preprocessing[/bold] (StandardScaler, "
                    "OneHotEncoder) -> [bold]model[/bold] -> [bold]evaluation[/bold] "
                    "(accuracy, R², silhouette score).\n"
                    "All wrapped in sklearn's Pipeline to prevent data leakage.",
                    title="Architecture",
                    border_style="green",
                )
            )

        elif choice == "Run classification demo (Wine dataset)":
            _try_classification()

        elif choice == "Run clustering demo (Iris dataset)":
            _try_clustering()

        _pause()


def _try_classification() -> None:
    """Run a live classification pipeline on the Wine dataset."""
    from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline

    console.print()
    console.print("  [bold]Running classification pipeline on Wine dataset...[/bold]")

    pipeline = ClassificationPipeline()
    metrics = pipeline.train_on_wine()

    table = Table(title="Classification Results (Wine Dataset)", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")

    console.print(table)

    # Feature importances
    importances = pipeline.get_feature_importances()
    console.print("\n  [bold]Top 5 feature importances:[/bold]")
    for name, score in list(importances.items())[:5]:
        bar = "[green]" + "#" * int(score * 40) + "[/green]"
        console.print(f"    {name:<25} {score:.4f} {bar}")


def _try_clustering() -> None:
    """Run a live clustering pipeline on the Iris dataset."""
    from agentexplorr.classical_ml.pipelines.clustering import ClusteringPipeline

    console.print()
    console.print("  [bold]Running KMeans clustering on Iris dataset (K=3)...[/bold]")

    pipeline = ClusteringPipeline(n_clusters=3)
    metrics = pipeline.fit_on_iris()

    table = Table(title="Clustering Results (Iris Dataset)", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
    table.add_row("Clusters Found", str(metrics["n_clusters"]))
    table.add_row("Noise Points", str(metrics["n_noise_points"]))

    for label, size in metrics.get("cluster_sizes", {}).items():
        table.add_row(f"  Cluster {label} size", str(size))

    console.print(table)

    console.print(
        Panel(
            "Silhouette > 0.5 = reasonable clusters. The Iris dataset has 3 species, "
            "and KMeans typically discovers them with high accuracy.\n"
            "Try DBSCAN for non-spherical cluster shapes.",
            title="Interpreting results",
            border_style="green",
        )
    )


# ── Interactive Calculator ──────────────────────────────────────────────


def _interactive_calc() -> None:
    """REPL-style calculator — type expressions, get results."""
    from agentexplorr.agents.tools.calculator import SafeExpressionError, safe_evaluate

    console.print()
    console.print(
        Panel(
            "Type math expressions and press Enter.\n"
            "Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e, factorial, ...\n"
            'Type [bold]"back"[/bold] to return to the main menu.',
            title="Calculator",
            border_style="cyan",
        )
    )

    while True:
        try:
            expr = questionary.text("calc>").ask()
        except (KeyboardInterrupt, EOFError):
            return

        if expr is None or expr.strip().lower() in ("back", "quit", "exit", "q"):
            return

        if not expr.strip():
            continue

        try:
            result = safe_evaluate(expr.strip())
            console.print(f"  [bold green]= {result}[/bold green]")
        except SafeExpressionError as e:
            console.print(f"  [bold red]Error:[/bold red] {e}")


# ── Benchmarks ──────────────────────────────────────────────────────────


def _show_benchmarks() -> None:
    """Display and optionally run benchmark questions."""
    from agentexplorr.agents.evaluation.benchmarks import (
        get_math_questions,
        get_search_questions,
    )

    while True:
        choice = _select(
            "Benchmarks",
            [
                "View all questions",
                "Run quick benchmark (calculator agent)",
                BACK,
            ],
        )
        if choice is None or choice == BACK:
            return

        if choice == "View all questions":
            console.print()
            math_qs = get_math_questions()
            search_qs = get_search_questions()

            table = Table(
                title=f"Agent Benchmark Suite ({len(math_qs) + len(search_qs)} questions)",
                border_style="cyan",
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Category", style="bold")
            table.add_column("Difficulty")
            table.add_column("Question")
            table.add_column("Expected Answer", style="green")

            for i, q in enumerate(math_qs, 1):
                diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
                    q.difficulty, "white"
                )
                table.add_row(
                    str(i),
                    "math",
                    f"[{diff_color}]{q.difficulty}[/{diff_color}]",
                    q.question,
                    q.expected_answer,
                )

            for i, q in enumerate(search_qs, len(math_qs) + 1):
                diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
                    q.difficulty, "white"
                )
                table.add_row(
                    str(i),
                    "search",
                    f"[{diff_color}]{q.difficulty}[/{diff_color}]",
                    q.question,
                    q.expected_answer,
                )

            console.print(table)

        elif choice == "Run quick benchmark (calculator agent)":
            _run_quick_benchmark()

        _pause()


def _run_quick_benchmark() -> None:
    """Run a quick benchmark using the calculator tool as a fake agent."""
    from agentexplorr.agents.evaluation.benchmarks import (
        AgentBenchmark,
        BenchmarkQuestion,
    )
    from agentexplorr.agents.tools.calculator import safe_evaluate

    console.print()
    console.print("  [bold]Running benchmark with calculator agent...[/bold]\n")

    # Build a simple agent that uses the calculator for math questions
    @dataclass
    class _CalcResult:
        final_answer: str
        tools_used: list[str]

    class _CalcAgent:
        """A simple agent that evaluates math expressions directly."""

        def run(self, query: str) -> _CalcResult:
            # Extract the mathematical expression from the question
            # For math benchmark questions, try to evaluate directly
            try:
                result = safe_evaluate(query)
                return _CalcResult(final_answer=str(result), tools_used=["calculator"])
            except Exception:
                return _CalcResult(final_answer="I don't know", tools_used=[])

    questions = [
        BenchmarkQuestion("What is 2 + 2?", "4", category="math", match_type="numeric"),
        BenchmarkQuestion("sqrt(144)", "12", category="math", match_type="numeric"),
        BenchmarkQuestion("3 * 7 + 1", "22", category="math", match_type="numeric"),
        BenchmarkQuestion("2 ** 10", "1024", category="math", match_type="numeric"),
        BenchmarkQuestion("100 / 4", "25", category="math", match_type="numeric"),
    ]

    benchmark = AgentBenchmark(verbose=False)
    summary = benchmark.run(_CalcAgent(), questions)

    table = Table(title="Benchmark Results", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Questions", str(summary.total_questions))
    table.add_row("Correct", str(summary.correct))
    table.add_row("Accuracy", f"{summary.accuracy:.1%}")
    table.add_row("Errors", str(summary.errors))

    console.print(table)

    if summary.tool_usage:
        console.print(f"\n  [dim]Tools used: {dict(summary.tool_usage)}[/dim]")


# ── Project Info ────────────────────────────────────────────────────────


def _show_project_info() -> None:
    """Display project modules, tech stack, and architecture."""
    console.print()

    # Modules table
    table = Table(title="AgentExplorr Modules", border_style="cyan")
    table.add_column("Module", style="bold")
    table.add_column("Description")
    table.add_column("Key tech")

    table.add_row("foundations", "Math behind ML (from scratch)", "NumPy")
    table.add_row("agents", "ReAct, Tool-Calling, Multi-Agent", "LangGraph + Ollama")
    table.add_row("mcp", "Model Context Protocol servers", "MCP SDK")
    table.add_row("llm_training", "LoRA / QLoRA fine-tuning", "PEFT + TRL + HuggingFace")
    table.add_row("classical_ml", "Pipelines, CNNs, Transformers", "sklearn + PyTorch + MLflow")
    table.add_row("rag", "Document -> Retrieve -> Generate", "ChromaDB + FAISS + Ollama")
    table.add_row("prompt_engineering", "Templates, CoT, structured output", "Jinja2 + Pydantic")
    console.print(table)

    # Tech stack
    console.print()
    stack = Table(title="Tech Stack", border_style="green")
    stack.add_column("Category", style="bold")
    stack.add_column("Technology")

    stack.add_row("LLM Inference", "Ollama (local, no API keys)")
    stack.add_row("Agent Framework", "LangGraph + LangChain")
    stack.add_row("Embeddings", "sentence-transformers (all-MiniLM-L6-v2)")
    stack.add_row("Vector Stores", "ChromaDB + FAISS")
    stack.add_row("Fine-tuning", "PEFT (LoRA/QLoRA) + TRL")
    stack.add_row("Experiment Tracking", "MLflow")
    stack.add_row("Package Manager", "uv")
    stack.add_row("Linter/Formatter", "Ruff")
    stack.add_row("Type Checker", "mypy (strict mode)")
    stack.add_row("Config", "Pydantic Settings (.env)")
    stack.add_row("Logging", "structlog")
    console.print(stack)

    _pause()


# ── Argparse (backward-compatible script mode) ──────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for script mode."""
    parser = argparse.ArgumentParser(
        prog="agentexplorr",
        description="AgentExplorr — AI/ML Playground. Run without arguments for interactive mode.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"agentexplorr {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # calc (script mode)
    sub_calc = subparsers.add_parser("calc", help="Evaluate a math expression")
    sub_calc.add_argument("expression", nargs="+", help="Math expression to evaluate")

    # info (script mode)
    subparsers.add_parser("info", help="Show project info")

    # benchmark (script mode)
    subparsers.add_parser("benchmark", help="Show benchmark questions")

    # foundations (script mode)
    subparsers.add_parser("foundations", help="Demo math foundations")

    return parser


def _script_calc(args: argparse.Namespace) -> None:
    """Script-mode calculator."""
    from agentexplorr.agents.tools.calculator import SafeExpressionError, safe_evaluate

    expression = " ".join(args.expression)
    try:
        result = safe_evaluate(expression)
        print(f"  {expression} = {result}")
    except SafeExpressionError as e:
        print(f"  Error: {e}")
        sys.exit(1)


def _script_info() -> None:
    """Script-mode info."""
    _show_project_info()


def _script_benchmark() -> None:
    """Script-mode benchmark listing."""
    _show_benchmarks()


def _script_foundations() -> None:
    """Script-mode foundations demo."""
    _explore_foundations()


# ── Entry point ─────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point.

    - No arguments: launch interactive mode
    - With arguments: run in script mode (backward compatible)
    """
    # If called with explicit argv=[] (like from tests), parse those args
    # If called with no args from the command line, check sys.argv
    args_to_parse = list(argv) if argv is not None else sys.argv[1:]

    # No arguments -> interactive mode
    if not args_to_parse:
        try:
            interactive_mode()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]\n")
        return

    # Has arguments -> script mode
    parser = build_parser()
    args = parser.parse_args(args_to_parse)

    handlers = {
        "calc": lambda: _script_calc(args),
        "info": _script_info,
        "benchmark": _script_benchmark,
        "foundations": _script_foundations,
    }

    handler = handlers.get(args.command)
    if handler:
        handler()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
