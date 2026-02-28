"""
Tests for the CLI entry point.

Tests cover both script mode (backward-compatible argparse) and
the interactive mode functions (with questionary mocked out).
"""

from __future__ import annotations

from unittest.mock import patch

from agentexplorr.cli import (
    _explore_agents,
    _explore_foundations,
    _explore_prompts,
    _explore_rag,
    _interactive_calc,
    _show_benchmarks,
    _show_project_info,
    build_parser,
    interactive_mode,
    main,
)

# ── Script-mode tests ───────────────────────────────────────────────────


class TestScriptMode:
    """Tests for backward-compatible argparse CLI."""

    def test_parser_has_subcommands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["calc", "2+3"])
        assert args.command == "calc"

    def test_calc_command(self) -> None:
        main(["calc", "2 + 3"])

    def test_calc_complex_expression(self) -> None:
        main(["calc", "sqrt(144)", "+", "pi"])

    def test_info_command(self) -> None:
        main(["info"])

    def test_benchmark_command(self) -> None:
        main(["benchmark"])


# ── Interactive mode tests (with mocked questionary) ────────────────────


_P_SELECT = "agentexplorr.cli._select"
_P_PAUSE = "agentexplorr.cli._pause"


class TestInteractiveMode:
    """Tests for the interactive console.

    We mock questionary.select and questionary.text so we don't need
    a real terminal. Each test verifies the function runs without error.
    """

    def test_interactive_mode_quit_immediately(self) -> None:
        with patch(_P_SELECT, return_value="Quit"):
            interactive_mode()

    def test_interactive_mode_ctrl_c(self) -> None:
        with patch(_P_SELECT, return_value=None):
            interactive_mode()

    def test_show_project_info(self) -> None:
        with patch(_P_PAUSE):
            _show_project_info()

    def test_show_benchmarks(self) -> None:
        with patch(_P_PAUSE):
            _show_benchmarks()

    def test_explore_agents_overview_then_back(self) -> None:
        with patch(_P_SELECT, side_effect=["Agent overview", "Back"]), patch(_P_PAUSE):
            _explore_agents()

    def test_explore_agents_tools(self) -> None:
        with patch(_P_SELECT, side_effect=["Available tools", "Back"]), patch(_P_PAUSE):
            _explore_agents()

    def test_explore_agents_architecture(self) -> None:
        with patch(_P_SELECT, side_effect=["Architecture diagram", "Back"]), patch(_P_PAUSE):
            _explore_agents()

    def test_explore_rag_overview(self) -> None:
        with patch(_P_SELECT, side_effect=["Pipeline overview", "Back"]), patch(_P_PAUSE):
            _explore_rag()

    def test_explore_rag_chunking(self) -> None:
        with patch(_P_SELECT, side_effect=["Chunking strategies", "Back"]), patch(_P_PAUSE):
            _explore_rag()

    def test_explore_rag_vector_stores(self) -> None:
        with patch(_P_SELECT, side_effect=["Vector store comparison", "Back"]), patch(_P_PAUSE):
            _explore_rag()

    def test_explore_foundations_activations(self) -> None:
        with patch(_P_SELECT, side_effect=["Activation functions", "Back"]), patch(_P_PAUSE):
            _explore_foundations()

    def test_explore_foundations_loss(self) -> None:
        with patch(_P_SELECT, side_effect=["Loss functions", "Back"]), patch(_P_PAUSE):
            _explore_foundations()

    def test_explore_foundations_optimizers(self) -> None:
        with patch(_P_SELECT, side_effect=["Optimizer comparison", "Back"]), patch(_P_PAUSE):
            _explore_foundations()

    def test_explore_foundations_all(self) -> None:
        with patch(_P_SELECT, side_effect=["All at once", "Back"]), patch(_P_PAUSE):
            _explore_foundations()

    def test_explore_prompts_cot(self) -> None:
        with patch(_P_SELECT, side_effect=["Chain of Thought demo", "Back"]), patch(_P_PAUSE):
            _explore_prompts()

    def test_explore_prompts_overview(self) -> None:
        with patch(_P_SELECT, side_effect=["Techniques overview", "Back"]), patch(_P_PAUSE):
            _explore_prompts()

    def test_interactive_calc_and_back(self) -> None:
        with patch("questionary.text") as mock_text:
            mock_text.return_value.ask.side_effect = ["2 + 3", "sqrt(144)", "back"]
            _interactive_calc()

    def test_interactive_calc_error(self) -> None:
        with patch("questionary.text") as mock_text:
            mock_text.return_value.ask.side_effect = ["import os", "back"]
            _interactive_calc()

    def test_interactive_calc_ctrl_c(self) -> None:
        with patch("questionary.text") as mock_text:
            mock_text.return_value.ask.side_effect = KeyboardInterrupt
            _interactive_calc()

    def test_no_args_launches_interactive(self) -> None:
        with patch("agentexplorr.cli.interactive_mode") as mock_interactive:
            main([])
            mock_interactive.assert_called_once()
