"""
Tests for the CLI entry point.
"""

from __future__ import annotations

from agentexplorr.cli import build_parser, main


class TestCLI:
    def test_parser_has_subcommands(self) -> None:
        parser = build_parser()
        # Should not raise
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_calc_command(self, capsys: object) -> None:
        main(["calc", "2 + 3"])
        # capsys is a pytest fixture; we just verify no exception is raised

    def test_info_command(self) -> None:
        # Should not raise
        main(["info"])

    def test_benchmark_command(self) -> None:
        # Should not raise
        main(["benchmark"])

    def test_no_command_shows_info(self) -> None:
        # No command defaults to info
        main([])
