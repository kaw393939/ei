"""
Smoke tests for the main CLI application with plugin system.
"""
from click.testing import CliRunner

from ei_cli.cli.app import cli, main


class TestCLIApp:
    """Test main CLI application."""

    def test_cli_help(self):
        """Test CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "EverydayAI CLI" in result.output
        assert "Personal AI toolkit for regular people" in result.output

    def test_cli_version(self):
        """Test CLI shows version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # Version check - accept different CLI name formats
        assert "0.1." in result.output and "version" in result.output

    def test_cli_has_image_command(self):
        """Test image command is registered via plugin."""
        runner = CliRunner()
        result = runner.invoke(cli, ["image", "--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower()

    def test_cli_has_search_command(self):
        """Test search command is registered via plugin."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower()

    def test_cli_has_vision_command(self):
        """Test vision command is registered via plugin."""
        runner = CliRunner()
        result = runner.invoke(cli, ["vision", "--help"])
        assert result.exit_code == 0
        assert "vision" in result.output.lower()

    def test_cli_has_speak_command(self):
        """Test speak command is registered via plugin."""
        runner = CliRunner()
        result = runner.invoke(cli, ["speak", "--help"])
        assert result.exit_code == 0
        assert "speak" in result.output.lower()

    def test_cli_has_transcribe_command(self):
        """Test transcribe command is registered via plugin."""
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower()

    def test_cli_invalid_command(self):
        """Test CLI handles invalid command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0
        assert "Error" in result.output or "No such command" in result.output

    def test_main_entry_point(self):
        """Test main entry point function."""
        # Just verify it's callable - actual execution would need mocking
        assert callable(main)
        assert main.__name__ == "main"

    def test_plugins_loaded(self):
        """Test that plugins are loaded dynamically."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Check that at least some plugin commands are available
        output = result.output.lower()
        assert "vision" in output or "image" in output or "speak" in output
