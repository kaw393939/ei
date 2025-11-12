"""
Integration tests for CLI commands using ServiceFactory mocks.

Tests command argument parsing, error handling, and output formatting
with the plugin-based architecture.
"""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from ei_cli.plugins.image import image
from ei_cli.plugins.search import search
from ei_cli.services.ai_service import (
    ImageGenerationResult,
    SearchCitation,
    SearchResult,
)
from ei_cli.services.base import ServiceError


class TestImageCommand:
    """Tests for image generation command."""

    @patch("ei_cli.plugins.image.ServiceFactory")
    def test_image_basic_generation(self, mock_factory_class: Mock) -> None:
        """Test basic image generation."""
        mock_service = Mock()
        mock_service.check_available.return_value = (True, None)
        mock_service.generate_image.return_value = ImageGenerationResult(
            image_url="https://example.com/image.png",
            model="dall-e-3",
        )
        mock_factory = Mock()
        mock_factory.get_ai_service.return_value = mock_service
        mock_factory_class.return_value = mock_factory

        runner = CliRunner()
        result = runner.invoke(image, ["a sunset"])

        assert result.exit_code == 0
        assert "https://" in result.output or "Generated" in result.output

    @patch("ei_cli.plugins.image.ServiceFactory")
    def test_image_service_unavailable(
        self,
        mock_factory_class: Mock,
    ) -> None:
        """Test handling when service is unavailable."""
        mock_service = Mock()
        mock_service.check_available.return_value = (
            False,
            "Missing API key",
        )
        mock_factory = Mock()
        mock_factory.get_ai_service.return_value = mock_service
        mock_factory_class.return_value = mock_factory

        runner = CliRunner()
        result = runner.invoke(image, ["test"])

        assert result.exit_code == 1
        assert "not available" in result.output

    @patch("ei_cli.plugins.image.ServiceFactory")
    def test_image_service_error(self, mock_factory_class: Mock) -> None:
        """Test handling service errors."""
        mock_service = Mock()
        mock_service.check_available.return_value = (True, None)
        mock_service.generate_image.side_effect = ServiceError(
            message="API failed",
            service_name="ai_service",
        )
        mock_factory = Mock()
        mock_factory.get_ai_service.return_value = mock_service
        mock_factory_class.return_value = mock_factory

        runner = CliRunner()
        result = runner.invoke(image, ["test"])

        assert result.exit_code == 1

    def test_image_help(self) -> None:
        """Test image command help."""
        runner = CliRunner()
        result = runner.invoke(image, ["--help"])

        assert result.exit_code == 0
        assert "Generate" in result.output or "image" in result.output


class TestSearchCommand:
    """Tests for search command."""

    @patch("ei_cli.plugins.search.ServiceFactory")
    def test_search_basic(self, mock_factory_class: Mock) -> None:
        """Test basic search."""
        mock_service = Mock()
        mock_service.check_available.return_value = (True, None)
        mock_service.search.return_value = SearchResult(
            answer="Test answer",
            citations=[
                SearchCitation(
                    url="https://example.com",
                    title="Example",
                    start_index=0,
                    end_index=10,
                ),
            ],
            sources=["https://example.com"],
        )
        mock_factory = Mock()
        mock_factory.get_ai_service.return_value = mock_service
        mock_factory_class.return_value = mock_factory

        runner = CliRunner()
        result = runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert (
            "answer" in result.output.lower()
            or "Test answer" in result.output
        )

    @patch("ei_cli.plugins.search.ServiceFactory")
    def test_search_with_domains(self, mock_factory_class: Mock) -> None:
        """Test search with domain restrictions."""
        mock_service = Mock()
        mock_service.check_available.return_value = (True, None)
        mock_service.search.return_value = SearchResult(
            answer="Domain-specific answer",
            citations=[],
            sources=[],
        )
        mock_factory = Mock()
        mock_factory.get_ai_service.return_value = mock_service
        mock_factory_class.return_value = mock_factory

        runner = CliRunner()
        result = runner.invoke(
            search,
            ["test", "--domains", "example.com"],
        )

        assert result.exit_code == 0
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs["allowed_domains"] == ["example.com"]

    def test_search_help(self) -> None:
        """Test search command help."""
        runner = CliRunner()
        result = runner.invoke(search, ["--help"])

        assert result.exit_code == 0
        assert "search" in result.output.lower()
