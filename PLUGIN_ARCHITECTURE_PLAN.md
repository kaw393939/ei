# Plugin Architecture Refactoring Plan

## Executive Summary

Transform EverydayAI CLI from a monolithic command structure to a flexible plugin-based architecture that supports:
- Dynamic command discovery and loading
- Easy third-party extensions  
- Clean separation of concerns
- Excellent developer experience
- Seamless help system integration
- Future MCP server compatibility

**Timeline**: 2-3 days
**Current Status**: Branch `feature/plugin-architecture` created

---

## Current Architecture Analysis

### ✅ What's Already Good

1. **Service Layer** - Well-designed with factory pattern
   - `ServiceFactory` manages service instances
   - Clean separation between services (`AIService`, `ImageService`)
   - Easy to extend with new services

2. **Command Structure** - Each command is independent
   - Located in `src/ei_cli/cli/commands/`
   - Uses Click decorators
   - Minimal coupling between commands

3. **Configuration** - Centralized and typed
   - Pydantic-based settings
   - Environment variable support
   - Easy to extend

### ⚠️ Current Issues

1. **Hardcoded Command Registration** in `app.py`:
   ```python
   cli.add_command(crop)
   cli.add_command(image)
   cli.add_command(remove_bg)
   # ... etc - 12 commands manually registered
   ```

2. **No Dynamic Discovery** - Adding new commands requires code changes

3. **Commands to Remove**:
   - `crop` - doesn't work well
   - `remove_bg` - doesn't work well

4. **Testing Overhead** - Tests reference removed commands

---

## Plugin Architecture Design

### Core Concepts

```python
# Plugin interface that all commands implement
class CommandPlugin(Protocol):
    """Protocol for command plugins."""
    
    @property
    def name(self) -> str:
        """Command name (e.g., 'vision', 'speak')."""
        
    @property  
    def click_command(self) -> click.Command:
        """The actual Click command object."""
        
    @property
    def enabled(self) -> bool:
        """Whether plugin is enabled."""
        
    def register(self, cli: click.Group) -> None:
        """Register command with CLI."""
```

### Plugin Discovery Mechanisms

#### 1. Entry Points (Recommended)
Best for third-party plugins installed via pip:

```toml
[project.entry-points."ei_cli.plugins"]
vision = "ei_cli.plugins.vision:VisionPlugin"
image = "ei_cli.plugins.image:ImagePlugin"
speak = "ei_cli.plugins.speak:SpeakPlugin"
```

#### 2. Directory Scanning (For Built-in Plugins)
Scan `src/ei_cli/plugins/` for plugin modules:

```python
def discover_builtin_plugins() -> list[CommandPlugin]:
    """Auto-discover plugins in plugins directory."""
    plugin_dir = Path(__file__).parent / "plugins"
    plugins = []
    
    for file in plugin_dir.glob("*.py"):
        if file.stem.startswith("_"):
            continue
        module = import_module(f"ei_cli.plugins.{file.stem}")
        if hasattr(module, "plugin"):
            plugins.append(module.plugin)
    
    return plugins
```

### Proposed Structure

```
src/ei_cli/
├── cli/
│   ├── app.py              # Main CLI with plugin loader
│   └── utils.py            # Shared CLI utilities
├── plugins/                # NEW: Plugin directory
│   ├── __init__.py
│   ├── base.py            # Plugin base classes/protocols
│   ├── loader.py          # Plugin discovery and loading
│   ├── vision.py          # Vision command plugin
│   ├── image.py           # Image generation plugin
│   ├── speak.py           # TTS plugin
│   ├── transcribe.py      # Transcription plugin
│   ├── search.py          # Search plugin
│   └── ...                # Other command plugins
├── services/              # Existing service layer (unchanged)
└── core/                  # Existing core utilities (unchanged)
```

---

## Implementation Plan

### Phase 1: Core Plugin Infrastructure (Day 1)

#### Task 1.1: Create Plugin Base (2 hours)

**File**: `src/ei_cli/plugins/base.py`

```python
"""Plugin base classes and protocols."""
from abc import ABC, abstractmethod
from typing import Protocol
import click


class CommandPlugin(Protocol):
    """Protocol for command plugins."""
    
    @property
    def name(self) -> str:
        """Unique command name."""
        ...
    
    @property
    def click_command(self) -> click.Command:
        """Click command object."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether plugin is enabled."""
        ...
    
    @property
    def category(self) -> str:
        """Plugin category for help organization."""
        ...


class BaseCommandPlugin(ABC):
    """Base implementation of CommandPlugin."""
    
    def __init__(self):
        self._enabled = True
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name."""
        
    @property
    @abstractmethod
    def click_command(self) -> click.Command:
        """Click command."""
        
    @property
    def enabled(self) -> bool:
        """Check if enabled via config."""
        return self._enabled
    
    @property
    def category(self) -> str:
        """Default category."""
        return "General"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} enabled={self.enabled}>"
```

#### Task 1.2: Create Plugin Loader (3 hours)

**File**: `src/ei_cli/plugins/loader.py`

```python
"""Plugin discovery and loading."""
import importlib
import logging
from pathlib import Path
from typing import Iterator

from ei_cli.plugins.base import CommandPlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """Discovers and loads command plugins."""
    
    def __init__(self):
        self._plugins: dict[str, CommandPlugin] = {}
        self._loaded = False
    
    def discover_all(self) -> Iterator[CommandPlugin]:
        """Discover all available plugins."""
        if self._loaded:
            yield from self._plugins.values()
            return
        
        # Load built-in plugins
        yield from self._discover_builtin()
        
        # Load entry point plugins
        yield from self._discover_entry_points()
        
        self._loaded = True
    
    def _discover_builtin(self) -> Iterator[CommandPlugin]:
        """Discover built-in plugins in plugins directory."""
        plugin_dir = Path(__file__).parent
        
        for file in plugin_dir.glob("*.py"):
            # Skip private and base modules
            if file.stem.startswith("_") or file.stem in ("base", "loader"):
                continue
            
            try:
                module = importlib.import_module(f"ei_cli.plugins.{file.stem}")
                if hasattr(module, "plugin"):
                    plugin = module.plugin
                    self._plugins[plugin.name] = plugin
                    logger.debug(f"Loaded built-in plugin: {plugin.name}")
                    yield plugin
            except Exception as e:
                logger.warning(f"Failed to load plugin {file.stem}: {e}")
    
    def _discover_entry_points(self) -> Iterator[CommandPlugin]:
        """Discover plugins via entry points."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points  # Python 3.9 fallback
        
        # Load plugins from ei_cli.plugins entry point
        for ep in entry_points(group="ei_cli.plugins"):
            try:
                plugin = ep.load()
                self._plugins[plugin.name] = plugin
                logger.debug(f"Loaded entry point plugin: {plugin.name}")
                yield plugin
            except Exception as e:
                logger.warning(f"Failed to load entry point {ep.name}: {e}")
    
    def get_plugin(self, name: str) -> CommandPlugin | None:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def get_enabled_plugins(self) -> list[CommandPlugin]:
        """Get all enabled plugins."""
        return [p for p in self._plugins.values() if p.enabled]
```

### Phase 2: Convert Existing Commands to Plugins (Day 1-2)

#### Task 2.1: Convert Vision Command (Template)

**File**: `src/ei_cli/plugins/vision.py`

```python
"""Vision analysis command plugin."""
import click
from ei_cli.cli.commands import vision as vision_cmd
from ei_cli.plugins.base import BaseCommandPlugin


class VisionPlugin(BaseCommandPlugin):
    """Plugin for vision/image analysis."""
    
    @property
    def name(self) -> str:
        return "vision"
    
    @property
    def click_command(self) -> click.Command:
        return vision_cmd.vision
    
    @property
    def category(self) -> str:
        return "Vision & Images"


# Plugin instance for discovery
plugin = VisionPlugin()
```

#### Task 2.2: Convert All Remaining Commands (6-8 hours)

Follow same pattern for:
- ✅ `vision` → `VisionPlugin`
- ✅ `image` → `ImagePlugin`
- ✅ `speak` → `SpeakPlugin` 
- ✅ `speak_elevenlabs` → `ElevenLabsSpeakPlugin`
- ✅ `transcribe` → `TranscribePlugin`
- ✅ `transcribe_video` → `TranscribeVideoPlugin`
- ✅ `translate_audio` → `TranslateAudioPlugin`
- ✅ `search` → `SearchPlugin`
- ✅ `multi_vision` → `MultiVisionPlugin`
- ✅ `setup_youtube` → `YouTubeSetupPlugin`
- ❌ `crop` → **REMOVE**
- ❌ `remove_bg` → **REMOVE**

### Phase 3: Update Main CLI App (Day 2)

#### Task 3.1: Refactor app.py (2 hours)

**File**: `src/ei_cli/cli/app.py`

```python
"""Main CLI application with plugin support."""
import sys
from pathlib import Path
import click

from ei_cli.config import reload_settings
from ei_cli.plugins.loader import PluginLoader

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.1", prog_name="eai")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML/JSON config file",
)
def cli(config: Path | None) -> None:
    """EverydayAI CLI - Personal AI toolkit for regular people."""
    if config:
        try:
            reload_settings(config)
        except Exception as e:
            click.echo(
                click.style(f"❌ Configuration Error: {e}", fg="red"),
                err=True,
            )
            sys.exit(1)


def load_plugins() -> None:
    """Discover and register all plugins."""
    loader = PluginLoader()
    
    # Group plugins by category for better help organization
    categories: dict[str, list] = {}
    
    for plugin in loader.discover_all():
        if not plugin.enabled:
            logger.debug(f"Skipping disabled plugin: {plugin.name}")
            continue
        
        try:
            # Register command
            cli.add_command(plugin.click_command)
            
            # Track for help organization
            category = plugin.category
            if category not in categories:
                categories[category] = []
            categories[category].append(plugin.name)
            
            logger.debug(f"Registered plugin: {plugin.name}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.name}: {e}")
    
    # Log summary
    logger.info(f"Loaded {len(categories)} categories, {sum(len(v) for v in categories.values())} commands")


# Load plugins on import
load_plugins()


def main() -> None:
    """Entry point for poetry script."""
    cli()


if __name__ == "__main__":
    main()
```

### Phase 4: Enhanced Help System (Day 2)

#### Task 4.1: Category-Based Help (2 hours)

```python
# Custom help formatter that groups commands by category
class CategoryHelpFormatter(click.HelpFormatter):
    """Help formatter that groups commands by category."""
    
    def write_dl(self, rows, col_max=30, col_spacing=2):
        """Write definition list grouped by category."""
        # Group by category
        categories = {}
        for name, help_text in rows:
            plugin = loader.get_plugin(name)
            category = plugin.category if plugin else "Other"
            if category not in categories:
                categories[category] = []
            categories[category].append((name, help_text))
        
        # Write each category
        for category, items in sorted(categories.items()):
            self.write_paragraph()
            self.write_text(click.style(category, bold=True))
            super().write_dl(items, col_max, col_spacing)
```

### Phase 5: Testing Updates (Day 3)

#### Task 5.1: Remove Crop/RemoveBG Tests (1 hour)

Delete:
- `tests/python/unit/cli/test_crop.py`
- `tests/python/unit/cli/test_remove_bg.py`
- `tests/python/integration/` references to these commands
- `tests/python/unit/test_app.py` - remove test methods for these commands

#### Task 5.2: Add Plugin System Tests (3 hours)

**File**: `tests/python/unit/plugins/test_loader.py`

```python
"""Tests for plugin loader."""
import pytest
from ei_cli.plugins.loader import PluginLoader
from ei_cli.plugins.base import BaseCommandPlugin


class MockPlugin(BaseCommandPlugin):
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def click_command(self):
        import click
        @click.command()
        def mock_cmd():
            pass
        return mock_cmd


class TestPluginLoader:
    def test_discover_builtin_plugins(self):
        """Test discovery of built-in plugins."""
        loader = PluginLoader()
        plugins = list(loader.discover_all())
        
        assert len(plugins) > 0
        assert any(p.name == "vision" for p in plugins)
    
    def test_get_enabled_plugins(self):
        """Test filtering enabled plugins."""
        loader = PluginLoader()
        list(loader.discover_all())  # Load plugins
        
        enabled = loader.get_enabled_plugins()
        assert len(enabled) > 0
        assert all(p.enabled for p in enabled)
    
    def test_plugin_registration(self):
        """Test plugin can be registered."""
        import click
        
        @click.group()
        def test_cli():
            pass
        
        plugin = MockPlugin()
        test_cli.add_command(plugin.click_command)
        
        assert "mock" in test_cli.commands
```

#### Task 5.3: Update Integration Tests (2 hours)

- Update `tests/python/integration/test_cli_commands.py`
- Remove crop/remove_bg references
- Ensure all plugin-based commands still work

---

## Migration Strategy

### Backwards Compatibility

1. **Keep existing command files** during migration:
   - `src/ei_cli/cli/commands/vision.py` → stays
   - `src/ei_cli/plugins/vision.py` → wraps it
   
2. **Gradual migration**:
   - Phase 1: Add plugin system alongside existing
   - Phase 2: Convert commands one by one
   - Phase 3: Remove old registration (breaking change)

### Configuration

```yaml
# config.yaml - plugin control
plugins:
  enabled:
    - vision
    - image
    - speak
    - transcribe
    - search
  disabled:
    - crop          # Disabled by default
    - remove_bg     # Disabled by default
```

---

## Developer Experience

### Creating a New Plugin

```python
# my_plugin.py
from ei_cli.plugins.base import BaseCommandPlugin
import click

@click.command()
@click.argument("text")
def my_command(text: str):
    """My awesome command."""
    click.echo(f"Processing: {text}")

class MyPlugin(BaseCommandPlugin):
    @property
    def name(self) -> str:
        return "mycommand"
    
    @property
    def click_command(self):
        return my_command
    
    @property
    def category(self) -> str:
        return "Custom Tools"

plugin = MyPlugin()
```

### Third-Party Plugin Installation

```toml
# In third-party package's pyproject.toml
[project.entry-points."ei_cli.plugins"]
mycommand = "my_eai_plugin:plugin"
```

```bash
# User installs plugin
pip install eai-my-plugin

# Command automatically available
eai mycommand "hello"
```

---

## MCP Server Compatibility

The plugin architecture naturally supports MCP:

```python
# Future MCP server
class MCPServer:
    def __init__(self):
        self.loader = PluginLoader()
    
    def get_tools(self) -> list[MCPTool]:
        """Convert plugins to MCP tools."""
        tools = []
        for plugin in self.loader.get_enabled_plugins():
            tool = self._plugin_to_mcp_tool(plugin)
            tools.append(tool)
        return tools
```

---

## Benefits Summary

### For Users
- ✅ Cleaner command listing (no broken commands)
- ✅ Organized help by category
- ✅ Faster startup (lazy loading)
- ✅ Community plugins available

### For Developers
- ✅ Easy to add new commands
- ✅ No need to modify core code
- ✅ Clear plugin interface
- ✅ Good documentation

### For Maintainers
- ✅ Better separation of concerns
- ✅ Easier testing
- ✅ Simpler to deprecate commands
- ✅ Future-proof architecture

---

## Success Criteria

- [ ] All 10 working commands converted to plugins
- [ ] crop and remove_bg commands removed
- [ ] Plugin loader discovers all built-in plugins
- [ ] Help system shows categorized commands
- [ ] All 604 tests still passing (minus removed commands)
- [ ] Example third-party plugin works
- [ ] Documentation updated
- [ ] No breaking changes for end users

---

## Timeline

### Day 1 (6-8 hours)
- ✅ Create plugin base and loader
- ✅ Convert 3-4 commands to plugins
- ✅ Remove crop/remove_bg

### Day 2 (6-8 hours)
- ✅ Convert remaining commands
- ✅ Update main app.py
- ✅ Enhanced help system

### Day 3 (4-6 hours)
- ✅ Update all tests
- ✅ Documentation
- ✅ Example plugin
- ✅ Final verification

**Total: 16-22 hours (~2-3 days)**

---

## Next Steps

1. ✅ Branch created: `feature/plugin-architecture`
2. Review and approve this plan
3. Begin Phase 1: Core plugin infrastructure
4. Iterate through phases with testing
5. Merge when complete

---

## Questions for Review

1. **Entry points vs directory scanning?** 
   - Recommend: Both (directory for built-in, entry points for third-party)

2. **Plugin configuration location?**
   - Recommend: In existing `config.yaml` under `plugins` section

3. **Category names?**
   - Vision & Images
   - Audio & Speech
   - Document Processing
   - Search & Analysis
   - System Tools

4. **Keep backward compatibility?**
   - Recommend: Yes initially, then deprecate in v0.2.0
