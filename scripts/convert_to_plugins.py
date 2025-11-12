"""Script to convert command files to plugins."""

import re
from pathlib import Path

# Plugin metadata for each command
PLUGIN_METADATA = {
    "vision": ("AI", "Analyze images using GPT-4/5 Vision"),
    "image": ("AI", "Generate images using gpt-image-1"),
    "speak": ("Audio", "Generate speech from text using OpenAI TTS"),
    "speak_elevenlabs": ("Audio", "Generate speech using ElevenLabs TTS"),
    "transcribe": ("Audio", "Transcribe audio to text using Whisper"),
    "transcribe_video": ("Audio", "Extract and transcribe audio from videos"),
    "translate_audio": ("Audio", "Translate audio between languages"),
    "search": ("Web", "Search the web using Google Custom Search"),
    "multi_vision": ("AI", "Analyze multiple images in a session"),
    "setup_youtube": ("Setup", "Configure YouTube integration"),
}

def convert_to_plugin(file_path: Path) -> None:
    """Convert a command file to a plugin."""
    name = file_path.stem
    
    if name in ["__init__", "base", "loader"]:
        return
    
    # Get metadata
    category, help_text = PLUGIN_METADATA.get(name, ("Tools", f"{name.title()} command"))
    
    # Read current content
    content = file_path.read_text()
    
    # Find the command function name (e.g., @click.command() def vision(...))
    # Look for @click.command pattern followed by def
    pattern = r'@click\.command\([^)]*\)\s*\ndef\s+(\w+)\s*\('
    match = re.search(pattern, content)
    
    if not match:
        print(f"Warning: Could not find command function in {name}.py")
        return
    
    command_func_name = match.group(1)
    
    # Check if plugin class already exists
    if f"class {name.title().replace('_', '')}Plugin" in content:
        print(f"Plugin class already exists in {name}.py, skipping")
        return
    
    # Add imports if not present
    imports_to_add = []
    if "from ei_cli.plugins.base import BaseCommandPlugin" not in content:
        imports_to_add.append("from ei_cli.plugins.base import BaseCommandPlugin")
    
    # Find the last import line
    import_lines = []
    for i, line in enumerate(content.split('\n')):
        if line.startswith('import ') or line.startswith('from '):
            import_lines.append(i)
    
    if import_lines and imports_to_add:
        lines = content.split('\n')
        last_import = max(import_lines)
        # Insert after last import
        for imp in reversed(imports_to_add):
            lines.insert(last_import + 1, imp)
        content = '\n'.join(lines)
    
    # Generate plugin class
    class_name = ''.join(word.title() for word in name.split('_')) + 'Plugin'
    
    plugin_code = f'''

class {class_name}(BaseCommandPlugin):
    """Plugin for {help_text.lower()}."""

    def __init__(self) -> None:
        """Initialize the {name} plugin."""
        super().__init__(
            name="{name}",
            category="{category}",
            help_text="{help_text}",
        )

    def get_command(self) -> click.Command:
        """Get the {name} command."""
        return {command_func_name}


# Plugin instance for auto-discovery
plugin = {class_name}()
'''
    
    # Append plugin code
    new_content = content + plugin_code
    
    # Write back
    file_path.write_text(new_content)
    print(f"✓ Converted {name}.py to plugin")

def main() -> None:
    """Convert all command files to plugins."""
    plugins_dir = Path("src/ei_cli/plugins")
    
    for file_path in plugins_dir.glob("*.py"):
        if file_path.stem not in ["__init__", "base", "loader"]:
            try:
                convert_to_plugin(file_path)
            except Exception as e:
                print(f"Error converting {file_path.name}: {e}")

if __name__ == "__main__":
    main()
