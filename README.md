# EverydayAI CLI````markdown

# EverydayAI CLI

**A powerful command-line toolkit for AI-powered multimedia processing**

**Personal AI toolkit for regular people**  

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)**Status:** 🟡 Alpha - Core tools working, more features planned

[![PyPI version](https://img.shields.io/pypi/v/everydayai-cli.svg)](https://pypi.org/project/everydayai-cli/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)Created by Keith Williams - Director of Enterprise AI @ NJIT



Transform images, audio, and video using AI - designed for content creators, educators, marketers, and anyone who needs powerful AI tools without writing code.[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[![Code Coverage](https://img.shields.io/badge/coverage-63.79%25-yellow.svg)](https://github.com/kaw393939/eai)

## Installation[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/kaw393939/eai)

```bash[![PyPI version](https://img.shields.io/pypi/v/everydayai-cli.svg)](https://pypi.org/project/everydayai-cli/)

pip install everydayai-cli

```## What is EverydayAI CLI?



Or with pipx (recommended for CLI tools):A command-line toolkit that makes AI-powered multimedia processing accessible to everyone - not just developers. Built for content creators, educators, podcasters, marketers, and anyone who wants to leverage AI without writing code.



```bash**v0.2.0 introduces a plugin architecture** making the CLI extensible and allowing third-party command additions.

pipx install everydayai-cli

```## Features



## Quick Start### ✅ Currently Available



```bash- 🖼️ **Image Generation**: Create images with DALL-E via gpt-image-1 model

# Analyze an image- 👁️ **Vision Analysis**: Single and multi-image analysis with GPT-5

eai vision photo.jpg --prompt "What's in this image?"- 🗣️ **Text-to-Speech**: OpenAI TTS with 6 voices and ElevenLabs integration

- �️ **Audio Transcription**: Whisper-powered transcription with preprocessing

# Generate an image- 🌐 **Web Search**: AI-powered search with citations and sources

eai image "a futuristic city at sunset" -o city.png- � **YouTube Processing**: Video download, transcription, and translation

- � **Plugin Architecture**: Extensible command system with dynamic discovery

# Transcribe audio- ⚙️ **Flexible Configuration**: YAML config + environment variables

eai transcribe podcast.mp3- 🎯 **Robust Error Handling**: Structured errors with helpful suggestions



# Download and transcribe YouTube video### 🚧 Removed in v0.2.0

eai transcribe_video "https://youtube.com/watch?v=..." -o transcript.txt

```- ❌ **Smart Cropping**: Removed (didn't work reliably)

- ❌ **Background Removal**: Removed (poor quality results)

---

### 🚧 Coming Soon (See [ROADMAP.md](ROADMAP.md))

## Tool Capabilities

- 🤖 **OpenAI Agents SDK**: Multi-agent workflows and orchestration

### 🖼️ Image Generation (`eai image`)- 📦 **Plugin Marketplace**: Community-contributed commands

- 🔄 **Workflow Chains**: Sequential tool orchestration

Create stunning images from text descriptions using OpenAI's DALL-E via the gpt-image-1 model.- 🎨 **Enhanced Image Tools**: Better quality smart crop and background removal

- � **Batch Processing**: Process multiple files efficiently

**What it does:**

- Generates high-quality images from natural language prompts## Installation

- Supports multiple sizes: 256x256, 512x512, 1024x1024, 1024x1792, 1792x1024

- Quality options: standard (fast, cost-effective) or HD (detailed, premium)```bash

- Style control: vivid (hyper-real, dramatic) or natural (realistic, subtle)# From PyPI - Live Now! 🎉

- Automatic prompt enhancement for better resultspip install everydayai-cli

- Analytics tracking (category, complexity scoring)

# From source

**Usage:**git clone https://github.com/kaw393939/eai.git

cd eai

```bashpoetry install

# Basic generation

eai image "a serene mountain landscape" -o landscape.png# Verify installation

eai --version

# High quality, specific size```

eai image "corporate logo design" -o logo.png --size 1024x1024 --quality hd

## Quick Start

# Landscape format with vivid style

eai image "sunset over ocean" -o sunset.png --size 1792x1024 --style vivid```bash

# Analyze an image with AI

# JSON output for automationeai vision photo.jpg --prompt "Describe this image in detail"

eai image "product mockup" --json

```# Analyze multiple images at once

eai multi_vision image1.jpg image2.jpg image3.jpg --compare

**Use cases:**

- Marketing visuals and social media content# Generate an image

- Concept art and design mockupseai image "A serene mountain landscape" -o mountain.png

- Educational illustrations

- Blog post headers and graphics# AI-powered web search with citations

- Product visualizationeai search "latest developments in AI 2024"



---# Generate professional speech from text

eai speak "Welcome to our presentation" -o welcome.mp3

### 👁️ Vision Analysis (`eai vision`)

# Transcribe audio to text

Analyze images with GPT-5 Vision to extract information, answer questions, or describe visual content.eai transcribe podcast.mp3



**What it does:**# Download and transcribe YouTube video

- Understands image content with human-level comprehensioneai transcribe_video "https://youtube.com/watch?v=..." -o transcript.txt

- Answers specific questions about images

- Extracts text (OCR) from photos and documents# Use ElevenLabs for premium voices

- Identifies objects, people, scenes, and activitieseai elevenlabs speak "Professional narration" -o narration.mp3 --voice adam

- Analyzes visual patterns, colors, and composition```

- Supports both local files and URLs

- Configurable detail levels (auto, low, high) for cost/quality balance## Commands



**Usage:**### Core Commands



```bash| Command | Description |

# General description|---------|-------------|

eai vision photo.jpg| `eai image` | Generate images with DALL-E (gpt-image-1) |

| `eai vision` | Analyze single images with GPT-5 |

# Ask specific questions| `eai multi_vision` | Analyze multiple images simultaneously |

eai vision receipt.jpg --prompt "What is the total amount?"| `eai speak` | Text-to-speech with OpenAI voices |

eai vision diagram.png --prompt "Explain this architecture diagram"| `eai transcribe` | Audio-to-text with Whisper |

| `eai search` | Web search with AI-powered answers |

# OCR text extraction| `eai youtube` | Manage YouTube authentication |

eai vision document.jpg --prompt "Extract all text from this image"| `eai transcribe_video` | Download and transcribe videos |

| `eai translate_audio` | Translate audio to English |

# High detail analysis for complex images| `eai elevenlabs` | Premium TTS with ElevenLabs |

eai vision technical_drawing.png --detail high --max-tokens 2000

### `eai image`

# Analyze from URL

eai vision "https://example.com/image.jpg" --prompt "Describe this"Generate images using DALL-E.



# JSON output```bash

eai vision photo.jpg --prompt "List all objects visible" --jsoneai image PROMPT [OPTIONS]

```

Options:

**Use cases:**  -o, --output PATH          Output file path

- Document digitization and OCR  -s, --size TEXT           Image size (256x256, 512x512, 1024x1024, 1024x1792, 1792x1024)

- Image accessibility (generating alt text)  -q, --quality TEXT        Quality: standard, hd

- Visual quality control and inspection  --style TEXT              Style: vivid, natural

- Educational content analysis  --json                    Output JSON format

- Technical diagram interpretation```

- Receipt and invoice processing

### `eai vision`

---

Analyze images using GPT-5 Vision.

### 🔍 Multi-Image Analysis (`eai multi_vision`)

```bash

Analyze 2-3 images simultaneously to compare, contrast, or understand relationships between multiple visuals.eai vision IMAGE [OPTIONS]



**What it does:**Options:

- Analyzes multiple images in one request (2-3 images supported)  -p, --prompt TEXT         Question or instruction about the image

- Compares images to find similarities and differences  -m, --model TEXT          Model to use (default: gpt-5)

- Tracks changes across image sequences  -d, --detail TEXT         Detail level: auto, low, high

- Understands contextual relationships between images  -t, --max-tokens INT      Maximum tokens in response

- Supports comparison mode for detailed side-by-side analysis  --json                    Output as JSON

- Same GPT-5 Vision capabilities as single image analysis```



**Usage:**### `eai multi_vision`



```bashAnalyze multiple images simultaneously.

# Compare two images

eai multi_vision before.jpg after.jpg --prompt "What changed?"```bash

eai multi_vision IMAGE1 IMAGE2 [IMAGE3] [OPTIONS]

# Analyze a sequence

eai multi_vision step1.jpg step2.jpg step3.jpg --prompt "Describe each step"Options:

  -p, --prompt TEXT         Analysis prompt for all images

# Detailed comparison mode  -c, --compare             Enable detailed comparison mode

eai multi_vision original.jpg edited.jpg --compare  -d, --detail TEXT         Detail level: auto, low, high

  --json                    Output as JSON

# Find common elements```

eai multi_vision img1.jpg img2.jpg --prompt "What do these images have in common?"

### `eai transcribe`

# JSON output for automation

eai multi_vision photo1.jpg photo2.jpg --jsonTranscribe audio files to text.

```

```bash

**Use cases:**eai transcribe AUDIO_FILE [OPTIONS]

- Before/after comparisons

- Quality assurance across product batchesOptions:

- Design iteration review  -f, --format TEXT         Output format: text, json, srt, vtt

- Progress tracking (construction, growth, changes)  -l, --language TEXT       Source language code (e.g., 'en', 'es')

- A/B testing visual elements  -o, --output FILE         Save to file

- Multi-angle product photography analysis  --no-preprocess          Skip audio preprocessing

  --parallel               Use parallel processing (3-5x faster)

---```



### 🗣️ Text-to-Speech (`eai speak`)### `eai transcribe_video`



Convert text to natural-sounding speech using OpenAI's TTS models with multiple voices and formats.Download and transcribe videos.



**What it does:**```bash

- Generates professional audio from text (up to 4096 characters)eai transcribe_video URL [OPTIONS]

- 6 high-quality voices: alloy, echo, fable, onyx, nova, shimmer

- Two quality tiers: tts-1 (fast, standard) and tts-1-hd (premium, natural)Options:

- Multiple output formats: MP3, Opus, AAC, FLAC, WAV, PCM  -f, --format TEXT         Output format: text, json, srt, vtt

- Adjustable playback speed (0.25x to 4.0x)  -l, --language TEXT       Source language hint

- Optional pronunciation and style guidance  -o, --output FILE         Save transcript to file

- Streaming mode for long-form content with progress tracking  --keep-audio             Keep downloaded audio file

- Built-in audio playback  --parallel               Use parallel processing

```

**Usage:**

### `eai search`

```bash

# Basic speech generationAI-powered web search with citations.

eai speak "Hello world" -o hello.mp3

```bash

# Choose specific voiceeai search QUERY [OPTIONS]

eai speak "Welcome to our presentation" -o welcome.mp3 --voice nova

Options:

# Premium HD quality  -o, --output FILE         Save results to file

eai speak "Professional narration" -o narration.mp3 --model tts-1-hd  --json                    Output as JSON

  -d, --domains TEXT        Limit to specific domains

# From text file with streaming  --city TEXT              User location (city)

eai speak --input script.txt -o audiobook.mp3 --stream  --country TEXT           User location (country)

```

# Custom speed (slower for learning)

eai speak "Technical instructions" -o instructions.mp3 --speed 0.75## Plugin Architecture



# Compact format for streaming**New in v0.2.0**: Commands are now implemented as plugins, making the CLI extensible.

eai speak "Podcast intro" -o intro.opus --format opus

### Using Third-Party Plugins

# Pronunciation guidance

eai speak "Dr. Nguyen works at CERN" -o speech.mp3 \Install plugins via pip with the `eai.plugins` entry point:

  --instructions "Pronounce 'Nguyen' as 'win', 'CERN' as 'sern'"

```bash

# Generate and play immediatelypip install eai-plugin-example

eai speak "Quick demo" -o demo.mp3 --playeai example-command  # Plugin commands auto-discovered

``````



**Voice characteristics:**### Creating Plugins

- **alloy**: Neutral, professional (default)

- **echo**: Clear, articulateCreate your own commands by implementing the `CommandPlugin` protocol:

- **fable**: Warm, friendly

- **onyx**: Deep, authoritative```python

- **nova**: Energetic, upbeatfrom ei_cli.plugins import CommandPlugin, BaseCommandPlugin

- **shimmer**: Soft, gentleimport click



**Format guide:**class MyPlugin(BaseCommandPlugin):

- **MP3**: Universal compatibility (~30KB per "hello world")    name = "my-command"

- **Opus**: Best for streaming (~7KB)    category = "custom"

- **AAC**: Optimized for Apple devices (~25KB)    help_text = "My custom command"

- **FLAC**: Lossless quality (~38KB)    

- **WAV**: Uncompressed editing (~93KB)    def get_command(self) -> click.Command:

- **PCM**: Raw audio data (~93KB)        @click.command(name=self.name, help=self.help_text)

        def my_command():

**Use cases:**            click.echo("Hello from my plugin!")

- Podcast and video narration        return my_command

- E-learning content

- Audiobook creationplugin = MyPlugin()

- Accessibility (text-to-speech for visually impaired)```

- IVR and voice assistant responses

- Multilingual content localizationRegister via entry points in your `pyproject.toml`:



---```toml

[project.entry-points."eai.plugins"]

### 🎙️ Audio Transcription (`eai transcribe`)my-plugin = "my_package.plugin:plugin"

```

Convert audio to text with OpenAI's Whisper model - incredibly accurate across languages and accents.

See plugin documentation for details on creating custom commands.

**What it does:**

- Transcribes audio in 90+ languages with high accuracy### `eai speak`

- Supports all common audio formats (MP3, WAV, M4A, FLAC, etc.)

- Automatic audio preprocessing (mono conversion, optimal sample rate)Generate professional speech from text using AI.

- Multiple output formats: plain text, JSON with timestamps, SRT/VTT subtitles

- Parallel processing option for large files (3-5x faster)```bash

- Language hint support for improved accuracyeai speak TEXT [OPTIONS]

- Optional custom prompts for terminology guidanceeai speak --input FILE [OPTIONS]



**Usage:**Options:

  --input, -i PATH         Read text from file

```bash  --output, -o PATH        Output audio file (required)

# Basic transcription  --voice, -v VOICE        Voice: alloy, echo, fable, onyx, nova, shimmer,

eai transcribe podcast.mp3                           ash, ballad, coral, sage, verse (tts-1),

                           marin, cedar (tts-1-hd) [default: alloy]

# Save to file  --model, -m MODEL        Model: tts-1, tts-1-hd [default: tts-1]

eai transcribe interview.mp3 -o transcript.txt  --speed, -s FLOAT        Playback speed 0.25-4.0 [default: 1.0]

  --format, -f FORMAT      Audio format: mp3, opus, aac, flac, wav, pcm

# Generate subtitles                           [default: mp3]

eai transcribe video_audio.mp3 --format srt -o subtitles.srt  --instructions TEXT      Pronunciation/style guidance (max 4096 chars)

  --stream                 Enable streaming mode with progress

# JSON output with timestamps  --play                   Play audio after generation

eai transcribe lecture.mp3 --format json -o lecture.json

Examples:

# Specify language for better accuracy  # Basic usage with default voice

eai transcribe spanish_audio.mp3 --language es  eai speak "Hello world" -o hello.mp3



# Fast parallel processing  # Premium voice with high quality

eai transcribe long_meeting.mp3 --parallel -o meeting.txt  eai speak "Professional recording" -o pro.mp3 -v marin -m tts-1-hd



# Custom terminology guidance  # Long-form with streaming

eai transcribe technical_talk.mp3 \  eai speak --input script.txt -o audiobook.mp3 --stream

  --prompt "Technical terms: Kubernetes, PostgreSQL, GraphQL"

  # Custom pronunciation guidance

# Skip preprocessing for pre-optimized audio  eai speak "Dr. Nguyen at CERN" -o speech.mp3 \

eai transcribe optimized.wav --no-preprocess    --instructions "Pronounce 'Nguyen' as 'win', 'CERN' as 'sern'"

```

  # Small file size for streaming

**Output formats:**  eai speak "Compact audio" -o compact.opus -f opus

- **text**: Clean, readable transcript

- **json**: Timestamps, confidence scores, metadata  # Generate and play immediately

- **srt**: Standard subtitle format with timecodes  eai speak "Listen now" -o demo.mp3 --play

- **vtt**: WebVTT subtitle format for web video```



**Use cases:****Voice Options:**

- Meeting and interview transcription

- Podcast show notes generation- **Standard** (all models): alloy, echo, fable, onyx, nova, shimmer

- Video subtitle creation- **tts-1 only**: ash, ballad, coral, sage, verse

- Lecture and presentation documentation- **tts-1-hd only**: marin (most natural), cedar (rich depth)

- Call center quality assurance

- Legal depositions and court recordings**Format Guide:**

- Accessibility compliance

- **mp3**: Default, widely compatible (~30KB)

---- **opus**: Streaming optimized (~7KB)

- **aac**: Apple devices (~25KB)

### 📺 YouTube Video Processing (`eai transcribe_video`)- **flac**: Lossless quality (~38KB)

- **wav**: Uncompressed editing (~93KB)

Download and transcribe videos from YouTube and other platforms using yt-dlp.- **pcm**: Raw audio data (~93KB)



**What it does:**See [docs/TTS_GUIDE.md](docs/TTS_GUIDE.md) for comprehensive TTS documentation.

- Downloads audio from YouTube, Vimeo, and 1000+ video platforms

- Automatic video information extraction (title, duration)## Configuration

- Converts to optimal format for transcription

- Full Whisper transcription with all features### Configuration File

- Optional audio file preservation

- Cookie support for restricted videosCreate `.ei/config.yaml` in your project or `~/.ei/config.yaml` for global

- Browser cookie extraction for age-restricted contentsettings:



**Usage:**```yaml

ai:

```bash  api_key: ${EI_API_KEY} # Or set directly (not recommended)

# Basic video transcription  model: gpt-4-vision-preview

eai transcribe_video "https://youtube.com/watch?v=abc123"  max_tokens: 2000



# Save transcriptoutput:

eai transcribe_video "https://youtube.com/watch?v=abc123" -o transcript.txt  format: json # or "human"



# Generate subtitles from videologging:

eai transcribe_video "VIDEO_URL" --format srt -o subtitles.srt  level: INFO

  format: json # or "text"

# Keep audio file for later use```

eai transcribe_video "VIDEO_URL" --keep-audio --audio-format mp3

### Environment Variables

# Fast parallel processing

eai transcribe_video "VIDEO_URL" --parallel -o transcript.txt```bash

export EI_API_KEY="your-openai-api-key"

# Language hintexport EI_LOG_LEVEL="INFO"

eai transcribe_video "FRENCH_VIDEO_URL" --language frexport EI_OUTPUT_FORMAT="json"

```

# Use browser cookies for restricted videos

eai transcribe_video "RESTRICTED_VIDEO_URL" --cookies-from-browser chrome### Configuration Hierarchy

```

Configuration sources (later overrides earlier):

**Supported platforms:**

- YouTube (including age-restricted and premium content)1. Built-in defaults

- Vimeo2. Global config (`~/.ei/config.yaml`)

- Dailymotion3. Project config (`./.ei/config.yaml`)

- Facebook4. Environment variables (`EI_*`)

- Twitter/X5. Command-line arguments (`--option`)

- Instagram

- TikTok## Templates

- And 1000+ more via yt-dlp

Available templates:

**Use cases:**

- Video subtitle generation- **email-writing**: Professional email composition

- Lecture note creation- **lesson-plans**: Educational lesson planning

- YouTube content repurposing- **simple-website**: Static website creation

- Video accessibility compliance- **project-planning**: Project structure and planning

- Content research and analysis- **data-analysis**: Simple data analysis tasks

- Educational content transcription

Create custom templates in `~/.vibe/templates/`.

---

## Development

### 🌐 Audio Translation (`eai translate_audio`)

### Setup

Translate audio from any language to English using Whisper's translation capability.

```bash

**What it does:**# Clone repository

- Translates audio from any supported language to Englishgit clone https://github.com/kaw393939/eai.git

- Same high accuracy as transcriptioncd eai

- All Whisper preprocessing and optimization features

- Multiple output formats (text, JSON, SRT, VTT)# Install dependencies

- Note: Only translates TO English (Whisper limitation)poetry install



**Usage:**# Run in development mode

poetry run eai --help

```bash```

# Translate to English

eai translate_audio spanish_audio.mp3### Testing



# Save translation```bash

eai translate_audio french_podcast.mp3 -o english_translation.txt# Run all tests

poetry run pytest

# Generate English subtitles from foreign video

eai translate_audio foreign_audio.mp3 --format srt -o english_subs.srt# Run with coverage

poetry run pytest --cov=src/ei_cli --cov-report=html

# JSON output with timestamps

eai translate_audio interview.mp3 --format json# Run specific test category

poetry run pytest -m unit          # Unit tests only

# Style guidancepoetry run pytest -m integration   # Integration tests only

eai translate_audio presentation.mp3 \```

  --prompt "Formal business translation"

```### Quality Checks



**Important:** This only translates TO English. For text translation between other languages, use ChatGPT API separately.```bash

# Linting

**Use cases:**poetry run ruff check src/ tests/

- International content localization

- Foreign language interview translation# Type checking

- Global video subtitle creationpoetry run mypy src/

- Multilingual customer support analysis

- Research interview translation# Security scanning

poetry run bandit -r src/

---

# Run all quality checks

### 🔎 Web Search (`eai search`)poetry run pre-commit run --all-files

```

AI-powered web search with comprehensive answers, citations, and source attribution.

## Architecture

**What it does:**

- Performs intelligent web searches with Google Custom SearchThe CLI follows a clean layered architecture:

- AI-generated comprehensive answers (not just links)

- Includes citations and sources for verification- **CLI Layer** (`cli/`): Command parsing and user interaction

- Domain filtering for targeted searches- **Tools Layer** (`tools/`): Core AI and image processing tools

- Location-aware results (city/country context)- **Core Layer** (`core/`): Configuration, errors, shared utilities

- Structured output with rich formatting

- JSON export for automationKey principles:



**Usage:**- **EAFP over LBYL**: "Easier to Ask Forgiveness than Permission"

- **Structured Errors**: All errors provide machine-readable context

```bash- **Configuration Flexibility**: Multiple config sources, sensible defaults

# Basic search- **Type Safety**: Full mypy strict mode compliance

eai search "latest developments in AI 2024"

See [TECHNICAL_DEBT_AUDIT.md](TECHNICAL_DEBT_AUDIT.md) for current architecture

# Save resultsstatus and [ROADMAP.md](ROADMAP.md) for planned improvements.

eai search "Python async programming guide" -o results.txt

## Testing Strategy

# Domain-specific search

eai search "machine learning tutorials" --domains "edu,github.io"Current test coverage: **63.79%** (Target: 90%)



# Location-aware search**Test Results (v0.2.0):**

eai search "best Italian restaurants" --city "New York" --country "US"- ✅ 559 tests passing

- ⏭️ 41 tests skipped (image streaming not yet implemented)

# JSON output- ✅ Configuration system: 100% coverage

eai search "climate change statistics" --json -o data.json- ✅ Error handling: High coverage

```- ✅ Plugin system: Validated via integration tests

- ✅ All commands: Manually tested and working

**Output includes:**

- Comprehensive AI-generated answer### Running Tests

- Source citations with titles and URLs

- Publication dates and metadata```bash

- Structured formatting for readability# All tests

poetry run pytest

**Use cases:**

- Research and fact-checking# With coverage report

- Competitive analysispoetry run pytest --cov=src/ei_cli --cov-report=html

- Content research for writing

- Technical documentation discovery# Specific categories

- Academic researchpoetry run pytest tests/python/unit/        # Unit tests

- Market researchpoetry run pytest tests/python/integration/ # Integration tests

```

---

We're actively working toward 90% coverage. See test documentation for details.

### 🎵 Premium Text-to-Speech (`eai elevenlabs`)

## Contributing

High-quality voice synthesis with ElevenLabs for premium, natural-sounding speech.

We welcome contributions! Here's how to get started:

**What it does:**

- Premium voices with exceptional naturalness and emotion1. Fork the repository

- 40+ pre-made voices across different styles and accents2. Create a feature branch (`git checkout -b feature/amazing-feature`)

- Advanced voice control (stability, similarity boost)3. Write tests first (TDD approach)

- Multiple output formats and quality settings4. Implement your feature

- Streaming support for long content5. Ensure all quality gates pass (`poetry run pre-commit run --all-files`)

- Voice cloning capabilities (with subscription)6. Commit changes (`git commit -m 'Add amazing feature'`)

7. Push to branch (`git push origin feature/amazing-feature`)

**Usage:**8. Open a Pull Request



```bashPlease read [TECHNICAL_DEBT_AUDIT.md](TECHNICAL_DEBT_AUDIT.md) to understand

# List available voicescurrent priorities.

eai elevenlabs list-voices

## License

# Generate speech with specific voice

eai elevenlabs speak "Welcome to our podcast" -o intro.mp3 --voice adamMIT License - See [LICENSE](LICENSE) for details.



# Premium quality settings## Author

eai elevenlabs speak "Professional narration" -o narration.mp3 \

  --voice marin --stability 0.75 --similarity 0.8**Keith Williams**



# Different output format- Director of Enterprise AI @ NJIT

eai elevenlabs speak "Audio message" -o message.mp3 --format mp3_44100_192- 23 years teaching computer science

```- Building EverydayAI Newark

- [keithwilliams.io](https://keithwilliams.io)

**When to use ElevenLabs vs OpenAI:**- [@kaw393939](https://github.com/kaw393939)

- **ElevenLabs**: Most natural-sounding, best for professional content, podcasts, audiobooks

- **OpenAI**: Fast, reliable, great quality, more cost-effective for bulk usage## Acknowledgments



**Use cases:**- Part of **EverydayAI Newark** - training everyone for distributed productivity

- Professional podcast production  gains

- Audiobook narration- Built to make AI accessible to non-developers

- High-quality commercial voice-overs- Inspired by Swiss design principles - clarity, function, minimal complexity

- Character voices for games/animations

- Premium e-learning content## Links



---- **Website**: [keithwilliams.io](https://keithwilliams.io)

- **GitHub**: [github.com/kaw393939/eai](https://github.com/kaw393939/eai)

### 🔐 YouTube Authentication (`eai youtube`)- **PyPI**: [pypi.org/project/everydayai-cli](https://pypi.org/project/everydayai-cli)

- **Documentation**:

Manage authentication for downloading restricted YouTube content.  - [TECHNICAL_DEBT_AUDIT.md](TECHNICAL_DEBT_AUDIT.md) - Current status

  - [ROADMAP.md](ROADMAP.md) - Planned features

**What it does:**- **Issues**:

- Stores browser cookies for YouTube authentication  [github.com/kaw393939/eai/issues](https://github.com/kaw393939/eai/issues)

- Enables download of age-restricted content

- Handles member-only videos---

- Persists login sessions

- Cookie freshness monitoring**Status:** 🟡 Alpha - Core features working, plugin system stable  

**Version:** 0.2.0  

**Usage:****Coverage:** 63.79% → Target: 90%

````

```bash
# Check authentication status
eai youtube check

# Setup YouTube authentication
eai youtube setup

# Clear saved cookies
eai youtube clear
```

**Use cases:**
- Downloading age-restricted educational content
- Accessing member-only videos
- Enterprise video archival
- Research and documentation

---

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Optional: ElevenLabs API key for premium TTS:

```bash
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

Configuration file (`~/.ei_cli/config.yaml` or project-level `config.yaml`):

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  
elevenlabs:
  api_key: ${ELEVENLABS_API_KEY}
```

---

## Plugin Architecture

**New in v0.2.0**: Extensible plugin system allows adding custom commands.

### Creating Plugins

```python
from ei_cli.plugins import BaseCommandPlugin
import click

class MyPlugin(BaseCommandPlugin):
    name = "my-command"
    category = "custom"
    help_text = "My custom command"
    
    def get_command(self) -> click.Command:
        @click.command(name=self.name, help=self.help_text)
        def my_command():
            click.echo("Hello from my plugin!")
        return my_command

plugin = MyPlugin()
```

Register in `pyproject.toml`:

```toml
[project.entry-points."eai.plugins"]
my-plugin = "my_package.plugin:plugin"
```

---

## Development

```bash
# Clone repository
git clone https://github.com/kaw393939/eai.git
cd eai

# Install with Poetry
poetry install

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/ei_cli --cov-report=html
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/kaw393939/eai
- **PyPI**: https://pypi.org/project/everydayai-cli/
- **Issues**: https://github.com/kaw393939/eai/issues

## Author

**Keith Williams** - Director of Enterprise AI @ NJIT  
Building EverydayAI Newark to make AI accessible to everyone.

---

**Version:** 0.2.0  
**Status:** Alpha - Core features stable and production-ready
