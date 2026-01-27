# Wispy

A lightweight, local voice-to-text application for macOS, optimized for Apple Silicon.

Wispy runs entirely on your Mac — no cloud services, no API keys, no data leaves your device. Just press a hotkey, speak, and your transcribed text is automatically pasted into any application. Can be used with any available local model.

## Features

- **100% Local** — All transcription happens on-device using Apple Silicon's Neural Engine
- **Two Transcription Engines**:
  - **MLX-Whisper** — OpenAI's Whisper optimized for Apple Silicon, supports 99+ languages
  - **Parakeet** — NVIDIA's Parakeet model via MLX, ~30x faster (v2: English-only, v3: 25 languages)
- **Menu Bar App** — Lives in your menu bar, always ready
- **Configurable Hotkeys** — Customize your recording shortcuts
- **Multiple Recording Modes**:
  - Hold-to-record (default: Right Option)
  - Toggle mode (default: Left Option + Space)
  - Streaming mode with real-time transcription
- **Multiple Model Sizes** — From Tiny (~75MB) to Large (~3GB), choose your speed/accuracy tradeoff
- **Clipboard Preservation** — Your clipboard contents are automatically restored after pasting

## Requirements

- macOS 12.0 or later
- Apple Silicon Mac (M-series)
- Python 3.10+ (for running from source)
- ffmpeg

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/aceyadapter/wispy.git
cd wispy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 wispy.py
```

## Usage

1. **Launch Wispy** — A microphone icon appears in your menu bar
2. **Grant Permissions** — Allow Microphone and Accessibility access when prompted
3. **Record** — Hold Right Option and speak (or use Left Option + Space to toggle)
4. **Done** — Release the key and your transcription is automatically pasted

### Menu Bar Options

- **Microphone** — Select your input device (auto-detects when devices plug/unplug)
- **Engine** — Switch between Whisper and Parakeet
- **Model** — Choose model size (downloads automatically on first use)
- **Hotkeys** — Customize your recording shortcuts
- **Experimental** — Access experimental features like Streaming Mode

## Models

### Whisper Models (99+ languages)
| Model | Size | Use Case |
|-------|------|----------|
| Tiny | ~75 MB | Quick dictation, lower accuracy |
| Base | ~140 MB | Good balance (default) |
| Small | ~460 MB | Better accuracy |
| Medium | ~1.5 GB | High accuracy |
| Large v3 | ~3 GB | Best accuracy |

### Parakeet
| Model | Size | Use Case |
|-------|------|----------|
| Parakeet 0.6B v2 | ~2.5 GB | Very fast, English-only |
| Parakeet 0.6B v3 | ~2.5 GB | Very fast, 25 languages |

Models are downloaded from Hugging Face on first use and cached locally.

## Permissions

Wispy requires two macOS permissions:

1. **Microphone** — To capture your voice
   - System Settings > Privacy & Security > Microphone > Enable for Wispy/Terminal

2. **Accessibility** — To detect global hotkeys and paste text
   - System Settings > Privacy & Security > Accessibility > Enable for Wispy/Terminal

## Configuration

Settings are stored at `~/.config/wispy/config.json` and can be changed via the menu bar.

## Troubleshooting

**"Wispy is already running"**
- Check your menu bar for the existing instance, or use Activity Monitor to quit it

**Hotkeys not working**
- Ensure Accessibility permission is granted in System Settings

**No audio captured**
- Check Microphone permission and verify the correct input device is selected

**Transcription is slow**
- Try a smaller model (Tiny or Base) or switch to Parakeet for English

**ffmpeg not found / audio processing error**
- MLX-Whisper requires ffmpeg for audio processing. Install it via Homebrew:
  ```bash
  brew install ffmpeg
  ```
- After installing, run the application again with `python3 wispy.py`

## Privacy

Wispy processes all audio locally on your Mac. No audio or transcriptions are sent to any external servers. The only network requests are to Hugging Face to download models on first use.

## License

MIT License — see LICENSE file for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple
- [MLX-Whisper](https://github.com/ml-explore/mlx-examples) by Apple
- [Parakeet](https://github.com/NVIDIA/NeMo) by NVIDIA
- [rumps](https://github.com/jaredks/rumps) for macOS menu bar integration
