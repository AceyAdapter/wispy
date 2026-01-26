# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wispy is a macOS menu bar voice-to-text application optimized for Apple Silicon. It captures audio via configurable keyboard hotkeys, transcribes speech locally using MLX-Whisper or Parakeet, and pastes the result into the active application.

## Commands

```bash
# Install dependencies (use virtual environment)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the application
python3 wispy.py

# Build standalone app
pip install pyinstaller
pyinstaller wispy.spec
# Output: dist/Wispy.app
```

## Architecture

Multi-file Python application with event-driven architecture:

### Core Files
- `wispy.py` - Main application, menu bar UI (rumps), hotkey handling, transcription orchestration
- `streaming.py` - Real-time streaming transcription with VAD-triggered segments
- `vad.py` - Voice Activity Detection using WebRTC VAD
- `runtime_hook.py` - PyInstaller hook for multiprocessing support

### Key Components
- **Menu bar app**: `rumps` for macOS menu bar integration (LSUIElement)
- **Audio capture**: `sounddevice` with callback-based streaming, 16kHz mono
- **Transcription engines**:
  - MLX-Whisper (99+ languages, multiple model sizes)
  - Parakeet MLX (English only, ~30x faster)
- **Keyboard control**: `pynput` for global hotkey listening and Cmd+V simulation
- **Text output**: `pyperclip` for clipboard operations

### Recording Modes
1. **Hold mode**: Hold Right Option (default) to record, release to transcribe
2. **Toggle mode**: Left Option + Space to start/stop recording
3. **Streaming mode**: Real-time transcription with VAD-based segmentation

### Flow
Key press → Start recording → (VAD segments in streaming mode) → Key release → Stop recording → Save temp WAV → Transcribe → Copy to clipboard → Simulate Cmd+V → Cleanup

### Safety Features
- Single-instance lock prevents multiple app launches
- 5-minute max recording duration prevents memory exhaustion
- Graceful shutdown with signal handlers and resource cleanup
- Multiprocessing freeze_support() for PyInstaller compatibility

## Configuration

Config stored at `~/.config/wispy/config.json`:
- `hotkeys.hold_key` - Key for hold-to-record (default: `alt_r`)
- `hotkeys.toggle_modifier` - Modifier for toggle mode (default: `alt_l`)
- `engine` - Transcription engine: `whisper` or `parakeet`
- `model` - HuggingFace model repo for selected engine

## macOS Permissions Required

- **Microphone**: System Settings > Privacy & Security > Microphone
- **Accessibility**: System Settings > Privacy & Security > Accessibility (for global hotkeys and paste simulation)
