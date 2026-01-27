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
```

## Architecture

Multi-file Python application with event-driven architecture:

### Core Files
- `wispy.py` - Main application, menu bar UI (rumps), hotkey handling, transcription orchestration
- `streaming.py` - Real-time streaming transcription with VAD-triggered segments
- `vad.py` - Voice Activity Detection using WebRTC VAD
- `llm_processor.py` - LLM post-processing for text cleanup (punctuation, filler word removal)

### Key Components
- **Menu bar app**: `rumps` for macOS menu bar integration (LSUIElement)
- **Audio capture**: `sounddevice` with callback-based streaming, 16kHz mono
- **Transcription engines**:
  - MLX-Whisper (99+ languages, multiple model sizes)
  - Parakeet MLX (v2: English only, v3: 25 languages, ~30x faster than Whisper)
- **LLM post-processing**: MLX-LM with lightweight Qwen 2.5 models for text cleanup
- **Keyboard control**: `pynput` for global hotkey listening and Cmd+V simulation
- **Text output**: `pyperclip` for clipboard operations with automatic preservation/restoration

### Recording Modes
1. **Hold mode**: Hold Right Option (default) to record, release to transcribe
2. **Toggle mode**: Left Option + Space to start/stop recording
3. **Streaming mode**: Real-time transcription with VAD-based segmentation (under Experimental menu)

### Flow
Key press → Start recording → (VAD segments in streaming mode) → Key release → Stop recording → Save temp WAV → Transcribe → (LLM cleanup if enabled) → Copy to clipboard → Simulate Cmd+V → Restore original clipboard → Cleanup

### Safety Features
- Single-instance lock prevents multiple app launches
- 5-minute max recording duration with auto-stop prevents memory exhaustion
- Queue size limits (50 segments max) in streaming mode
- Graceful shutdown with signal handlers and resource cleanup
- Thread-safe UI updates with locks

### Additional Features
- **Device auto-detection**: Automatically refreshes device list and switches when devices plug/unplug
- **Model download progress**: Real-time download status shown in menu bar
- **Clipboard preservation**: Original clipboard contents restored after pasting transcription
- **LLM text cleanup** (Experimental): Post-process transcriptions to fix punctuation, remove filler words (um, uh, like, you know, etc.), and correct obvious errors

## Configuration

Config stored at `~/.config/wispy/config.json`:
- `hotkeys.hold_key` - Key for hold-to-record (default: `alt_r`)
- `hotkeys.toggle_modifier` - Modifier for toggle mode (default: `alt_l`)
- `engine` - Transcription engine: `whisper` or `parakeet`
- `model` - HuggingFace model repo for selected engine
- `llm_enabled` - Enable LLM text cleanup (default: `false`)
- `llm_model` - HuggingFace model repo for LLM (default: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`)

### Supported Hotkey Values
- Modifier keys: `ctrl_l`, `ctrl_r`, `alt_l`, `alt_r`, `cmd_l`, `cmd_r`, `shift_l`, `shift_r`
- Function keys: `f1` through `f12`
- Special keys: `space`, `tab`, `caps_lock`
- Character keys: Dynamically assignable via interactive configuration

## macOS Permissions Required

- **Microphone**: System Settings > Privacy & Security > Microphone
- **Accessibility**: System Settings > Privacy & Security > Accessibility (for global hotkeys and paste simulation)
