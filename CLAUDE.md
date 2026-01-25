# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wispy is a macOS voice-to-text application optimized for Apple Silicon. It captures audio via a keyboard hotkey (Right Control), transcribes speech locally using MLX-Whisper, and pastes the result into the active application.

## Commands

```bash
# Install dependencies (use virtual environment)
pip install -r requirements.txt

# Run the application
python3 wispy.py
```

## Architecture

Single-file Python application (`wispy.py`) with event-driven architecture:

- **Audio capture**: `sounddevice` with callback-based streaming, buffered to `audio_chunks` list
- **Transcription**: `mlx-whisper` (Apple Silicon optimized) with lazy model loading from `mlx-community/whisper-base-mlx`
- **Keyboard control**: `pynput` for global hotkey listening (Right Control) and Cmd+V simulation
- **Text output**: `pyperclip` for clipboard, keyboard simulation for paste

**Flow**: Key press → Start recording → Key release → Stop recording → Save temp WAV → Transcribe → Copy to clipboard → Simulate Cmd+V → Cleanup

## macOS Permissions Required

- Microphone access (System Settings > Privacy & Security)
- Accessibility permission (System Settings > Privacy & Security) for global keyboard events
