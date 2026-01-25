#!/usr/bin/env python3
"""Wispy - Local voice-to-text tool using Whisper on Apple Silicon."""

import tempfile
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key
import mlx_whisper

# Configuration
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1
MODEL_REPO = "mlx-community/whisper-base-mlx"

# Global state
recording = False
audio_chunks = []
stream = None
kbd_controller = Controller()


def audio_callback(indata, frames, time, status):
    """Callback for audio stream - buffers incoming audio."""
    if status:
        print(f"Audio status: {status}")
    if recording:
        audio_chunks.append(indata.copy())


def start_recording():
    """Begin audio capture."""
    global recording, audio_chunks, stream

    if recording:
        return

    print("Recording...")
    audio_chunks = []
    recording = True

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        callback=audio_callback
    )
    stream.start()


def stop_recording():
    """End capture and return the audio data as a numpy array."""
    global recording, stream

    if not recording:
        return None

    recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_chunks:
        print("No audio recorded")
        return None

    # Concatenate all chunks
    audio_data = np.concatenate(audio_chunks, axis=0)
    return audio_data


def save_audio_to_temp(audio_data):
    """Save audio data to a temporary WAV file."""
    # Convert float32 to int16 for WAV file
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    wavfile.write(temp_path, SAMPLE_RATE, audio_int16)
    return temp_path


def transcribe(audio_path):
    """Transcribe audio file using mlx-whisper."""
    print("Transcribing...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=MODEL_REPO
    )
    return result["text"].strip()


def paste_text(text):
    """Copy text to clipboard and simulate Cmd+V to paste."""
    if not text:
        print("No text to paste")
        return

    pyperclip.copy(text)

    # Small delay to ensure clipboard is ready
    import time
    time.sleep(0.05)

    # Simulate Cmd+V
    kbd_controller.press(Key.cmd)
    kbd_controller.tap('v')
    kbd_controller.release(Key.cmd)

    print(f"Pasted: {text}")


def on_press(key):
    """Handle key press events."""
    try:
        if key == keyboard.Key.ctrl_r:  # Right Control key
            start_recording()
    except AttributeError:
        pass


def on_release(key):
    """Handle key release events."""
    try:
        if key == keyboard.Key.ctrl_r:  # Right Control key
            audio_data = stop_recording()

            if audio_data is not None and len(audio_data) > 0:
                # Save to temp file
                temp_path = save_audio_to_temp(audio_data)

                try:
                    # Transcribe
                    text = transcribe(temp_path)

                    # Paste the result
                    paste_text(text)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    except AttributeError:
        pass

    # Return True to keep the listener running
    return True


def main():
    """Main entry point."""
    print("Loading Whisper model...")

    # Pre-load the model by doing a dummy transcription
    # This ensures the model is cached for faster subsequent calls
    # mlx-whisper loads the model lazily on first transcribe call

    print(f"Model: {MODEL_REPO}")
    print("Ready! Hold Right Control key to record, release to transcribe and paste.")
    print("Press Ctrl+C to exit.")
    print()
    print("Note: You may need to grant Accessibility and Microphone permissions")
    print("in System Settings > Privacy & Security")
    print()

    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
