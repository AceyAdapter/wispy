#!/usr/bin/env python3
"""Wispy - Local voice-to-text tool using Whisper on Apple Silicon."""

import tempfile
import os
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key
import mlx_whisper
import rumps
from huggingface_hub import scan_cache_dir

from streaming import StreamingTranscriber

# Track loaded model to avoid redundant loads
_loaded_model_repo = None


def preload_model(model_repo: str, on_status=None):
    """Pre-load a model by running a dummy transcription."""
    global _loaded_model_repo

    if _loaded_model_repo == model_repo:
        print(f"Model already loaded: {model_repo}")
        return

    if on_status:
        on_status(f"Loading model...", "⏳")

    print(f"Pre-loading model: {model_repo}")

    # Create a short silent audio file to trigger model load
    silent_audio = np.zeros(SAMPLE_RATE, dtype=np.int16)  # 1 second of silence
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        wavfile.write(temp_path, SAMPLE_RATE, silent_audio)
        mlx_whisper.transcribe(temp_path, path_or_hf_repo=model_repo)
        _loaded_model_repo = model_repo
        print(f"Model loaded: {model_repo}")
        if on_status:
            on_status("Ready - Hold Right Ctrl to record", "🎤")
    except Exception as e:
        print(f"Error loading model: {e}")
        if on_status:
            on_status(f"Error loading model", "❌")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Configuration
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1

# Available MLX Whisper models (repo, display name, approximate size)
MODELS = [
    ("mlx-community/whisper-tiny-mlx", "Tiny", "~75 MB"),
    ("mlx-community/whisper-base-mlx", "Base", "~140 MB"),
    ("mlx-community/whisper-small-mlx", "Small", "~460 MB"),
    ("mlx-community/whisper-medium-mlx", "Medium", "~1.5 GB"),
    ("mlx-community/whisper-large-v3-mlx", "Large v3", "~3 GB"),
]
DEFAULT_MODEL = "mlx-community/whisper-base-mlx"


class WispyApp(rumps.App):
    """Menu bar application for Wispy voice-to-text."""

    def __init__(self):
        super().__init__("Wispy", title="🎤")

        # State
        self.recording = False
        self.processing = False  # Prevent overlapping transcriptions
        self.audio_chunks = []
        self.stream = None
        self.kbd_controller = Controller()
        self.selected_device = None
        self.keyboard_listener = None
        self.ui_queue = Queue()
        self.lock = threading.Lock()
        self.model_repo = DEFAULT_MODEL

        # Streaming mode state
        self.streaming_mode = False
        self.streaming_transcriber = None

        # Get available devices
        self.input_devices = self._get_input_devices()

        # Set default device
        default_input = sd.default.device[0]
        if default_input in self.input_devices:
            self.selected_device = default_input
        elif self.input_devices:
            self.selected_device = self.input_devices[0]

        # Build menu
        self.status_item = rumps.MenuItem("Ready - Hold Right Ctrl to record")
        self.status_item.set_callback(None)

        # Device submenu
        self.device_menu = rumps.MenuItem("Microphone")
        self._build_device_menu()

        # Model submenu
        self.model_menu = rumps.MenuItem("Model")
        self._build_model_menu()

        # Streaming mode toggle
        self.streaming_toggle = rumps.MenuItem(
            "Streaming Mode",
            callback=self._toggle_streaming_mode
        )
        self.streaming_toggle.state = self.streaming_mode

        self.menu = [
            self.status_item,
            None,  # Separator
            self.device_menu,
            self.model_menu,
            self.streaming_toggle,
            None,  # Separator
        ]

    def _get_input_devices(self):
        """Get list of input device indices."""
        devices = sd.query_devices()
        return [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    def _build_device_menu(self):
        """Build the microphone selection submenu."""
        devices = sd.query_devices()
        default_input = sd.default.device[0]

        for i in self.input_devices:
            device = devices[i]
            name = device['name']
            if i == default_input:
                name += " (default)"

            item = rumps.MenuItem(name, callback=self._select_device)
            item.device_id = i
            item.state = (i == self.selected_device)
            self.device_menu[name] = item

    def _get_downloaded_models(self):
        """Get set of downloaded model repo names."""
        try:
            cache_info = scan_cache_dir()
            return {repo.repo_id for repo in cache_info.repos}
        except Exception:
            return set()

    def _build_model_menu(self):
        """Build the model selection submenu."""
        downloaded = self._get_downloaded_models()

        for repo, name, size in MODELS:
            is_downloaded = repo in downloaded
            status = "✓" if is_downloaded else f"↓ {size}"
            label = f"{name} ({status})"

            item = rumps.MenuItem(label, callback=self._select_model)
            item.model_repo = repo
            item.model_name = name
            item.state = (repo == self.model_repo)
            self.model_menu[label] = item

    def _select_model(self, sender):
        """Handle model selection from menu."""
        self.model_repo = sender.model_repo

        # Update checkmarks
        for item in self.model_menu.values():
            if hasattr(item, 'model_repo'):
                item.state = (item.model_repo == self.model_repo)

        print(f"Model changed: {sender.model_name} ({sender.model_repo})")

        # Load model in background thread
        def load():
            preload_model(self.model_repo, on_status=self.set_status)
            rumps.notification("Wispy", "Model ready", sender.model_name)

        threading.Thread(target=load, daemon=True).start()

    def _toggle_streaming_mode(self, sender):
        """Toggle streaming transcription mode."""
        self.streaming_mode = not self.streaming_mode
        sender.state = self.streaming_mode
        mode = "Streaming" if self.streaming_mode else "Standard"
        rumps.notification("Wispy", "Mode changed", f"{mode} mode enabled")
        print(f"Mode changed: {mode}")

    def _select_device(self, sender):
        """Handle device selection from menu."""
        self.selected_device = sender.device_id
        # Update checkmarks
        for item in self.device_menu.values():
            if hasattr(item, 'device_id'):
                item.state = (item.device_id == self.selected_device)

        device_name = sd.query_devices(self.selected_device)['name']
        rumps.notification("Wispy", "Microphone changed", device_name)

    def set_status(self, status, icon=None):
        """Update status display (thread-safe via queue)."""
        print(status)
        self.ui_queue.put((status, icon))

    def _paste_segment(self, text):
        """Paste a transcribed segment immediately (called from background thread)."""
        if not text:
            return
        pyperclip.copy(text + " ")  # Add space between segments
        time.sleep(0.05)
        self.kbd_controller.press(Key.cmd)
        self.kbd_controller.tap('v')
        self.kbd_controller.release(Key.cmd)

    def _process_ui_queue(self, _):
        """Process pending UI updates on main thread."""
        while not self.ui_queue.empty():
            status, icon = self.ui_queue.get_nowait()
            self.status_item.title = status
            if icon:
                self.title = icon

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - buffers incoming audio."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.audio_chunks.append(indata.copy())
            # Feed to streaming transcriber if in streaming mode
            if self.streaming_mode and self.streaming_transcriber:
                self.streaming_transcriber.process_audio(indata)

    def start_recording(self):
        """Begin audio capture."""
        with self.lock:
            if self.recording or self.processing or self.selected_device is None:
                return

            self.set_status("Recording...", "🔴")
            self.audio_chunks = []
            self.recording = True

            # Initialize streaming transcriber if in streaming mode
            if self.streaming_mode:
                self.streaming_transcriber = StreamingTranscriber(
                    model_repo=self.model_repo,
                    on_status=lambda s: self.set_status(s, "🔴"),
                    on_segment=self._paste_segment
                )
                self.streaming_transcriber.start()

            # Use smaller blocksize for streaming mode (optimizes for VAD chunks)
            blocksize = 512 if self.streaming_mode else None

            self.stream = sd.InputStream(
                device=self.selected_device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                blocksize=blocksize,
                callback=self.audio_callback
            )
            self.stream.start()

    def stop_recording(self):
        """End capture and return the audio data as a numpy array."""
        with self.lock:
            if not self.recording:
                return None

            self.recording = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if not self.audio_chunks:
                self.set_status("No audio recorded", "🎤")
                return None

            audio_data = np.concatenate(self.audio_chunks, axis=0)
            return audio_data

    def process_recording(self):
        """Stop recording, transcribe, and paste."""
        with self.lock:
            if self.processing:
                return
            self.processing = True

        # Capture streaming state before stopping
        is_streaming = self.streaming_mode and self.streaming_transcriber is not None
        transcriber = self.streaming_transcriber

        audio_data = self.stop_recording()

        if audio_data is None or len(audio_data) == 0:
            if transcriber:
                transcriber.stop()
            self.streaming_transcriber = None
            self.processing = False
            self.set_status("Ready - Hold Right Ctrl to record", "🎤")
            return

        try:
            if is_streaming:
                # Streaming mode: segments are pasted as they're transcribed
                # Just finalize any remaining audio
                self.set_status("Finalizing...", "⏳")
                transcriber.stop()  # This will paste remaining via on_segment callback
                self.streaming_transcriber = None
                # Don't paste again - segments were pasted in real-time
                self.set_status("Done", "✅")
                self.processing = False
                threading.Timer(2.0, lambda: self.set_status("Ready - Hold Right Ctrl to record", "🎤")).start()
                return
            else:
                # Standard mode: batch transcribe
                self.set_status("Transcribing...", "⏳")

                # Save to temp file
                audio_int16 = (audio_data * 32767).astype(np.int16)
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                wavfile.write(temp_path, SAMPLE_RATE, audio_int16)

                try:
                    result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=self.model_repo)
                    text = result["text"].strip()
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            if text:
                # Copy and paste
                pyperclip.copy(text)
                time.sleep(0.05)
                self.kbd_controller.press(Key.cmd)
                self.kbd_controller.tap('v')
                self.kbd_controller.release(Key.cmd)
                print(f"Pasted: {text}")
                self.set_status(f"Pasted: {text[:30]}...", "✅")
            else:
                self.set_status("No speech detected", "🎤")
        except Exception as e:
            self.set_status(f"Error: {str(e)[:30]}", "❌")
        finally:
            self.processing = False
            # Reset status after delay (set_status is already thread-safe)
            threading.Timer(2.0, lambda: self.set_status("Ready - Hold Right Ctrl to record", "🎤")).start()

    def on_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.ctrl_r:
                self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            if key == keyboard.Key.ctrl_r:
                # Process in a thread to not block
                threading.Thread(target=self.process_recording, daemon=True).start()
        except AttributeError:
            pass
        return True

    def run(self):
        """Start the app with keyboard listener."""
        # Start keyboard listener in background
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

        # Timer to process UI updates on main thread
        self.ui_timer = rumps.Timer(self._process_ui_queue, 0.1)
        self.ui_timer.start()

        device_name = sd.query_devices(self.selected_device)['name'] if self.selected_device else "None"
        print(f"Using microphone: {device_name}")
        print(f"Using model: {self.model_repo}")

        # Pre-load model in background
        def startup_load():
            preload_model(self.model_repo, on_status=self.set_status)
            print("Ready! Hold Right Control key to record, release to transcribe and paste.")

        threading.Thread(target=startup_load, daemon=True).start()

        # Run the menu bar app
        super().run()


def main():
    """Main entry point."""
    print("Wispy - Voice to Text")
    print("=" * 50)
    print()
    print("Note: You may need to grant Accessibility and Microphone permissions")
    print("in System Settings > Privacy & Security")
    print()
    print("Starting menu bar app...")

    app = WispyApp()
    app.run()


if __name__ == "__main__":
    main()
