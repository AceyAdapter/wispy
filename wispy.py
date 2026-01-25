#!/usr/bin/env python3
"""Wispy - Local voice-to-text tool using Whisper on Apple Silicon."""

# CRITICAL: Must be at the very top before any other imports
# Prevents fork bomb when bundled with PyInstaller on macOS
import multiprocessing
multiprocessing.freeze_support()

import sys
import tempfile
import os
import json
import threading
import time
import signal
import atexit
import fcntl
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
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from parakeet_mlx import from_pretrained as parakeet_from_pretrained

import streaming
from streaming import StreamingTranscriber

# Single-instance lock file
LOCK_FILE = Path.home() / ".config" / "wispy" / "wispy.lock"
_lock_fd = None

# Config file path
CONFIG_PATH = Path.home() / ".config" / "wispy" / "config.json"

# Default hotkey configuration
DEFAULT_HOTKEYS = {
    "hold_key": "ctrl_r",        # Hold to record, release to transcribe
    "toggle_modifier": "alt_l",  # Modifier for toggle mode (+ Space)
}

# Map of key names to pynput Key objects
KEY_MAP = {
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "cmd_r": keyboard.Key.cmd_r,
    "cmd_l": keyboard.Key.cmd_l,
    "shift_r": keyboard.Key.shift_r,
    "shift_l": keyboard.Key.shift_l,
    "space": keyboard.Key.space,
    "tab": keyboard.Key.tab,
    "caps_lock": keyboard.Key.caps_lock,
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
}

# Display names for keys
KEY_DISPLAY = {
    "ctrl_r": "Right Control",
    "ctrl_l": "Left Control",
    "alt_r": "Right Option",
    "alt_l": "Left Option",
    "cmd_r": "Right Command",
    "cmd_l": "Left Command",
    "shift_r": "Right Shift",
    "shift_l": "Left Shift",
    "space": "Space",
    "tab": "Tab",
    "caps_lock": "Caps Lock",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
    "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
    "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
}

# Configuration
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1

# Safety limits to prevent memory exhaustion
MAX_RECORDING_SECONDS = 300  # 5 minutes max recording
MAX_RECORDING_SAMPLES = SAMPLE_RATE * MAX_RECORDING_SECONDS

# Available engines
ENGINES = ["whisper", "parakeet"]
DEFAULT_ENGINE = "whisper"

# Available models per engine (repo, display name, approximate size)
WHISPER_MODELS = [
    ("mlx-community/whisper-tiny-mlx", "Tiny", "~75 MB"),
    ("mlx-community/whisper-base-mlx", "Base", "~140 MB"),
    ("mlx-community/whisper-small-mlx", "Small", "~460 MB"),
    ("mlx-community/whisper-medium-mlx", "Medium", "~1.5 GB"),
    ("mlx-community/whisper-large-v3-mlx", "Large v3", "~3 GB"),
]

PARAKEET_MODELS = [
    ("mlx-community/parakeet-tdt-0.6b-v2", "Parakeet 0.6B v2", "~2.5 GB"),
]

DEFAULT_MODELS = {
    "whisper": "mlx-community/whisper-base-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
}


def load_config():
    """Load configuration from file, returning defaults if not found."""
    config = {
        "hotkeys": DEFAULT_HOTKEYS.copy(),
        "engine": DEFAULT_ENGINE,
        "model": None,  # Will use DEFAULT_MODELS[engine] if None
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
                # Load hotkeys
                saved_hotkeys = saved.get("hotkeys", {})
                valid = {k: saved_hotkeys[k] for k in DEFAULT_HOTKEYS if k in saved_hotkeys}
                config["hotkeys"] = {**DEFAULT_HOTKEYS, **valid}
                # Load engine and model
                if "engine" in saved and saved["engine"] in ENGINES:
                    config["engine"] = saved["engine"]
                if "model" in saved:
                    config["model"] = saved["model"]
        except Exception as e:
            print(f"Error loading config: {e}")
    return config


def save_config(config):
    """Save configuration to file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


def acquire_single_instance_lock():
    """
    Acquire a file lock to ensure only one instance runs.
    Returns True if lock acquired, False if another instance is running.
    """
    global _lock_fd
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        return True
    except (IOError, OSError):
        if _lock_fd:
            _lock_fd.close()
            _lock_fd = None
        return False


def release_single_instance_lock():
    """Release the single-instance file lock."""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None

    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# Global app reference for signal handlers
_app_instance = None


def _signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    print(f"\nReceived signal {signum}, shutting down...")
    if _app_instance:
        _app_instance.cleanup()
    release_single_instance_lock()
    sys.exit(0)


def key_matches(pressed_key, key_name):
    """Check if a pressed key matches a configured key name."""
    # Check special keys
    if key_name in KEY_MAP:
        return pressed_key == KEY_MAP[key_name]
    # Check character keys (stored as "char_x")
    if key_name.startswith("char_") and hasattr(pressed_key, 'char'):
        return pressed_key.char == key_name[5:]
    return False


# Track loaded model to avoid redundant loads
_loaded_model_repo = None
_loaded_engine = None
_parakeet_model = None
_model_operation_in_progress = False  # Prevent concurrent model operations


def _is_model_downloaded(model_repo: str) -> bool:
    """Check if a model is already downloaded."""
    try:
        cache_info = scan_cache_dir()
        return model_repo in {repo.repo_id for repo in cache_info.repos}
    except Exception:
        return False


class ProgressTracker:
    """Track download progress across multiple files."""
    def __init__(self, on_status):
        self.on_status = on_status
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.last_update = 0
        self.current_file_size = 0
        self.current_file_downloaded = 0

    def update(self, n):
        """Called by tqdm with number of bytes downloaded."""
        self.current_file_downloaded += n
        self.downloaded_bytes += n
        now = time.time()
        if now - self.last_update > 0.3:  # Update every 300ms
            self._show_progress()
            self.last_update = now

    def _show_progress(self):
        if self.on_status and self.total_bytes > 0:
            total_mb = self.total_bytes / (1024 * 1024)
            if total_mb >= 1000:
                self.on_status(f"Downloading ({total_mb/1024:.1f} GB)", "⬇️")
            else:
                self.on_status(f"Downloading ({total_mb:.0f} MB)", "⬇️")

    def file_start(self, size):
        """Called when starting a new file."""
        self.current_file_size = size
        self.current_file_downloaded = 0

    def file_done(self):
        """Called when a file is done."""
        pass


def _download_model_with_progress(model_repo: str, on_status=None):
    """Download a model with progress updates."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import enable_progress_bars

    if on_status:
        on_status("Checking download...", "⬇️")

    # Get list of files and total size
    api = HfApi()
    try:
        repo_info = api.repo_info(model_repo, files_metadata=True)
        files = repo_info.siblings or []
        total_size = sum(f.size or 0 for f in files)
    except Exception:
        total_size = 0
        files = []

    tracker = ProgressTracker(on_status)
    tracker.total_bytes = total_size

    if on_status and total_size > 0:
        total_mb = total_size / (1024 * 1024)
        if total_mb >= 1000:
            on_status(f"Downloading ({total_mb/1024:.1f} GB)", "⬇️")
        else:
            on_status(f"Downloading ({total_mb:.0f} MB)", "⬇️")

    # Download with progress tracking using tqdm override
    import tqdm as tqdm_module
    original_tqdm = tqdm_module.tqdm

    class ProgressTqdm(original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['disable'] = False  # Enable so we get updates
            super().__init__(*args, **kwargs)
            if hasattr(self, 'total') and self.total:
                tracker.file_start(self.total)

        def update(self, n=1):
            super().update(n)
            tracker.update(n)

        def close(self):
            super().close()
            tracker.file_done()

    # Temporarily replace tqdm
    tqdm_module.tqdm = ProgressTqdm
    enable_progress_bars()

    try:
        snapshot_download(model_repo)
    finally:
        # Restore original tqdm
        tqdm_module.tqdm = original_tqdm
        disable_progress_bars()


def preload_model(model_repo: str, engine: str = "whisper", on_status=None, on_complete=None):
    """Pre-load a model by running a dummy transcription."""
    global _loaded_model_repo, _loaded_engine, _parakeet_model, _model_operation_in_progress

    if _loaded_model_repo == model_repo and _loaded_engine == engine:
        print(f"Model already loaded: {model_repo}")
        if on_complete:
            on_complete()
        return

    # Prevent concurrent operations
    if _model_operation_in_progress:
        print("Model operation already in progress, skipping")
        return

    _model_operation_in_progress = True

    try:
        # Check if model needs to be downloaded
        needs_download = not _is_model_downloaded(model_repo)
        if needs_download:
            if on_status:
                on_status("Downloading...", "⬇️")
            print(f"Downloading model: {model_repo}")
            _download_model_with_progress(model_repo, on_status)

        if on_status:
            on_status("Loading model...", "⏳")

        print(f"Pre-loading model: {model_repo} (engine: {engine})")

        if engine == "parakeet":
            # Load Parakeet model
            _parakeet_model = parakeet_from_pretrained(model_repo)
            _loaded_model_repo = model_repo
            _loaded_engine = engine
            print(f"Model loaded: {model_repo}")
            if on_status:
                on_status("Ready", "🎤")
        else:
            # Load Whisper model with dummy transcription
            silent_audio = np.zeros(SAMPLE_RATE, dtype=np.int16)
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            try:
                wavfile.write(temp_path, SAMPLE_RATE, silent_audio)
                mlx_whisper.transcribe(temp_path, path_or_hf_repo=model_repo)
                _loaded_model_repo = model_repo
                _loaded_engine = engine
                print(f"Model loaded: {model_repo}")
                if on_status:
                    on_status("Ready", "🎤")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Notify completion (to refresh menu icons)
        if on_complete:
            on_complete()

    except Exception as e:
        print(f"Error loading model: {e}")
        if on_status:
            on_status("Error loading model", "❌")
    finally:
        _model_operation_in_progress = False


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

        # Load configuration
        self.config = load_config()
        self.hotkeys = self.config["hotkeys"]

        # Engine and model selection
        self.engine = self.config["engine"]
        # Use saved model if valid for current engine, otherwise use default
        saved_model = self.config.get("model")
        valid_repos = [m[0] for m in (WHISPER_MODELS if self.engine == "whisper" else PARAKEET_MODELS)]
        if saved_model and saved_model in valid_repos:
            self.model_repo = saved_model
        else:
            self.model_repo = DEFAULT_MODELS[self.engine]
        self.modifier_pressed = False  # Track toggle modifier key state
        self.capturing_hotkey = None   # Which hotkey we're capturing (None if not capturing)

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
        self.status_item = rumps.MenuItem("Ready")
        self.status_item.set_callback(None)

        # Device submenu
        self.device_menu = rumps.MenuItem("Microphone")
        self._build_device_menu()

        # Engine submenu
        self.engine_menu = rumps.MenuItem("Engine")
        self._build_engine_menu()

        # Model submenu
        self.model_menu = rumps.MenuItem("Model")
        self._build_model_menu()

        # Streaming mode toggle
        self.streaming_toggle = rumps.MenuItem(
            "Streaming Mode",
            callback=self._toggle_streaming_mode
        )
        self.streaming_toggle.state = self.streaming_mode

        # Hotkeys submenu
        self.hotkeys_menu = rumps.MenuItem("Hotkeys")
        self._build_hotkeys_menu()

        self.menu = [
            self.status_item,
            None,  # Separator
            self.device_menu,
            self.engine_menu,
            self.model_menu,
            self.hotkeys_menu,
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

    def _save_config(self):
        """Save current configuration to file."""
        self.config["hotkeys"] = self.hotkeys
        self.config["engine"] = self.engine
        self.config["model"] = self.model_repo
        save_config(self.config)

    def _build_engine_menu(self):
        """Build the engine selection submenu."""
        engine_names = {
            "whisper": "Whisper (99+ languages)",
            "parakeet": "Parakeet (English, 30x faster)",
        }
        for engine in ENGINES:
            label = engine_names.get(engine, engine)
            item = rumps.MenuItem(label, callback=self._select_engine)
            item.engine_id = engine
            item.state = (engine == self.engine)
            self.engine_menu[label] = item

    def _select_engine(self, sender):
        """Handle engine selection from menu."""
        # Prevent switching while download/load in progress
        if _model_operation_in_progress:
            rumps.notification("Wispy", "Please wait", "Model operation in progress")
            return

        new_engine = sender.engine_id

        # Already selected
        if new_engine == self.engine:
            return

        # Check if switching to Parakeet and model not downloaded
        if new_engine == "parakeet":
            downloaded = self._get_downloaded_models()
            parakeet_repo = DEFAULT_MODELS["parakeet"]
            if parakeet_repo not in downloaded:
                # Show warning dialog
                response = rumps.alert(
                    title="Download Required",
                    message="Parakeet requires downloading ~2.5 GB. Continue?",
                    ok="Download",
                    cancel="Cancel"
                )
                if response != 1:  # User cancelled
                    return

        self.engine = new_engine
        self.model_repo = DEFAULT_MODELS[self.engine]
        self._save_config()

        # Update engine checkmarks
        for item in self.engine_menu.values():
            if hasattr(item, 'engine_id'):
                item.state = (item.engine_id == self.engine)

        # Rebuild model menu for new engine
        self._build_model_menu()

        engine_name = "Whisper" if self.engine == "whisper" else "Parakeet"
        rumps.notification("Wispy", "Engine changed", f"Using {engine_name}")
        print(f"Engine changed: {engine_name}")

        # Load default model for new engine
        def load():
            preload_model(self.model_repo, self.engine, on_status=self.set_status, on_complete=self._build_model_menu)

        threading.Thread(target=load, daemon=True).start()

    def _build_model_menu(self):
        """Build the model selection submenu."""
        # Clear existing items
        if hasattr(self.model_menu, '_menu') and self.model_menu._menu:
            self.model_menu.clear()

        downloaded = self._get_downloaded_models()
        models = WHISPER_MODELS if self.engine == "whisper" else PARAKEET_MODELS

        for repo, name, size in models:
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
        # Prevent switching while download/load in progress
        if _model_operation_in_progress:
            rumps.notification("Wispy", "Please wait", "Model operation in progress")
            return

        self.model_repo = sender.model_repo
        self._save_config()

        # Update checkmarks
        for item in self.model_menu.values():
            if hasattr(item, 'model_repo'):
                item.state = (item.model_repo == self.model_repo)

        print(f"Model changed: {sender.model_name} ({sender.model_repo})")

        # Load model in background thread
        def load():
            preload_model(self.model_repo, self.engine, on_status=self.set_status, on_complete=self._build_model_menu)
            rumps.notification("Wispy", "Model ready", sender.model_name)

        threading.Thread(target=load, daemon=True).start()

    def _toggle_streaming_mode(self, sender):
        """Toggle streaming transcription mode."""
        self.streaming_mode = not self.streaming_mode
        sender.state = self.streaming_mode
        mode = "Streaming" if self.streaming_mode else "Standard"
        rumps.notification("Wispy", "Mode changed", f"{mode} mode enabled")
        print(f"Mode changed: {mode}")

    def _build_hotkeys_menu(self):
        """Build the hotkeys configuration submenu."""
        # Clear existing items (only if menu is initialized)
        if hasattr(self.hotkeys_menu, '_menu') and self.hotkeys_menu._menu:
            self.hotkeys_menu.clear()

        # Hold key
        hold_display = KEY_DISPLAY.get(self.hotkeys["hold_key"], self.hotkeys["hold_key"])
        hold_item = rumps.MenuItem(
            f"Hold to Record: {hold_display}",
            callback=self._configure_hold_key
        )
        self.hotkeys_menu["hold"] = hold_item

        # Toggle modifier (always paired with Space)
        mod_display = KEY_DISPLAY.get(self.hotkeys["toggle_modifier"], self.hotkeys["toggle_modifier"])
        mod_item = rumps.MenuItem(
            f"Toggle: {mod_display} + Space",
            callback=self._configure_toggle_modifier
        )
        self.hotkeys_menu["modifier"] = mod_item

    def _configure_hold_key(self, sender):
        """Start capturing the hold-to-record key."""
        self.capturing_hotkey = "hold_key"
        self.set_status("Press new Hold key...", "⌨️")
        rumps.notification("Wispy", "Configure Hotkey", "Press the key to use for hold-to-record")

    def _configure_toggle_modifier(self, sender):
        """Start capturing the toggle modifier key."""
        self.capturing_hotkey = "toggle_modifier"
        self.set_status("Press new Modifier...", "⌨️")
        rumps.notification("Wispy", "Configure Hotkey", "Press the modifier key for toggle mode")

    def _capture_key(self, key):
        """Capture a key press for hotkey configuration."""
        # Get the key name - check special keys first
        key_name = None
        for name, k in KEY_MAP.items():
            if key == k:
                key_name = name
                break

        # Check for character keys (letters, numbers, etc.)
        if key_name is None and hasattr(key, 'char') and key.char:
            key_name = f"char_{key.char}"

        if key_name is None:
            self.set_status("Unsupported key, try again", "❌")
            return False

        # Update the hotkey
        self.hotkeys[self.capturing_hotkey] = key_name
        self._save_config()
        self._build_hotkeys_menu()

        key_display = KEY_DISPLAY.get(key_name, key_name.replace("char_", "").upper())
        self.set_status(f"Set to: {key_display}", "✅")
        rumps.notification("Wispy", "Hotkey Updated", f"Set to {key_display}")

        self.capturing_hotkey = None
        threading.Timer(1.5, lambda: self.set_status("Ready", "🎤")).start()
        return True

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
            # Check if we've exceeded max recording duration
            current_samples = sum(len(chunk) for chunk in self.audio_chunks)
            if current_samples >= MAX_RECORDING_SAMPLES:
                # Auto-stop recording to prevent memory exhaustion
                print("Warning: Max recording duration reached, auto-stopping")
                self.recording = False
                # Trigger processing in a thread
                threading.Thread(target=self.process_recording, daemon=True).start()
                return

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
                # Set Parakeet model reference for streaming module
                if self.engine == "parakeet":
                    streaming._parakeet_model = _parakeet_model

                self.streaming_transcriber = StreamingTranscriber(
                    model_repo=self.model_repo,
                    engine=self.engine,
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
            self.set_status("Ready", "🎤")
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
                threading.Timer(2.0, lambda: self.set_status("Ready", "🎤")).start()
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
                    if self.engine == "parakeet":
                        # Use Parakeet for transcription
                        global _parakeet_model
                        if _parakeet_model is None:
                            _parakeet_model = parakeet_from_pretrained(self.model_repo)
                        result = _parakeet_model.transcribe(temp_path)
                        text = result.text.strip()
                    else:
                        # Use Whisper for transcription
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
            threading.Timer(2.0, lambda: self.set_status("Ready", "🎤")).start()

    def on_press(self, key):
        """Handle key press events."""
        try:
            # If we're capturing a hotkey, handle that first
            if self.capturing_hotkey:
                self._capture_key(key)
                return

            # Track toggle modifier key state
            if key_matches(key, self.hotkeys["toggle_modifier"]):
                self.modifier_pressed = True
            # Toggle recording on modifier + Space
            elif key == keyboard.Key.space and self.modifier_pressed:
                if not self.recording:
                    self.start_recording()
                else:
                    # Process in a thread to not block
                    threading.Thread(target=self.process_recording, daemon=True).start()
            # Hold key to record
            elif key_matches(key, self.hotkeys["hold_key"]):
                self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            if key_matches(key, self.hotkeys["toggle_modifier"]):
                self.modifier_pressed = False
            # Release hold key to stop and transcribe
            elif key_matches(key, self.hotkeys["hold_key"]):
                threading.Thread(target=self.process_recording, daemon=True).start()
        except AttributeError:
            pass
        return True

    def cleanup(self):
        """Clean up resources before shutdown."""
        print("Cleaning up...")

        # Stop any ongoing recording
        if self.recording:
            self.recording = False
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

        # Stop streaming transcriber
        if self.streaming_transcriber:
            try:
                self.streaming_transcriber.stop()
            except Exception:
                pass
            self.streaming_transcriber = None

        # Stop keyboard listener
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
            self.keyboard_listener = None

        # Stop UI timer
        if hasattr(self, 'ui_timer') and self.ui_timer:
            try:
                self.ui_timer.stop()
            except Exception:
                pass

        print("Cleanup complete.")

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
            preload_model(self.model_repo, self.engine, on_status=self.set_status, on_complete=self._build_model_menu)
            print("Ready! Hold Right Ctrl or press Left Option+Space to record.")

        threading.Thread(target=startup_load, daemon=True).start()

        # Run the menu bar app
        super().run()


def main():
    """Main entry point."""
    global _app_instance

    print("Wispy - Voice to Text")
    print("=" * 50)
    print()

    # Check for single instance
    if not acquire_single_instance_lock():
        print("Error: Wispy is already running.")
        print("Check your menu bar or use Activity Monitor to quit the existing instance.")
        sys.exit(1)

    # Register cleanup handlers
    atexit.register(release_single_instance_lock)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    print("Note: You may need to grant Accessibility and Microphone permissions")
    print("in System Settings > Privacy & Security")
    print()
    print("Starting menu bar app...")

    try:
        app = WispyApp()
        _app_instance = app
        app.run()
    finally:
        if _app_instance:
            _app_instance.cleanup()
        release_single_instance_lock()


if __name__ == "__main__":
    main()
