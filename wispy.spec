# -*- mode: python ; coding: utf-8 -*-
"""
Wispy PyInstaller spec file for macOS ARM64
Optimized for MLX-based transcription with proper native library handling
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Find site-packages for collecting dependencies
import site
site_packages = site.getsitepackages()[0]

# Collect MLX native libraries
mlx_path = os.path.join(site_packages, 'mlx')
mlx_binaries = []
mlx_datas = []
if os.path.exists(mlx_path):
    # Both files are in mlx/lib/ directory
    lib_dir = os.path.join(mlx_path, 'lib')
    if os.path.exists(lib_dir):
        libmlx = os.path.join(lib_dir, 'libmlx.dylib')
        metallib = os.path.join(lib_dir, 'mlx.metallib')
        if os.path.exists(libmlx):
            mlx_binaries.append((libmlx, 'mlx/lib'))
        if os.path.exists(metallib):
            # metallib is a data file, not a binary
            mlx_datas.append((metallib, 'mlx/lib'))

# Collect sounddevice/portaudio
sounddevice_path = os.path.join(site_packages, '_sounddevice_data', 'portaudio-binaries')
sounddevice_binaries = []
if os.path.exists(sounddevice_path):
    for f in os.listdir(sounddevice_path):
        if f.endswith('.dylib'):
            sounddevice_binaries.append((os.path.join(sounddevice_path, f), '_sounddevice_data/portaudio-binaries'))

# Collect mlx_whisper assets (tiktoken files, etc)
mlx_whisper_path = os.path.join(site_packages, 'mlx_whisper')
mlx_whisper_datas = []
if os.path.exists(mlx_whisper_path):
    assets_dir = os.path.join(mlx_whisper_path, 'assets')
    if os.path.exists(assets_dir):
        mlx_whisper_datas.append((assets_dir, 'mlx_whisper/assets'))

# Collect tiktoken data files
tiktoken_path = os.path.join(site_packages, 'tiktoken')
tiktoken_datas = []
if os.path.exists(tiktoken_path):
    tiktoken_datas.append((tiktoken_path, 'tiktoken'))

# Collect ffmpeg binary (required for audio processing)
import shutil
ffmpeg_binaries = []
ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path:
    ffmpeg_binaries.append((ffmpeg_path, '.'))

a = Analysis(
    ['wispy.py'],
    pathex=[],
    binaries=mlx_binaries + sounddevice_binaries + ffmpeg_binaries,
    datas=mlx_datas + mlx_whisper_datas + tiktoken_datas,
    hiddenimports=[
        # Core ML dependencies
        'mlx',
        'mlx.core',
        'mlx.nn',
        'mlx._reprlib_fix',  # Required for MLX initialization
        'mlx_whisper',
        'mlx_lm',
        'mlx_lm.utils',
        'mlx_lm.models',

        # Parakeet support
        'parakeet_mlx',

        # Tokenization
        'tiktoken',
        'tiktoken.core',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',

        # Data handling
        'dacite',
        'huggingface_hub',
        'safetensors',

        # Audio
        'sounddevice',
        'soundfile',
        'webrtcvad',

        # UI and system
        'rumps',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._darwin',
        'pyperclip',

        # Standard library that may need hints
        'multiprocessing',
        'multiprocessing.resource_tracker',
        'multiprocessing.sharedctypes',

        # Local modules
        'streaming',
        'vad',
        'llm_processor',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'matplotlib',
        'PIL',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Wispy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # CRITICAL: Disable UPX to prevent ARM64 binary corruption
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file='entitlements.plist',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # CRITICAL: Disable UPX
    upx_exclude=[],
    name='Wispy',
)

app = BUNDLE(
    coll,
    name='Wispy.app',
    icon=None,  # Add icon path here if available: 'icon.icns'
    bundle_identifier='com.wispy.app',
    info_plist={
        'CFBundleName': 'Wispy',
        'CFBundleDisplayName': 'Wispy',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'LSUIElement': True,  # Menu bar app, no dock icon
        'NSMicrophoneUsageDescription': 'Wispy needs microphone access to transcribe your speech.',
        'NSAppleEventsUsageDescription': 'Wispy needs accessibility access to paste transcribed text.',
        'NSAccessibilityUsageDescription': 'Wispy needs accessibility access to listen for keyboard hotkeys and paste transcribed text.',
        'LSArchitecturePriority': ['arm64'],
        'NSHighResolutionCapable': True,
    },
)
