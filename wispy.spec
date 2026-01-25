# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = [
    'streaming',
    'vad',
    # Multiprocessing support
    'multiprocessing',
    'multiprocessing.resource_tracker',
    'multiprocessing.sharedctypes',
]

# Collect all necessary packages
packages_to_collect = [
    'mlx',
    'mlx_whisper',
    'parakeet_mlx',
    'huggingface_hub',
    'scipy',
    'sounddevice',
    'webrtcvad',
    'tqdm',
    'numpy',
    'rumps',
    'pynput',
    'pyperclip',
]

for pkg in packages_to_collect:
    try:
        tmp_ret = collect_all(pkg)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect {pkg}: {e}")

# Ensure scipy submodules are included
hiddenimports += collect_submodules('scipy')

a = Analysis(
    ['wispy.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Wispy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Wispy',
)

app = BUNDLE(
    coll,
    name='Wispy.app',
    icon=None,
    bundle_identifier='com.wispy.voicetotext',
    info_plist={
        'CFBundleName': 'Wispy',
        'CFBundleDisplayName': 'Wispy',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'LSUIElement': True,  # Menu bar app - hides from dock
        'NSMicrophoneUsageDescription': 'Wispy needs microphone access to capture your voice for transcription.',
        'NSAppleEventsUsageDescription': 'Wispy needs accessibility access to simulate keyboard input for pasting transcribed text.',
        # Required for Accessibility permissions
        'NSAccessibilityUsageDescription': 'Wispy needs accessibility access to detect hotkey presses and paste transcribed text.',
    },
)
