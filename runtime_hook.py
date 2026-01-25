"""
PyInstaller runtime hook for multiprocessing support on macOS.

This hook ensures that multiprocessing works correctly when the app is
bundled with PyInstaller. Without this, spawned processes would try to
re-execute the entire application, causing a fork bomb.
"""

import sys
import multiprocessing

# On macOS, the default start method is 'spawn' which requires freeze_support()
# This must be called before any other multiprocessing code runs
if sys.platform == 'darwin':
    multiprocessing.freeze_support()

    # Also set the start method explicitly to 'spawn' for consistency
    # This is the safest option for bundled apps
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
