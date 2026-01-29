"""
Runtime hook for Wispy PyInstaller bundle
Configures environment for MLX and other native libraries
"""

import os
import sys
import multiprocessing

# Essential for macOS multiprocessing in frozen apps
multiprocessing.freeze_support()

def configure_environment():
    """Configure runtime environment for bundled app"""

    # Determine if running as bundled app
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        bundle_dir = sys._MEIPASS

        # Configure MLX library paths
        mlx_lib_dir = os.path.join(bundle_dir, 'mlx', 'lib')
        if os.path.exists(mlx_lib_dir):
            # Add to DYLD_LIBRARY_PATH for dynamic library loading
            existing_path = os.environ.get('DYLD_LIBRARY_PATH', '')
            if existing_path:
                os.environ['DYLD_LIBRARY_PATH'] = f"{mlx_lib_dir}:{existing_path}"
            else:
                os.environ['DYLD_LIBRARY_PATH'] = mlx_lib_dir

            # Set MLX_METALLIB_PATH to help MLX find the metallib file
            metallib_path = os.path.join(mlx_lib_dir, 'mlx.metallib')
            if os.path.exists(metallib_path):
                os.environ['MLX_METALLIB_PATH'] = metallib_path

        # Set HuggingFace cache to user directory (not inside bundle)
        # This allows model downloads to persist between app launches
        hf_cache = os.path.expanduser('~/.cache/huggingface')
        os.environ.setdefault('HF_HOME', hf_cache)
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(hf_cache, 'hub'))

        # Ensure cache directory exists
        os.makedirs(hf_cache, exist_ok=True)

        # Configure Metal framework hints
        os.environ.setdefault('METAL_DEVICE_WRAPPER_TYPE', '1')

        # Disable tokenizers parallelism to avoid fork issues
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

        # Add bundle directory to PATH for ffmpeg
        existing_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{bundle_dir}:{existing_path}"

# Run configuration immediately when hook is loaded
configure_environment()
