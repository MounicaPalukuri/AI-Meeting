"""
Input validation utilities for the Meeting Intelligence System.
Handles file validation, dependency checks, and error reporting.
"""

import os
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".webm"}

# Maximum file size: 500MB
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_audio_file(file_path: str) -> tuple[bool, str]:
    """
    Validate an uploaded audio file.
    
    Returns:
        (is_valid, message) tuple
    """
    if not file_path:
        return False, "❌ No file provided. Please upload an audio file."

    if not os.path.exists(file_path):
        return False, f"❌ File not found: {file_path}"

    # Check file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        return False, (
            f"❌ Unsupported file format: '{ext}'\n"
            f"Supported formats: {supported}"
        )

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "❌ The uploaded file is empty (0 bytes)."

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, (
            f"❌ File too large: {size_mb:.1f}MB\n"
            f"Maximum allowed: {MAX_FILE_SIZE_MB}MB"
        )

    # Quick sanity check — try to probe with FFmpeg
    if shutil.which("ffprobe"):
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return False, (
                    "❌ The file appears to be corrupted or is not a valid audio file.\n"
                    f"FFprobe error: {result.stderr.strip()}"
                )
            duration = float(result.stdout.strip())
            if duration < 0.5:
                return False, "❌ Audio file is too short (less than 0.5 seconds)."
        except (subprocess.TimeoutExpired, ValueError) as e:
            logger.warning(f"FFprobe validation warning: {e}")
            # Don't fail — FFprobe might have issues but file could still be okay

    return True, "✅ File validated successfully."


def check_dependencies() -> dict[str, dict]:
    """
    Check if all required external dependencies are available.

    Returns:
        Dictionary with dependency status information.
    """
    deps = {}

    # Check FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            deps["ffmpeg"] = {"installed": True, "path": ffmpeg_path, "version": version_line}
        except Exception:
            deps["ffmpeg"] = {"installed": True, "path": ffmpeg_path, "version": "unknown"}
    else:
        deps["ffmpeg"] = {
            "installed": False,
            "message": "FFmpeg not found. Install from https://ffmpeg.org/download.html"
        }

    # Check Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=15
            )
            models = result.stdout.strip() if result.returncode == 0 else "Could not list models"
            deps["ollama"] = {"installed": True, "path": ollama_path, "models": models}
        except Exception:
            deps["ollama"] = {"installed": True, "path": ollama_path, "models": "unknown"}
    else:
        deps["ollama"] = {
            "installed": False,
            "message": "Ollama not found. Install from https://ollama.com/download"
        }

    return deps


def check_ollama_model(model_name: str) -> tuple[bool, str]:
    """
    Check if a specific Ollama model is available.
    
    Returns:
        (available, message) tuple
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return False, "❌ Ollama is not running. Start it with: ollama serve"
        
        if model_name in result.stdout:
            return True, f"✅ Model '{model_name}' is available."
        else:
            return False, (
                f"⚠️ Model '{model_name}' not found.\n"
                f"Pull it with: ollama pull {model_name}\n"
                f"Available models:\n{result.stdout}"
            )
    except FileNotFoundError:
        return False, "❌ Ollama is not installed. Install from https://ollama.com/download"
    except subprocess.TimeoutExpired:
        return False, "❌ Ollama is not responding. Make sure it's running: ollama serve"
