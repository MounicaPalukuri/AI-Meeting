"""
Audio Preprocessor Module
Handles audio file preprocessing using FFmpeg:
  - Converts any supported audio format to 16kHz mono WAV (required by Whisper)
  - Normalizes audio levels
  - Extracts audio metadata (duration, sample rate, etc.)
"""

import os
import subprocess
import shutil
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocesses audio files for transcription using FFmpeg.
    Whisper models require 16kHz mono WAV input for best results.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the audio preprocessor.
        
        Args:
            output_dir: Directory for processed audio files. 
                        Uses system temp dir if not specified.
        """
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="meeting_intel_")
        os.makedirs(self.output_dir, exist_ok=True)
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify that FFmpeg is installed and accessible."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "FFmpeg is not installed or not in PATH.\n"
                "Install FFmpeg:\n"
                "  Windows: winget install FFmpeg  (or download from https://ffmpeg.org)\n"
                "  macOS:   brew install ffmpeg\n"
                "  Linux:   sudo apt install ffmpeg"
            )
        logger.info("FFmpeg found on system PATH.")

    def get_audio_info(self, input_path: str) -> dict:
        """
        Extract audio metadata using FFprobe.
        
        Args:
            input_path: Path to the audio file.
        
        Returns:
            Dictionary with duration, sample_rate, channels, codec, and bitrate.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.warning(f"FFprobe failed: {result.stderr}")
                return {"duration": 0, "error": result.stderr}

            import json
            data = json.loads(result.stdout)

            # Extract format info
            fmt = data.get("format", {})
            duration = float(fmt.get("duration", 0))

            # Extract audio stream info
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            info = {
                "duration": duration,
                "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
                "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
                "codec": audio_stream.get("codec_name", "unknown") if audio_stream else "unknown",
                "bitrate": int(fmt.get("bit_rate", 0)),
            }

            logger.info(
                f"Audio info: {info['duration']:.1f}s, "
                f"{info['sample_rate']}Hz, "
                f"{info['channels']}ch, "
                f"{info['codec']}"
            )
            return info

        except subprocess.TimeoutExpired:
            logger.error("FFprobe timed out.")
            return {"duration": 0, "error": "FFprobe timed out"}
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {"duration": 0, "error": str(e)}

    def preprocess(self, input_path: str, normalize: bool = True) -> tuple[str, dict]:
        """
        Preprocess audio file for Whisper transcription.
        
        Converts to 16kHz mono WAV with optional loudness normalization.
        
        Args:
            input_path:  Path to the input audio file.
            normalize:   Whether to apply loudness normalization.
        
        Returns:
            (output_path, audio_info) tuple.
        """
        # Get source audio info
        audio_info = self.get_audio_info(input_path)

        # Build output filename
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_processed.wav")

        # Build FFmpeg command
        # Target: 16kHz, mono, 16-bit PCM WAV (optimal for Whisper)
        cmd = ["ffmpeg", "-y", "-i", input_path]

        if normalize:
            # Two-pass loudness normalization using loudnorm filter
            # First pass: analyze loudness levels
            analyze_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"
            ]
            try:
                analyze_result = subprocess.run(
                    analyze_cmd, capture_output=True, text=True, timeout=120
                )
                # If analysis succeeds, apply normalization
                if analyze_result.returncode == 0:
                    cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
                else:
                    logger.warning("Loudness analysis failed, skipping normalization.")
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"Loudness analysis error: {e}, skipping normalization.")

        # Output settings: 16kHz mono 16-bit PCM WAV
        cmd.extend([
            "-ar", "16000",      # 16kHz sample rate
            "-ac", "1",          # Mono channel
            "-sample_fmt", "s16", # 16-bit signed integer
            "-c:a", "pcm_s16le", # PCM 16-bit little-endian codec
            output_path
        ])

        logger.info(f"Preprocessing audio: {input_path} → {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown FFmpeg error"
                raise RuntimeError(f"FFmpeg preprocessing failed: {error_msg}")

            # Verify output file
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FFmpeg produced an empty or missing output file.")

            output_info = self.get_audio_info(output_path)
            audio_info["processed_duration"] = output_info.get("duration", 0)

            logger.info(f"Audio preprocessing complete: {output_path}")
            return output_path, audio_info

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Audio preprocessing timed out (>5 minutes). "
                "The file may be too large or corrupted."
            )

    def cleanup(self):
        """Remove temporary processed audio files."""
        if os.path.exists(self.output_dir) and self.output_dir.startswith(tempfile.gettempdir()):
            try:
                shutil.rmtree(self.output_dir)
                logger.info(f"Cleaned up temp directory: {self.output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {self.output_dir}: {e}")
