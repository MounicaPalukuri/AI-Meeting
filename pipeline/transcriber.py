"""
Whisper Transcription Module
Converts speech to text using faster-whisper (CTranslate2 backend — C++ inference).
This provides whisper.cpp-level performance via Python bindings.

Supports two backends:
  1. faster-whisper (default, pip-installable, CTranslate2/C++ engine)
  2. whisper.cpp CLI (if you've compiled it from source)
"""

import os
import subprocess
import logging
from typing import Optional, List

from models.schemas import TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Speech-to-text transcriber using Whisper models.
    
    Default backend: faster-whisper (CTranslate2).
    Optional backend: whisper.cpp CLI binary.
    """

    def __init__(
        self,
        model_size: str = "base",
        backend: str = "faster-whisper",
        whisper_cpp_path: Optional[str] = None,
        device: str = "auto",
        language: Optional[str] = None,
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_size:      Whisper model size (tiny, base, small, medium, large-v3).
            backend:         "faster-whisper" or "whisper-cpp".
            whisper_cpp_path: Path to compiled whisper.cpp binary (only for whisper-cpp backend).
            device:          Compute device: "auto", "cpu", or "cuda".
            language:        Language code (e.g., "en"). None = auto-detect.
        """
        self.model_size = model_size
        self.backend = backend
        self.whisper_cpp_path = whisper_cpp_path
        self.device = device
        self.language = language
        self._model = None

        if self.backend == "faster-whisper":
            self._init_faster_whisper()
        elif self.backend == "whisper-cpp":
            self._verify_whisper_cpp()

    def _init_faster_whisper(self):
        """Initialize the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            # Determine compute type based on device
            if self.device == "auto":
                try:
                    import torch
                    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    compute_device = "cpu"
            else:
                compute_device = self.device

            compute_type = "float16" if compute_device == "cuda" else "int8"

            logger.info(
                f"Loading faster-whisper model: {self.model_size} "
                f"(device={compute_device}, compute_type={compute_type})"
            )

            self._model = WhisperModel(
                self.model_size,
                device=compute_device,
                compute_type=compute_type,
            )

            logger.info("Faster-whisper model loaded successfully.")

        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed.\n"
                "Install it with: pip install faster-whisper\n"
                "Or switch to whisper-cpp backend."
            )

    def _verify_whisper_cpp(self):
        """Verify whisper.cpp binary is available."""
        if not self.whisper_cpp_path:
            raise RuntimeError(
                "whisper_cpp_path is required when using whisper-cpp backend.\n"
                "Provide the path to the compiled 'main' binary."
            )
        if not os.path.isfile(self.whisper_cpp_path):
            raise RuntimeError(
                f"whisper.cpp binary not found at: {self.whisper_cpp_path}\n"
                "Build whisper.cpp from https://github.com/ggerganov/whisper.cpp"
            )

    def transcribe(self, audio_path: str) -> tuple[str, List[TranscriptSegment]]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to preprocessed WAV audio file (16kHz mono).
        
        Returns:
            (full_text, segments) tuple.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting transcription: {audio_path} (backend={self.backend})")

        if self.backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio_path)
        elif self.backend == "whisper-cpp":
            return self._transcribe_whisper_cpp(audio_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _transcribe_faster_whisper(self, audio_path: str) -> tuple[str, List[TranscriptSegment]]:
        """Transcribe using faster-whisper (CTranslate2)."""
        try:
            segments_iter, info = self._model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,           # Filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
                word_timestamps=False,
            )

            detected_lang = info.language
            lang_prob = info.language_probability
            logger.info(f"Detected language: {detected_lang} (probability: {lang_prob:.2f})")

            segments = []
            full_text_parts = []

            for seg in segments_iter:
                transcript_seg = TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
                segments.append(transcript_seg)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)

            if not full_text.strip():
                logger.warning("Transcription produced empty text — audio may be silent or non-speech.")
                full_text = "[No speech detected in the audio]"

            logger.info(f"Transcription complete: {len(segments)} segments, {len(full_text)} chars")
            return full_text, segments

        except Exception as e:
            logger.error(f"Faster-whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    def _transcribe_whisper_cpp(self, audio_path: str) -> tuple[str, List[TranscriptSegment]]:
        """Transcribe using whisper.cpp CLI binary."""
        try:
            # Determine model file path (whisper.cpp convention)
            model_dir = os.path.dirname(self.whisper_cpp_path)
            model_file = os.path.join(model_dir, "models", f"ggml-{self.model_size}.bin")

            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Whisper.cpp model file not found: {model_file}\n"
                    f"Download it with: cd {model_dir} && bash models/download-ggml-model.sh {self.model_size}"
                )

            cmd = [
                self.whisper_cpp_path,
                "-m", model_file,
                "-f", audio_path,
                "--output-txt",
                "--print-progress", "false",
            ]

            if self.language:
                cmd.extend(["-l", self.language])

            logger.info(f"Running whisper.cpp: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                raise RuntimeError(f"whisper.cpp failed: {result.stderr}")

            # Parse output — whisper.cpp outputs timestamped lines
            full_text_parts = []
            segments = []

            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Parse timestamp format: [HH:MM:SS.mmm --> HH:MM:SS.mmm]  text
                if line.startswith("["):
                    try:
                        timestamp_end = line.index("]")
                        timestamp_str = line[1:timestamp_end]
                        text = line[timestamp_end + 1:].strip()

                        parts = timestamp_str.split(" --> ")
                        if len(parts) == 2:
                            start = self._parse_whisper_cpp_timestamp(parts[0].strip())
                            end = self._parse_whisper_cpp_timestamp(parts[1].strip())

                            segments.append(TranscriptSegment(
                                start=start, end=end, text=text
                            ))
                            full_text_parts.append(text)
                    except (ValueError, IndexError):
                        full_text_parts.append(line)
                else:
                    full_text_parts.append(line)

            full_text = " ".join(full_text_parts)

            if not full_text.strip():
                full_text = "[No speech detected in the audio]"

            logger.info(f"whisper.cpp transcription complete: {len(segments)} segments")
            return full_text, segments

        except subprocess.TimeoutExpired:
            raise RuntimeError("whisper.cpp transcription timed out (>10 minutes).")

    @staticmethod
    def _parse_whisper_cpp_timestamp(ts: str) -> float:
        """Parse whisper.cpp timestamp format HH:MM:SS.mmm to seconds."""
        parts = ts.split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return 0.0
