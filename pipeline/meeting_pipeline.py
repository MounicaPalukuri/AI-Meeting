"""
End-to-End Meeting Intelligence Pipeline
Orchestrates the full flow: Audio → Preprocess → Transcribe → Analyze → Results

This module ties together all pipeline components into a single, easy-to-use interface.
"""

import os
import time
import logging
from typing import Optional

from models.schemas import MeetingAnalysis
from pipeline.audio_preprocessor import AudioPreprocessor
from pipeline.transcriber import WhisperTranscriber
from pipeline.llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class MeetingPipeline:
    """
    Complete meeting intelligence pipeline.
    
    Usage:
        pipeline = MeetingPipeline()
        result = pipeline.process("meeting_audio.mp3")
        print(result.summary)
        print(result.formatted_action_items)
    """

    def __init__(
        self,
        whisper_model: str = "base",
        whisper_backend: str = "faster-whisper",
        whisper_cpp_path: Optional[str] = None,
        llm_model: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
        device: str = "auto",
        language: Optional[str] = None,
    ):
        """
        Initialize all pipeline components.
        
        Args:
            whisper_model:    Whisper model size (tiny/base/small/medium/large-v3).
            whisper_backend:  "faster-whisper" or "whisper-cpp".
            whisper_cpp_path: Path to whisper.cpp binary (only for whisper-cpp backend).
            llm_model:        Ollama model name.
            ollama_host:      Ollama server URL.
            device:           Compute device (auto/cpu/cuda).
            language:         Audio language code (None = auto-detect).
        """
        logger.info("Initializing Meeting Intelligence Pipeline...")

        # Initialize components
        self.preprocessor = AudioPreprocessor()

        self.transcriber = WhisperTranscriber(
            model_size=whisper_model,
            backend=whisper_backend,
            whisper_cpp_path=whisper_cpp_path,
            device=device,
            language=language,
        )

        self.analyzer = LLMAnalyzer(
            model=llm_model,
            ollama_host=ollama_host,
        )

        logger.info("Pipeline initialized successfully.")

    def process(
        self,
        audio_path: str,
        progress_callback=None,
    ) -> MeetingAnalysis:
        """
        Process a meeting audio file through the complete pipeline.
        
        Args:
            audio_path:        Path to the audio file.
            progress_callback: Optional callable(step, total, message) for progress updates.
        
        Returns:
            MeetingAnalysis with transcript, summary, action items, and deadlines.
        """
        analysis = MeetingAnalysis(audio_file=os.path.basename(audio_path))
        total_steps = 5

        def update_progress(step: int, message: str):
            logger.info(f"[Step {step}/{total_steps}] {message}")
            if progress_callback:
                progress_callback(step / total_steps, message)

        start_time = time.time()

        try:
            # ── Step 1: Validate ─────────────────────────────────────
            update_progress(1, "🔍 Validating audio file...")
            from utils.validators import validate_audio_file
            is_valid, msg = validate_audio_file(audio_path)
            if not is_valid:
                analysis.error = msg
                return analysis

            # ── Step 2: Preprocess ───────────────────────────────────
            update_progress(2, "🎛️ Preprocessing audio with FFmpeg...")
            processed_path, audio_info = self.preprocessor.preprocess(audio_path)
            analysis.duration_seconds = audio_info.get("duration", 0)

            # ── Step 3: Transcribe ───────────────────────────────────
            update_progress(3, "🎙️ Transcribing speech to text...")
            transcript, segments = self.transcriber.transcribe(processed_path)
            analysis.transcript = transcript
            analysis.segments = segments

            if transcript == "[No speech detected in the audio]":
                analysis.summary = "No speech was detected in the audio file."
                analysis.error = None
                return analysis

            # ── Step 4: Generate Summary ─────────────────────────────
            update_progress(4, "📝 Generating meeting summary...")
            analysis.summary = self.analyzer.generate_summary(transcript)

            # ── Step 5: Extract Action Items & Deadlines ─────────────
            update_progress(5, "✅ Extracting action items and deadlines...")
            analysis.action_items = self.analyzer.extract_action_items(transcript)
            analysis.deadlines = self.analyzer.extract_deadlines(transcript)

            elapsed = time.time() - start_time
            logger.info(f"Pipeline complete in {elapsed:.1f}s")

            return analysis

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Pipeline failed after {elapsed:.1f}s: {e}")
            analysis.error = f"Pipeline error: {str(e)}"
            return analysis

        finally:
            # Clean up temporary files
            try:
                self.preprocessor.cleanup()
            except Exception:
                pass
