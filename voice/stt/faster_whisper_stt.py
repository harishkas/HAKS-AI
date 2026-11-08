"""Faster-Whisper Speech-to-Text Engine for HAKS.

Calibrated for Intel Core i5-7200U (2 cores, 4 threads @ 2.50 GHz, 8 GB RAM).
Uses int8 quantization on CPU with lazy model loading, automatic Tamil/English
language identification, and mixed Tamil-English prompt conditioning.
"""

import logging
import time
from typing import Optional
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from config.settings import AppConfig, get_settings
from voice.models import TranscriptionResult
from voice.stt.base import SpeechToTextInterface

logger = logging.getLogger("haks.voice.stt.faster_whisper")


class FasterWhisperSTT(SpeechToTextInterface):
    """Faster-Whisper CPU transcription with int8 quantization and lazy loading."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        model_size: Optional[str] = None,
    ) -> None:
        self.config = config or get_settings()
        self.model_size: str = model_size or self.config.voice.whisper_model_size
        self._model: Optional[Any] = None
        self._is_loading: bool = False
        self._device: str = "cpu"
        self._compute_type: str = "int8"  # High-efficiency CPU int8
        self._cpu_threads: int = 2  # Match physical cores on Core i5-7200U

    @property
    def name(self) -> str:
        return f"Faster-Whisper ({self.model_size}, CPU int8)"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> bool:
        """Lazily load Whisper model into system memory."""
        if self._model is not None:
            return True

        if WhisperModel is None:
            logger.error("faster_whisper package is not installed.")
            return False

        if self._is_loading:
            return False

        self._is_loading = True
        start_t = time.perf_counter()
        logger.info(
            "Loading Faster-Whisper '%s' model on CPU (int8, %d threads)...",
            self.model_size,
            self._cpu_threads,
        )

        try:
            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=1,
            )
            elapsed = time.perf_counter() - start_t
            logger.info("Faster-Whisper model loaded in %.2f seconds.", elapsed)
            return True
        except Exception as exc:
            logger.error("Failed to load Faster-Whisper model '%s': %s", self.model_size, exc)
            self._model = None
            return False
        finally:
            self._is_loading = False

    def unload_model(self) -> None:
        """Release Whisper model from memory to free ~200-500 MB of system RAM."""
        if self._model is not None:
            logger.info("Unloading Faster-Whisper model from memory.")
            del self._model
            self._model = None

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "auto",
        initial_prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe 16kHz float32 audio chunk to Tamil, English, or Mixed text."""
        if len(audio_data) == 0:
            return TranscriptionResult(text="", language="en", confidence=0.0)

        # Lazy load on first use if not already loaded
        if not self.is_loaded:
            loaded = self.load_model()
            if not loaded:
                return TranscriptionResult(
                    text="[Speech recognition model unavailable]",
                    language="en",
                    confidence=0.0,
                )

        # Ensure float32 format normalized between -1.0 and 1.0
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        duration = len(audio_data) / 16000.0
        logger.debug("Transcribing %.2f seconds of audio with language='%s'...", duration, language)

        # Configure language selection: None signals Whisper to auto-detect
        lang_arg = None if language in ("auto", "mixed", "") else language

        # Prompt conditioning for Tamil-English mixed vocabulary
        conditioning_prompt = initial_prompt or (
            "வணக்கம், Hey HAKS assistant. Tamil and English conversational speech."
        )

        start_t = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                audio_data,
                language=lang_arg,
                initial_prompt=conditioning_prompt,
                beam_size=3,
                best_of=3,
                temperature=0.0,
                vad_filter=False,  # Audio has already been segmented by our VAD
            )

            collected_texts = []
            segment_records = []
            total_prob = 0.0
            seg_count = 0

            for seg in segments:
                text_clean = seg.text.strip()
                if text_clean:
                    collected_texts.append(text_clean)
                    segment_records.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": text_clean,
                        "avg_logprob": seg.avg_logprob,
                        "no_speech_prob": seg.no_speech_prob,
                    })
                    total_prob += np.exp(seg.avg_logprob)
                    seg_count += 1

            full_transcript = " ".join(collected_texts).strip()
            confidence = (total_prob / seg_count) if seg_count > 0 else 0.85
            detected_lang = info.language if hasattr(info, "language") else "en"
            lang_prob = info.language_probability if hasattr(info, "language_probability") else 0.9

            elapsed = time.perf_counter() - start_t
            logger.info(
                "Transcription completed in %.2fs | Lang: %s (%.2f) | Result: '%s'",
                elapsed,
                detected_lang,
                lang_prob,
                full_transcript,
            )

            return TranscriptionResult(
                text=full_transcript,
                language=detected_lang,
                language_probability=float(lang_prob),
                confidence=float(confidence),
                duration_seconds=duration,
                raw_segments=segment_records,
            )
        except Exception as exc:
            logger.error("Error during Faster-Whisper transcription: %s", exc)
            return TranscriptionResult(
                text=f"[Transcription error: {exc}]",
                language="en",
                confidence=0.0,
                duration_seconds=duration,
            )

    def benchmark(self) -> dict:
        """Benchmark STT load and inference on 2 seconds of synthetic audio."""
        if not self.is_loaded:
            load_start = time.perf_counter()
            self.load_model()
            load_time = time.perf_counter() - load_start
        else:
            load_time = 0.0

        dummy_audio = np.random.uniform(-0.1, 0.1, 32000).astype(np.float32)
        infer_start = time.perf_counter()
        res = self.transcribe(dummy_audio, language="en")
        infer_time = time.perf_counter() - infer_start

        return {
            "model_size": self.model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "load_time_seconds": round(load_time, 3),
            "inference_time_seconds": round(infer_time, 3),
            "real_time_factor": round(infer_time / 2.0, 3),
        }
