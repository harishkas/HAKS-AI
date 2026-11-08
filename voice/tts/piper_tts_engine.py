"""Piper Local ONNX Text-to-Speech Engine Adapter for HAKS.

Provides an architecture-ready adapter for local Piper ONNX voice models.
Handles model loading and synthesis when community Tamil/English ONNX models are installed.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Optional

from voice.models import TTSVoiceInfo
from voice.tts.base import TextToSpeechInterface

logger = logging.getLogger("haks.voice.tts.piper")


class PiperTTSEngine(TextToSpeechInterface):
    """Local Piper ONNX voice synthesis engine."""

    def __init__(self, model_path: Optional[Path] = None, config_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper_tts")
        self._piper_voice: Optional[Any] = None

    @property
    def name(self) -> str:
        return "Piper Local TTS (ONNX)"

    def is_available(self) -> bool:
        if self.model_path is None or not Path(self.model_path).exists():
            return False
        return True

    def supports_tamil(self) -> bool:
        if not self.is_available():
            return False
        return "ta" in str(self.model_path).lower()

    def supports_english(self) -> bool:
        if not self.is_available():
            return False
        return "en" in str(self.model_path).lower()

    def get_status_info(self) -> dict:
        """Detailed diagnostic information on Piper availability."""
        return {
            "is_configured": self.is_available(),
            "model_path": str(self.model_path) if self.model_path else "Not configured",
            "notes": (
                "Piper requires a community-contributed Tamil ONNX model (such as Jeyaram-K/piper-tamil-voices "
                "or tinisoft/piper-ta_IN-rasa_female-medium). Standard Piper distribution does not bundle official Tamil voices."
            ),
        }

    def list_voices(self) -> list[TTSVoiceInfo]:
        if not self.is_available():
            return []
        model_name = self.model_path.stem if self.model_path else "Piper Voice"
        return [
            TTSVoiceInfo(
                id=str(self.model_path),
                name=f"Piper Local ({model_name})",
                language="ta-IN" if "ta" in model_name.lower() else "en-US",
                gender="Female" if "female" in model_name.lower() else "Male",
                engine="piper",
            )
        ]

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> Path:
        """Synthesize text using local Piper ONNX voice."""
        if not self.is_available():
            raise RuntimeError(
                "Piper voice model is not configured. Download a Piper .onnx model and configure the path, "
                "or use the default Microsoft Edge Neural TTS."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = output_path.with_suffix(".wav")

        def _synthesize():
            try:
                from piper import PiperVoice
                import wave

                if self._piper_voice is None:
                    self._piper_voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path) if self.config_path else None)

                with wave.open(str(wav_path), "wb") as wav_file:
                    self._piper_voice.synthesize(text, wav_file)
            except ImportError:
                raise RuntimeError("piper-tts package is not installed.")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _synthesize)
        return wav_path
