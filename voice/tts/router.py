"""Language-Aware Text-to-Speech Router for HAKS.

Coordinates between Edge Neural TTS, Windows SAPI, and Piper engines.
Detects Tamil script vs English, routes to the appropriate voice, and
provides graceful offline fallback without silent failures.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Optional
import uuid

from config.settings import AppConfig, get_settings
from voice.models import SpokenUtterance, TTSVoiceInfo
from voice.tts.base import TextToSpeechInterface
from voice.tts.edge_tts_engine import EdgeTTSEngine
from voice.tts.piper_tts_engine import PiperTTSEngine
from voice.tts.sapi_tts_engine import SapiTTSEngine

logger = logging.getLogger("haks.voice.tts.router")


class LanguageAwareTTSRouter:
    """Manages bilingual voice synthesis and engine selection."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_settings()
        self.edge_engine = EdgeTTSEngine()
        self.sapi_engine = SapiTTSEngine()
        self.piper_engine = PiperTTSEngine()

        self._output_cache_dir: Path = self.config.paths.data_dir / "tts_cache"
        self._output_cache_dir.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        """Release TTS engine resources."""
        if hasattr(self, "sapi_engine") and self.sapi_engine is not None:
            self.sapi_engine.shutdown()

    @staticmethod
    def contains_tamil(text: str) -> bool:
        """Return True if text contains Tamil script characters."""
        return bool(re.search(r"[\u0B80-\u0BFF]", text))

    def detect_language(self, text: str) -> str:
        """Classify text language as 'ta' (Tamil), 'en' (English), or 'mixed'."""
        has_tamil = self.contains_tamil(text)
        has_ascii = bool(re.search(r"[a-zA-Z]", text))
        if has_tamil and has_ascii:
            return "mixed"
        elif has_tamil:
            return "ta"
        return "en"

    def resolve_voice(self, text: str) -> str:
        """Resolve the appropriate voice name based on language detection."""
        lang = self.detect_language(text)
        if lang in ("ta", "mixed"):
            return self.config.voice.tts_tamil_voice
        return self.config.voice.tts_english_voice

    def get_all_available_voices(self) -> list[TTSVoiceInfo]:
        """Aggregate available voices across all engines."""
        voices = []
        if self.edge_engine.is_available():
            voices.extend(self.edge_engine.list_voices())
        if self.sapi_engine.is_available():
            voices.extend(self.sapi_engine.list_voices())
        if self.piper_engine.is_available():
            voices.extend(self.piper_engine.list_voices())
        return voices

    @staticmethod
    def is_engine_offline(engine_name: str) -> bool:
        """Explicitly distinguish offline TTS engines from internet-dependent engines."""
        return engine_name.lower() in ("sapi", "sapi5", "piper", "pyttsx3")

    @staticmethod
    def get_engine_network_requirement(engine_name: str) -> str:
        """Return human-readable network requirement description."""
        if engine_name.lower() in ("edge", "edge_tts"):
            return "Online Only (Microsoft Edge Cloud Neural TTS)"
        elif engine_name.lower() in ("sapi", "sapi5"):
            return "100% Offline (Windows SAPI5 Desktop Speech)"
        elif engine_name.lower() == "piper":
            return "100% Offline (Local ONNX Piper Engine)"
        return "Unknown"

    async def synthesize(
        self,
        text: str,
        engine_preference: Optional[str] = None,
        voice_override: Optional[str] = None,
        speed: Optional[float] = None,
        volume: Optional[float] = None,
        is_offline: bool = False,
    ) -> SpokenUtterance:
        """Synthesize text using the best engine for the language and configuration."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        engine_type = engine_preference or self.config.voice.tts_engine
        detected_lang = self.detect_language(clean_text)
        eff_speed = speed if speed is not None else self.config.voice.speech_speed
        eff_volume = volume if volume is not None else self.config.voice.speech_volume

        file_id = f"tts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_path = self._output_cache_dir / f"{file_id}.mp3"

        # 1. User explicitly selected Piper
        if engine_type == "piper" and self.piper_engine.is_available():
            try:
                res_path = await self.piper_engine.synthesize_to_file(
                    clean_text, out_path, voice=voice_override, speed=eff_speed, volume=eff_volume
                )
                return SpokenUtterance(
                    text=clean_text,
                    detected_language=detected_lang,
                    voice_name=voice_override or "Piper",
                    audio_path=str(res_path),
                )
            except Exception as exc:
                logger.warning("Piper TTS synthesis failed: %s. Falling back to Edge/SAPI.", exc)

        # 2. User explicitly selected SAPI or offline fallback
        if engine_type == "sapi" and self.sapi_engine.is_available():
            res_path = await self.sapi_engine.synthesize_to_file(
                clean_text, out_path, voice=voice_override, speed=eff_speed, volume=eff_volume
            )
            return SpokenUtterance(
                text=clean_text,
                detected_language=detected_lang,
                voice_name=voice_override or "Windows SAPI",
                audio_path=str(res_path),
            )

        # 3. Default / Auto: Prefer Edge Neural TTS only when NOT offline
        if not is_offline and self.edge_engine.is_available():
            try:
                # Select appropriate voice based on Tamil presence
                voice = voice_override
                if not voice:
                    if detected_lang in ("ta", "mixed"):
                        voice = self.config.voice.tts_tamil_voice
                    else:
                        voice = self.config.voice.tts_english_voice

                res_path = await self.edge_engine.synthesize_to_file(
                    clean_text, out_path, voice=voice, speed=eff_speed, volume=eff_volume
                )
                return SpokenUtterance(
                    text=clean_text,
                    detected_language=detected_lang,
                    voice_name=voice or "Edge-TTS",
                    audio_path=str(res_path),
                )
            except Exception as exc:
                logger.warning(
                    "Edge-TTS synthesis error (likely network unreachable): %s. Falling back to Windows SAPI.",
                    exc,
                )

        # 4. Fallback to Windows SAPI
        if self.sapi_engine.is_available():
            if detected_lang in ("ta", "mixed") and not self.sapi_engine.supports_tamil():
                logger.warning(
                    "Tamil text encountered but Windows SAPI has no Tamil desktop voice installed. "
                    "Synthesizing via default English voice."
                )

            res_path = await self.sapi_engine.synthesize_to_file(
                clean_text, out_path, speed=eff_speed, volume=eff_volume
            )
            return SpokenUtterance(
                text=clean_text,
                detected_language=detected_lang,
                voice_name="Windows SAPI (Offline Fallback)",
                audio_path=str(res_path),
            )

        raise RuntimeError("No functional Text-to-Speech engine is available on the system.")
