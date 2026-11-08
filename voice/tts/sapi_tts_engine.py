"""Windows SAPI Desktop Text-to-Speech Engine for HAKS.

Uses pyttsx3 to access native Windows SAPI voices 100% offline.
Enumerates installed Windows desktop voices and provides thread-safe synthesis to WAV.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from voice.models import TTSVoiceInfo
from voice.tts.base import TextToSpeechInterface

logger = logging.getLogger("haks.voice.tts.sapi")

_shared_sapi_executor: Optional[ThreadPoolExecutor] = None
_cached_sapi_voices: Optional[list[TTSVoiceInfo]] = None
_sapi_has_tamil: bool = False


def _get_sapi_executor() -> ThreadPoolExecutor:
    global _shared_sapi_executor
    if _shared_sapi_executor is None:
        _shared_sapi_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sapi_tts")
    return _shared_sapi_executor


def shutdown_sapi_executor() -> None:
    global _shared_sapi_executor
    if _shared_sapi_executor is not None:
        _shared_sapi_executor.shutdown(wait=False, cancel_futures=True)
        _shared_sapi_executor = None


class SapiTTSEngine(TextToSpeechInterface):
    """Offline Windows SAPI voice synthesis engine."""

    def __init__(self) -> None:
        self._executor = _get_sapi_executor()
        self._cached_voices: list[TTSVoiceInfo] = []
        self._has_tamil_voice: bool = False
        self._refresh_voices()

    @property
    def name(self) -> str:
        return "Windows SAPI (pyttsx3)"

    def is_available(self) -> bool:
        return pyttsx3 is not None

    def supports_tamil(self) -> bool:
        return self._has_tamil_voice

    def supports_english(self) -> bool:
        return True

    def _refresh_voices(self) -> None:
        """Scan system for installed Windows SAPI voices."""
        global _cached_sapi_voices, _sapi_has_tamil
        if pyttsx3 is None:
            return

        if _cached_sapi_voices is not None:
            self._cached_voices = _cached_sapi_voices
            self._has_tamil_voice = _sapi_has_tamil
            return

        def _scan():
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                v_list = []
                has_tamil = False
                for v in voices:
                    lang = "en-US"
                    if hasattr(v, "languages") and v.languages:
                        lang = v.languages[0]
                    v_name_lower = v.name.lower()
                    if "tamil" in v_name_lower or "ta-in" in str(lang).lower():
                        has_tamil = True
                        lang = "ta-IN"

                    v_list.append(
                        TTSVoiceInfo(
                            id=v.id,
                            name=v.name,
                            language=lang,
                            gender="Female" if "zira" in v_name_lower or "female" in v_name_lower else "Male",
                            engine="sapi",
                        )
                    )
                engine.stop()
                return v_list, has_tamil
            except Exception as exc:
                logger.warning("Could not query Windows SAPI voices: %s", exc)
                return [], False
            finally:
                try:
                    if "engine" in locals() and engine is not None:
                        del engine
                except Exception:
                    pass

        try:
            future = self._executor.submit(_scan)
            voices, has_tamil = future.result(timeout=3.0)
            _cached_sapi_voices = voices
            _sapi_has_tamil = has_tamil
            self._cached_voices = voices
            self._has_tamil_voice = has_tamil
        except Exception as exc:
            logger.warning("Error querying SAPI voices: %s", exc)

    def shutdown(self) -> None:
        """Release background executor resources."""
        pass

    def list_voices(self) -> list[TTSVoiceInfo]:
        return list(self._cached_voices)

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> Path:
        """Synthesize text to WAV file using Windows SAPI inside a dedicated worker thread."""
        if not self.is_available():
            raise RuntimeError("pyttsx3 is not available.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure path ends with .wav for pyttsx3
        wav_path = output_path.with_suffix(".wav")

        def _run_sapi():
            # pyttsx3 COM engine must be initialized in the worker thread
            engine = pyttsx3.init()
            try:
                if voice:
                    engine.setProperty("voice", voice)

                # Base speed ~200 words/min
                base_rate = engine.getProperty("rate") or 200
                engine.setProperty("rate", int(base_rate * speed))

                # Volume 0.0 to 1.0
                engine.setProperty("volume", max(0.0, min(1.0, volume)))

                engine.save_to_file(clean_text, str(wav_path))
                engine.runAndWait()
            finally:
                try:
                    engine.stop()
                except Exception:
                    pass
                try:
                    del engine
                except Exception:
                    pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _run_sapi)

        logger.info("SAPI audio synthesized to: %s", wav_path)
        return wav_path
