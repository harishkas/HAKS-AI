"""Microsoft Edge Neural Text-to-Speech Engine for HAKS.

Provides natural, authentic Tamil (ta-IN-PallaviNeural, ta-IN-ValluvarNeural)
and English (en-IN-NeerjaNeural, en-US-JennyNeural) neural speech synthesis.
"""

import asyncio
import logging
from pathlib import Path
import re
from typing import Optional

try:
    import edge_tts
except ImportError:
    edge_tts = None

from voice.models import TTSVoiceInfo
from voice.tts.base import TextToSpeechInterface

logger = logging.getLogger("haks.voice.tts.edge")

# Verified authentic Tamil and English voice definitions
EDGE_VOICES = [
    TTSVoiceInfo(id="ta-IN-PallaviNeural", name="Pallavi (Tamil India)", language="ta-IN", gender="Female", engine="edge"),
    TTSVoiceInfo(id="ta-IN-ValluvarNeural", name="Valluvar (Tamil India)", language="ta-IN", gender="Male", engine="edge"),
    TTSVoiceInfo(id="ta-LK-KumarNeural", name="Kumar (Tamil Sri Lanka)", language="ta-LK", gender="Male", engine="edge"),
    TTSVoiceInfo(id="ta-LK-SaranyaNeural", name="Saranya (Tamil Sri Lanka)", language="ta-LK", gender="Female", engine="edge"),
    TTSVoiceInfo(id="en-IN-NeerjaNeural", name="Neerja (English India)", language="en-IN", gender="Female", engine="edge"),
    TTSVoiceInfo(id="en-IN-PrabhatNeural", name="Prabhat (English India)", language="en-IN", gender="Male", engine="edge"),
    TTSVoiceInfo(id="en-US-JennyNeural", name="Jenny (English US)", language="en-US", gender="Female", engine="edge"),
    TTSVoiceInfo(id="en-US-GuyNeural", name="Guy (English US)", language="en-US", gender="Male", engine="edge"),
]


class EdgeTTSEngine(TextToSpeechInterface):
    """Bilingual Edge-TTS Neural Voice Engine."""

    def __init__(self) -> None:
        self.default_tamil_voice = "ta-IN-PallaviNeural"
        self.default_english_voice = "en-IN-NeerjaNeural"

    @property
    def name(self) -> str:
        return "Microsoft Edge Neural TTS"

    def is_available(self) -> bool:
        return edge_tts is not None

    def supports_tamil(self) -> bool:
        return True

    def supports_english(self) -> bool:
        return True

    def list_voices(self) -> list[TTSVoiceInfo]:
        return list(EDGE_VOICES)

    @staticmethod
    def contains_tamil(text: str) -> bool:
        """Check if string contains Tamil Unicode range (U+0B80 to U+0BFF)."""
        return bool(re.search(r"[\u0B80-\u0BFF]", text))

    def resolve_voice_for_text(self, text: str, requested_voice: Optional[str] = None) -> str:
        """Select voice based on explicit request or script detection."""
        if requested_voice and requested_voice != "auto":
            return requested_voice

        if self.contains_tamil(text):
            return self.default_tamil_voice
        return self.default_english_voice

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        volume: float = 1.0,
    ) -> Path:
        """Synthesize text and write MP3 audio to output_path."""
        if not self.is_available():
            raise RuntimeError("edge_tts package is not installed.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        selected_voice = self.resolve_voice_for_text(clean_text, voice)

        # Convert speed multiplier (e.g. 1.15) to rate string ("+15%")
        rate_percent = int(round((speed - 1.0) * 100))
        rate_str = f"{rate_percent:+d}%"

        # Convert volume multiplier (e.g. 0.8) to volume string ("-20%")
        vol_percent = int(round((volume - 1.0) * 100))
        vol_str = f"{vol_percent:+d}%"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Synthesizing %d characters via Edge-TTS (voice='%s', rate='%s', vol='%s')...",
            len(clean_text),
            selected_voice,
            rate_str,
            vol_str,
        )

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=selected_voice,
            rate=rate_str,
            volume=vol_str,
        )

        await communicate.save(str(output_path))
        logger.info("Synthesized audio saved to: %s", output_path)
        return output_path
