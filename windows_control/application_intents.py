"""Application Intent Recognition Engine for HAKS Windows Computer Control (Phase 9.1).

Converts recognized natural-language instructions in English, Tamil, and Tanglish
into structured, policy-governed application actions.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Optional

from windows_control.application_registry import ApplicationEntry, ApplicationRegistry

logger = logging.getLogger("haks.windows_control.application_intents")


class AppActionType(str, Enum):
    """Supported application lifecycle and window management actions."""

    LAUNCH = "launch_app"
    SWITCH = "switch_app"
    MINIMIZE = "minimize_app"
    MAXIMIZE = "maximize_app"
    RESTORE = "restore_app"
    CLOSE = "close_app"
    UNKNOWN = "unknown"


@dataclass
class ApplicationIntent:
    """Structured representation of an understood application control request."""

    action: AppActionType
    app: Optional[ApplicationEntry]
    raw_app_name: str
    raw_text: str
    language: str = "en"  # en, ta, tanglish
    confidence: float = 0.0
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return True if intent is confident, recognized, and matched to an application."""
        return self.action != AppActionType.UNKNOWN and self.app is not None and self.confidence >= 0.60


class ApplicationIntentParser:
    """Parses natural-language voice instructions into typed ApplicationIntent structures."""

    def __init__(self, registry: Optional[ApplicationRegistry] = None) -> None:
        self.registry = registry or ApplicationRegistry()

    def parse(self, text: str) -> ApplicationIntent:
        """Parse an instruction in English, Tamil, or Tanglish into an ApplicationIntent."""
        raw = text.strip()
        if not raw:
            return self._unknown_intent(raw)

        # 0. Strip wake words / prefixes (e.g. "Hey HAKS, ...", "HAKS ...")
        clean = re.sub(r"^(?:hey|hi|hello)?\s*haks[\s,:\-]+", "", raw, flags=re.IGNORECASE).strip()
        lower = clean.lower()

        # Reject conversational queries / questions (e.g. "What is Chrome?", "How does Notepad work?")
        question_words = (
            "what is", "what are", "who is", "who are", "tell me", "explain",
            "how do", "how to", "why is", "why does", "when is", "யார்", "என்ன", "எப்படி"
        )
        if any(lower.startswith(qw) for qw in question_words) or lower.endswith("?"):
            return self._unknown_intent(raw)

        # Detect primary language mode
        has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", clean))
        has_tanglish_verbs = bool(re.search(r"\b(pannu|panunga|pannunga|moodu|maathu|ready\s+pannu)\b", lower))
        lang = "ta" if has_tamil and not has_tanglish_verbs else ("tanglish" if has_tanglish_verbs else "en")

        # 1. Specialized Intent Pattern Checkers
        # --- CLOSE ACTION ---
        close_intent = self._check_close_intent(clean, lower, lang)
        if close_intent.is_actionable:
            return close_intent

        # --- MINIMIZE ACTION ---
        min_intent = self._check_minimize_intent(clean, lower, lang)
        if min_intent.is_actionable:
            return min_intent

        # --- MAXIMIZE ACTION ---
        max_intent = self._check_maximize_intent(clean, lower, lang)
        if max_intent.is_actionable:
            return max_intent

        # --- RESTORE ACTION ---
        res_intent = self._check_restore_intent(clean, lower, lang)
        if res_intent.is_actionable:
            return res_intent

        # --- SWITCH ACTION ---
        switch_intent = self._check_switch_intent(clean, lower, lang)
        if switch_intent.is_actionable:
            return switch_intent

        # --- LAUNCH / OPEN ACTION ---
        launch_intent = self._check_launch_intent(clean, lower, lang)
        if launch_intent.is_actionable:
            return launch_intent

        return self._unknown_intent(raw)

    # --------------------------------------------------------------------------
    # Individual Action Checkers
    # --------------------------------------------------------------------------

    def _check_close_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for application closure intent (graceful window close)."""
        # Tamil patterns: "<app> மூடு", "<app>-ஐ மூடு"
        # Tanglish: "<app> close pannu", "<app> ah close pannu", "<app> moodu"
        # English: "close <app>", "exit <app>", "quit <app>"

        patterns = [
            # Tanglish / Tamil verbs
            (r"(.*?)\s+(?:ah\s+)?close\s+(?:pannu|panunga|pannunga)", "tanglish"),
            (r"(.*?)\s+(?:ah\s+)?moodu\b", "tanglish"),
            (r"(.*?)(?:-?ஐ|-?ai)?\s*மூடு\b", "ta"),
            # English verbs
            (r"\b(?:close|quit|exit)\s+(?:the\s+|app\s+|application\s+)?(.*)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.CLOSE,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                        requires_confirmation=False,  # Phase 9.1 uses graceful WM_CLOSE protecting unsaved work
                    )
        return self._unknown_intent(text)

    def _check_minimize_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for minimize intent."""
        patterns = [
            (r"(.*?)\s+(?:ah\s+)?minimize\s+(?:pannu|panunga|pannunga)", "tanglish"),
            (r"(.*?)\s*சுருக்கு\b", "ta"),
            (r"(.*?)\s*மறை\b", "ta"),
            (r"\b(?:minimize|hide)\s+(?:the\s+|app\s+|application\s+)?(.*)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.MINIMIZE,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                    )
        return self._unknown_intent(text)

    def _check_maximize_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for maximize intent."""
        patterns = [
            (r"(.*?)\s+(?:ah\s+)?maximize\s+(?:pannu|panunga|pannunga)", "tanglish"),
            (r"(.*?)\s*பெரிதாக்கு\b", "ta"),
            (r"\b(?:maximize|fullscreen)\s+(?:the\s+|app\s+|application\s+)?(.*)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.MAXIMIZE,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                    )
        return self._unknown_intent(text)

    def _check_restore_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for restore / unminimize intent."""
        patterns = [
            (r"(.*?)\s+(?:ah\s+)?restore\s+(?:pannu|panunga|pannunga)", "tanglish"),
            (r"(.*?)\s*மீட்டமை\b", "ta"),
            (r"\b(?:restore|unminimize)\s+(?:the\s+|app\s+|application\s+)?(.*)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.RESTORE,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                    )
        return self._unknown_intent(text)

    def _check_switch_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for window switching intent."""
        patterns = [
            # Tanglish / Tamil: "<app>-ku switch pannu", "<app> ku switch pannu"
            (r"(.*?)(?:-?ku|-?க்கு)?\s+switch\s+(?:pannu|panunga|pannunga)", "tanglish"),
            (r"(.*?)(?:-?ku|-?க்கு)?\s+மாற்று\b", "ta"),
            (r"(.*?)\s*காட்டு\b", "ta"),
            # English: "switch to <app>", "bring up <app>", "focus <app>", "go to <app>", "show <app>"
            (r"\b(?:switch\s+to|bring\s+up|focus|go\s+to|show)\s+(?:the\s+|app\s+|application\s+)?(.*)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.SWITCH,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                    )
        return self._unknown_intent(text)

    def _check_launch_intent(self, text: str, lower: str, lang: str) -> ApplicationIntent:
        """Check for application launching intent."""
        # 1. Specialized Tanglish: e.g. "Chrome la search panna ready pannu"
        search_ready_m = re.search(r"([a-zA-Z0-9_\-\s]+)\s+la\s+search\s+panna\s+ready\s+pannu", lower)
        if search_ready_m:
            raw_target = search_ready_m.group(1).strip()
            app = self.registry.resolve_alias(raw_target)
            if app:
                return ApplicationIntent(
                    action=AppActionType.LAUNCH,
                    app=app,
                    raw_app_name=raw_target,
                    raw_text=text,
                    language="tanglish",
                    confidence=0.95,
                )

        # 2. General Launch Patterns
        patterns = [
            # Tanglish: "<app> (ah)? open pannu", "<app> start pannu", "<app> launch pannu"
            (r"(.*?)\s+(?:ah\s+)?(?:open|start|launch)\s+(?:pannu|panunga|pannunga)", "tanglish"),
            # Tamil: "<app> திற", "<app> துவக்கு", "<app> தொடங்கு", "<app>-ஐ திற"
            (r"([a-zA-Z0-9_\-\u0B80-\u0BFF\s]+?)(?:-?ஐ|-?ai)?\s*(?:திற|துவக்கு|தொடங்கு|இயக்கு)\b", "ta"),
            # English: "open <app>", "launch <app>", "start <app>", "run <app>"
            (r"\b(?:open|launch|start|run)\s+(?:the\s+|app\s+|application\s+)?([a-zA-Z0-9_\-\s]+)", "en"),
        ]

        for pat, p_lang in patterns:
            m = re.search(pat, text if p_lang == "ta" else lower, re.IGNORECASE)
            if m:
                raw_target = m.group(1).strip()
                # Exclude non-app noise (e.g. "open a file", "open folder", "pip install")
                if any(k in raw_target.lower() for k in ("pip", "file", "folder", "dependency", "dependencies", "repo", "git")):
                    continue

                app = self.registry.resolve_alias(raw_target)
                if app:
                    return ApplicationIntent(
                        action=AppActionType.LAUNCH,
                        app=app,
                        raw_app_name=raw_target,
                        raw_text=text,
                        language=p_lang if p_lang != "en" else lang,
                        confidence=0.95,
                    )

        return self._unknown_intent(text)

    def _unknown_intent(self, text: str) -> ApplicationIntent:
        """Fallback empty or unrecognized intent."""
        return ApplicationIntent(
            action=AppActionType.UNKNOWN,
            app=None,
            raw_app_name="",
            raw_text=text,
            confidence=0.0,
        )

