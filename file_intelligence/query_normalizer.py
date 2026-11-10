"""Multilingual Query Normalization Engine for HAKS File Intelligence (Phase 9.2).

Normalizes natural language search queries across English, Tamil script, Romanized Tamil,
and mixed-language Tanglish, mapping colloquial variations and expanding domain terms.
"""

from dataclasses import dataclass
import logging
import re
from typing import Optional

from file_intelligence.models import SearchMode, SearchQuery

logger = logging.getLogger("haks.file_intelligence.normalizer")

# Tamil script vocabulary mapping
TAMIL_SCRIPT_MAP = {
    "கோப்பு": "file",
    "கோப்புகள்": "files",
    "ஆவணம்": "document",
    "ஆவணங்கள்": "documents",
    "கோப்புறை": "folder",
    "தேடு": "search",
    "காட்டு": "show",
    "பட்டியல்": "list",
    "அறிக்கை": "report",
    "சுருக்கம்": "summary",
    "வரவுசெலவு": "budget",
    "விலைப்பட்டியல்": "invoice",
    "ரசீது": "receipt",
    "திட்டம்": "project",
    "குறியீடு": "code",
}

# Romanized Tamil / Tanglish vocabulary mapping
TANGLISH_VOCAB_MAP = {
    "thedu": "search",
    "thedunga": "search",
    "thedi": "search",
    "kaatu": "show",
    "kaatunga": "show",
    "paaru": "view",
    "solli": "tell",
    "sollu": "tell",
    "engay": "where",
    "engae": "where",
    "enga": "where",
    "irukku": "exists",
    "enna": "what",
    "koppu": "file",
    "koppugal": "files",
    "aavanam": "document",
    "kollai": "duplicate",
    "porul": "content",
    "periya": "large",
    "chinna": "small",
    "puthiya": "new",
    "pazhaya": "old",
}

# Domain vocabulary expansion (synonyms / multilingual tags)
DOMAIN_EXPANSIONS = {
    "budget": ["budget", "finance", "expense", "cost", "வரவுசெலவு"],
    "invoice": ["invoice", "bill", "receipt", "payment", "ரசீது"],
    "report": ["report", "analysis", "summary", "audit", "அறிக்கை"],
    "meeting": ["meeting", "minutes", "agenda", "notes", "கூட்டம்"],
    "code": ["code", "script", "program", "source", "குறியீடு"],
    "test": ["test", "verification", "spec", "benchmark", "சோதனை"],
}


class QueryNormalizer:
    """Normalizes conversational queries in English, Tamil, and Tanglish."""

    def normalize(self, raw_query: str, search_mode: Optional[SearchMode] = None) -> SearchQuery:
        """Process raw query into structured SearchQuery with detected language and normalized form."""
        text = raw_query.strip()
        if not text:
            return SearchQuery(raw_query="", normalized_query="", language="en")

        # 1. Detect language
        has_tamil_script = bool(re.search(r"[\u0B80-\u0BFF]", text))
        lower_words = text.lower().split()
        tanglish_words = [w for w in lower_words if w in TANGLISH_VOCAB_MAP]
        has_tanglish_markers = bool(tanglish_words) or bool(re.search(r"\b(la|oda|ku|ah|nu|pannu)\b", text.lower()))

        if has_tamil_script and not has_tanglish_markers:
            lang = "ta"
        elif has_tanglish_markers or (has_tamil_script and has_tanglish_markers):
            lang = "tanglish"
        elif any(c for c in text if ord(c) > 127):
            lang = "mixed"
        else:
            lang = "en"

        # 2. Strip conversational command noise (e.g., "HAKS, search for...", "can you find...")
        clean = re.sub(
            r"^(?:hey|hi|hello)?\s*haks[\s,:\-]+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        leading_noise = (
            r"^(?:search\s+for|find\s+me|find|look\s+for|show\s+me|list|where\s+is|where\s+are|"
            r"can\s+you\s+find|please\s+find|தேடு|காட்டு|சொல்லு)\s+"
        )
        stripped = re.sub(leading_noise, "", clean, flags=re.IGNORECASE).strip()
        target_text = stripped or clean

        # 3. Clean Tanglish particles ("-la", " la", "-ku", "-oda", " ah")
        normalized_words = []
        for word in target_text.split():
            w_lower = word.lower()
            # Strip trailing particle
            w_clean = re.sub(r"(-?(?:la|oda|ku|ah|nu|ai|இல்|ல்|க்கு))+$", "", w_lower)
            # Map Tanglish vocabulary
            w_mapped = TANGLISH_VOCAB_MAP.get(w_clean, TAMIL_SCRIPT_MAP.get(word, w_clean))
            normalized_words.append(w_mapped)

        normalized_query = " ".join(normalized_words).strip()

        # 4. Extract explicit file extension filters if present (e.g. "search *.pdf", "find docx files")
        file_types = []
        ext_matches = re.findall(r"\b(?:\*\.)?([a-zA-Z0-9]{2,5})\s+(?:files?|documents?|கோப்புகள்)\b", clean, re.IGNORECASE)
        for ext in ext_matches:
            ext_clean = ext.lower().replace(".", "")
            if ext_clean in ("pdf", "docx", "xlsx", "pptx", "txt", "md", "csv", "html", "py"):
                file_types.append(f".{ext_clean}")

        # Explicit dot extension (e.g. "*.pdf" or ".docx")
        direct_exts = re.findall(r"\b\.([a-zA-Z0-9]{2,5})\b", clean)
        for ext in direct_exts:
            if f".{ext.lower()}" not in file_types:
                file_types.append(f".{ext.lower()}")

        mode = search_mode or SearchMode.HYBRID

        return SearchQuery(
            raw_query=raw_query,
            normalized_query=normalized_query or clean,
            language=lang,
            search_mode=mode,
            file_types=file_types if file_types else None,
        )

    def expand_domain_terms(self, query: str) -> list[str]:
        """Expand domain keywords with multilingual synonyms."""
        words = query.lower().split()
        expanded: list[str] = []
        for w in words:
            expanded.append(w)
            for key, syns in DOMAIN_EXPANSIONS.items():
                if w == key or w in syns:
                    for s in syns:
                        if s not in expanded:
                            expanded.append(s)
        return expanded

