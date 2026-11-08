"""Intelligent Request Complexity Classifier for HAKS.

Identifies request complexity deterministically without requiring an LLM:
- SIMPLE
- GENERAL_KNOWLEDGE
- PROGRAMMING
- COMPLEX_REASONING
- TECHNICAL_ARCHITECTURE
- CODE_ANALYSIS
- DOCUMENT_ANALYSIS
- RESEARCH
- MULTISTEP_PROBLEM_SOLVING

Works 100% offline with zero external model overhead.
"""

from enum import Enum
import re
from typing import Optional
from pydantic import BaseModel, Field

from core.logger import get_logger

logger = get_logger("routing.complexity")


class ComplexityCategory(str, Enum):
    """Categorization of request reasoning complexity."""

    SIMPLE = "SIMPLE"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    PROGRAMMING = "PROGRAMMING"
    COMPLEX_REASONING = "COMPLEX_REASONING"
    TECHNICAL_ARCHITECTURE = "TECHNICAL_ARCHITECTURE"
    CODE_ANALYSIS = "CODE_ANALYSIS"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    RESEARCH = "RESEARCH"
    MULTISTEP_PROBLEM_SOLVING = "MULTISTEP_PROBLEM_SOLVING"


class ComplexityResult(BaseModel):
    """Detailed result of request complexity classification."""

    category: ComplexityCategory
    is_complex: bool
    estimated_tokens: int
    indicators: list[str] = Field(default_factory=list)
    reason: str


# ------------------------------------------------------------------------------
# Patterns for Complexity Determination
# ------------------------------------------------------------------------------

SIMPLE_GREETINGS = re.compile(r"(?i)^(hi|hello|hey|good morning|good afternoon|good evening|who are you|what can you do|thanks|thank you|bye|goodbye|வணக்கம்)\b")

CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?```")
STACK_TRACE_PATTERN = re.compile(r"(?i)(Traceback \(most recent call last\):|at [a-zA-Z0-9_\.]+\([a-zA-Z0-9_\.]+\.py:\d+\)|NullPointerException|Segmentation fault)")

COMPLEX_REASONING_PATTERNS = [
    ("formal_logic", re.compile(r"(?i)\b(formal proof|prove mathematically|deductive reasoning|inductive proof|first-order logic|counterfactual|game theory|nash equilibrium|np-complete|bayesian inference)\b")),
    ("optimization", re.compile(r"(?i)\b(linear programming|simplex method|dynamic programming formulation|knapsack problem|traveling salesman)\b")),
]

TECHNICAL_ARCHITECTURE_PATTERNS = [
    ("system_design", re.compile(r"(?i)\b(system architecture|technical architecture|microservices? architecture|high-level design|low-level design|hld|lld|distributed (?:systems?|architecture|application)|system design)\b")),
    ("infrastructure", re.compile(r"(?i)\b(kubernetes|event-driven|kafka|load balanc|database sharding|horizontal scaling|cap theorem|cqrs|cloud architecture|microservices?)\b")),
]

RESEARCH_PATTERNS = [
    ("comparative_research", re.compile(r"(?i)\b(comprehensive comparison between|compare and contrast in detail|state of the art in|literature survey|exhaustive trade-off analysis|benchmark study)\b")),
    ("academic_synthesis", re.compile(r"(?i)\b(in-depth research|scientific analysis|meta-analysis|historical evolution of)\b")),
]

MULTISTEP_PATTERNS = [
    ("multi_stage", re.compile(r"(?i)\b(step-by-step implementation plan|end-to-end pipeline|stage 1.*stage 2|multi-phase roadmap|walk me through all the steps|step 1.*step 2|step-by-step)\b")),
]

PROGRAMMING_PATTERNS = [
    ("coding_task", re.compile(r"(?i)\b(write a (?:python|javascript|c\+\+|rust|go|sql|html|css) (?:function|script|class|query)|how do i implement|code snippet for|regular expression for|regex pattern for)\b")),
    ("code_syntax", re.compile(r"(?m)\b(def\s+[a-zA-Z_]\w*\s*\(|class\s+[a-zA-Z_]\w*[:\(]|import\s+[a-zA-Z_]\w*|from\s+[a-zA-Z_]\w*\s+import|function\s+[a-zA-Z_]\w*\s*\(|const\s+[a-zA-Z_]\w*\s*=|async\s+def\s+[a-zA-Z_]\w*)\b")),
]

GENERAL_KNOWLEDGE_PATTERNS = [
    ("fact_lookup", re.compile(r"(?i)\b(what is the capital of|who was the|when did .* happen|define the term|how does photosynthesis work|who discovered|what is the boiling point of)\b")),
]


class ComplexityClassifier:
    """Classifies user queries into complexity categories."""

    def classify(self, text: str) -> ComplexityResult:
        """Evaluate text to determine reasoning complexity."""
        if not text or not text.strip():
            return ComplexityResult(
                category=ComplexityCategory.SIMPLE,
                is_complex=False,
                estimated_tokens=0,
                indicators=["empty_query"],
                reason="Empty query is classified as simple.",
            )

        clean = text.strip()
        words = clean.split()
        estimated_tokens = max(1, int(len(clean) / 4))
        indicators: list[str] = []

        # 1. Code Analysis (StackTrace or Code Blocks)
        if STACK_TRACE_PATTERN.search(clean):
            indicators.append("stack_trace_detected")
            return ComplexityResult(
                category=ComplexityCategory.CODE_ANALYSIS,
                is_complex=True,
                estimated_tokens=estimated_tokens,
                indicators=indicators,
                reason="Stack trace or exception detected requiring in-depth code analysis.",
            )

        if len(CODE_BLOCK_PATTERN.findall(clean)) > 0 or "```" in clean:
            indicators.append("code_block_detected")
            if len(clean) > 400 or clean.count("\n") > 12:
                return ComplexityResult(
                    category=ComplexityCategory.CODE_ANALYSIS,
                    is_complex=True,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="Substantial embedded code block detected requiring deep code analysis.",
                )

        # 2. Complex Reasoning
        for tag, pattern in COMPLEX_REASONING_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                return ComplexityResult(
                    category=ComplexityCategory.COMPLEX_REASONING,
                    is_complex=True,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="Complex mathematical, formal logic, or optimization reasoning detected.",
                )

        # 3. Technical Architecture
        for tag, pattern in TECHNICAL_ARCHITECTURE_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                return ComplexityResult(
                    category=ComplexityCategory.TECHNICAL_ARCHITECTURE,
                    is_complex=True,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="Distributed systems, infrastructure scaling, or technical architecture design detected.",
                )

        # 4. Research Tasks
        for tag, pattern in RESEARCH_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                return ComplexityResult(
                    category=ComplexityCategory.RESEARCH,
                    is_complex=True,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="In-depth comparative survey or research task detected.",
                )

        # 5. Multistep Problem Solving
        for tag, pattern in MULTISTEP_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                return ComplexityResult(
                    category=ComplexityCategory.MULTISTEP_PROBLEM_SOLVING,
                    is_complex=True,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="Multi-stage execution roadmap or step-by-step problem solving detected.",
                )

        # 6. Document Analysis (Long multi-paragraph input)
        if len(clean) > 1200 or clean.count("\n") > 20:
            indicators.append("length_gt_1200_chars")
            return ComplexityResult(
                category=ComplexityCategory.DOCUMENT_ANALYSIS,
                is_complex=True,
                estimated_tokens=estimated_tokens,
                indicators=indicators,
                reason="Long-form document input requiring document synthesis and analysis.",
            )

        # 7. Programming Assistance
        for tag, pattern in PROGRAMMING_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                # Simple coding is not necessarily "heavy complex reasoning" that requires cloud
                return ComplexityResult(
                    category=ComplexityCategory.PROGRAMMING,
                    is_complex=False,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="Standard programming or code implementation query.",
                )

        # 8. Simple Greetings & Quick Inquiries
        if SIMPLE_GREETINGS.match(clean) and len(words) <= 10:
            indicators.append("short_greeting")
            return ComplexityResult(
                category=ComplexityCategory.SIMPLE,
                is_complex=False,
                estimated_tokens=estimated_tokens,
                indicators=indicators,
                reason="Short conversational greeting or basic inquiry.",
            )

        # 9. General Knowledge
        for tag, pattern in GENERAL_KNOWLEDGE_PATTERNS:
            if pattern.search(clean):
                indicators.append(tag)
                return ComplexityResult(
                    category=ComplexityCategory.GENERAL_KNOWLEDGE,
                    is_complex=False,
                    estimated_tokens=estimated_tokens,
                    indicators=indicators,
                    reason="General knowledge or factual lookup.",
                )

        # 10. Default heuristic by word count
        if len(words) <= 15 and clean.count("\n") <= 1:
            return ComplexityResult(
                category=ComplexityCategory.SIMPLE,
                is_complex=False,
                estimated_tokens=estimated_tokens,
                indicators=["low_word_count"],
                reason="Concise query classified as simple.",
            )

        return ComplexityResult(
            category=ComplexityCategory.GENERAL_KNOWLEDGE,
            is_complex=False,
            estimated_tokens=estimated_tokens,
            indicators=["default_general"],
            reason="Standard general query.",
        )


_complexity_classifier_instance: Optional[ComplexityClassifier] = None


def get_complexity_classifier() -> ComplexityClassifier:
    """Return singleton ComplexityClassifier instance."""
    global _complexity_classifier_instance
    if _complexity_classifier_instance is None:
        _complexity_classifier_instance = ComplexityClassifier()
    return _complexity_classifier_instance
