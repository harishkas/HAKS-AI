"""Multilingual Benchmark Dataset & Retrieval Evaluator for HAKS File Intelligence (Phase 9.2).

Evaluates BGE-M3 + Hybrid Retrieval (Dense + Sparse + BM25/FTS5 + Windows Search)
across 100 English, 100 Tamil, and 100 Tanglish queries.

Metrics:
- Recall@5
- Recall@10
- MRR@10
- nDCG@10
- Precision@5
- Latency (p50, p95, p99)
"""

import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.settings import AppConfig
from file_intelligence.models import (
    DocumentChunk,
    EvaluationMetrics,
    FileMetadata,
    SearchMode,
    SearchQuery,
    SearchResultItem,
)
from file_intelligence.service import FileIntelligenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("haks.benchmark")

# ------------------------------------------------------------------------------
# 1. Representative Multilingual Corpus Definitions
# ------------------------------------------------------------------------------

CORPUS_DOCUMENTS = [
    # Technical & Architecture
    {
        "id": "tech_arch_01",
        "file_name": "haks_voice_architecture_spec.md",
        "content": "HAKS Voice Assistant Architecture uses WebRTC APM for acoustic echo cancellation and Silero VAD for real-time speech detection. Barge-in interruption triggers immediate playback stop in PCMPlayer.",
        "category": "technical",
    },
    {
        "id": "tech_db_02",
        "file_name": "sqlite_fts5_indexing_guide.txt",
        "content": "SQLite FTS5 full-text search provides lexical BM25 ranking over document tokens with porter stemming and trigram indexing. Fast query execution under 5 milliseconds on CPU.",
        "category": "technical",
    },
    {
        "id": "tech_docker_03",
        "file_name": "docker_deployment_containers.md",
        "content": "Container deployment scripts for local microservices using Docker Compose. Configures GPU passthrough, volume mounting, and persistent caching for AI model weights.",
        "category": "technical",
    },
    {
        "id": "tech_api_04",
        "file_name": "gemini_live_websocket_api.py",
        "content": "WebSocket protocol implementation for Gemini Live bidirectional audio streaming. Handles real-time PCM16 24kHz audio capture, server VAD interrupts, and turn completion.",
        "category": "technical",
    },
    {
        "id": "tech_faiss_05",
        "file_name": "faiss_vector_similarity_benchmark.txt",
        "content": "FAISS IndexFlatIP cosine similarity search over 1024-dimensional BGE-M3 embeddings. Benchmarking batch latency, memory footprint, and top-k search accuracy on Intel i5 CPU.",
        "category": "technical",
    },
    # Financial & Spreadsheets
    {
        "id": "fin_budget_01",
        "file_name": "annual_budget_forecast_2026.csv",
        "content": "Department,Q1_Budget,Q2_Budget,Q3_Budget,Q4_Budget,Total_USD\nEngineering,150000,160000,170000,180000,660000\nMarketing,45000,50000,52000,55000,202000\nOperations,30000,32000,33000,35000,130000",
        "category": "financial",
    },
    {
        "id": "fin_invoice_02",
        "file_name": "vendor_hardware_invoice_oct2026.txt",
        "content": "Vendor Hardware Invoice #INV-2026-992. Billing for Dell OptiPlex workstations, SSD upgrades, 16GB RAM modules, and Logitech headsets. Total balance payable: 4,850 USD.",
        "category": "financial",
    },
    {
        "id": "fin_expense_03",
        "file_name": "travel_expense_report_chennai.csv",
        "content": "Date,Employee,Description,City,Amount_INR\n2026-10-15,Venkatesh,Flight Bangalore to Chennai,Chennai,4500\n2026-10-16,Venkatesh,Hotel Accommodation,Chennai,6200\n2026-10-17,Venkatesh,Client Dinner,Chennai,2100",
        "category": "financial",
    },
    {
        "id": "fin_tax_04",
        "file_name": "gst_quarterly_tax_filing_q3.txt",
        "content": "GST tax filing return for Q3 2026. Input tax credit reconciled against supplier invoices. Output GST collected on software maintenance and consulting contracts.",
        "category": "financial",
    },
    {
        "id": "fin_salary_05",
        "file_name": "payroll_summary_october_2026.csv",
        "content": "EmpID,Name,Designation,Basic_Salary,Allowances,Net_Pay\n101,Anand,Lead Architect,180000,25000,195000\n102,Kavitha,AI Researcher,160000,22000,174000\n103,Manoj,Systems Engineer,120000,18000,132000",
        "category": "financial",
    },
    # Meeting Minutes & Notes
    {
        "id": "meet_sync_01",
        "file_name": "ai_agent_sprint_standup_notes.md",
        "content": "Weekly AI sprint standup: Reviewed BGE-M3 embedding latency on CPU. Fixed memory leak during model unload. Action item: implement reciprocal rank fusion k=60.",
        "category": "meeting",
    },
    {
        "id": "meet_exec_02",
        "file_name": "board_meeting_minutes_october.txt",
        "content": "Board of Directors meeting minutes. Strategic discussion on AI assistant product roadmap, enterprise privacy controls, local on-device execution, and funding targets.",
        "category": "meeting",
    },
    {
        "id": "meet_client_03",
        "file_name": "client_requirements_interview.md",
        "content": "Client discovery notes: Customer requires voice-driven file management, multilingual support in Tamil and English, and strict air-gapped data retention without cloud leaks.",
        "category": "meeting",
    },
    # HR & Resumes
    {
        "id": "hr_resume_01",
        "file_name": "senior_ai_engineer_resume_kavitha.txt",
        "content": "Curriculum Vitae: Kavitha Sundaram. Senior AI Engineer with 7 years experience in PyTorch, transformers, vector retrieval, FAISS, Tamil NLP, and speech recognition models.",
        "category": "hr",
    },
    {
        "id": "hr_policy_02",
        "file_name": "work_from_home_guidelines_2026.txt",
        "content": "Company remote work policy: Core business hours from 10:00 AM to 5:00 PM. High-speed broadband stipend, ergonomic chair reimbursement, and VPN security standards.",
        "category": "hr",
    },
    # Tamil Documents (தமிழ் ஆவணங்கள்)
    {
        "id": "ta_doc_01",
        "file_name": "திட்ட_அறிக்கை_செயற்கை_நுண்ணறிவு.txt",
        "content": "செயற்கை நுண்ணறிவு திட்ட அறிக்கை: ஹாக்ஸ் குரல் உதவியாளர் மற்றும் கோப்பு மேலாண்மை கட்டமைப்பு. தமிழ் மொழியில் கணினி கட்டுப்பாடு மற்றும் ஆவண தேடல் வசதிகள் செயல்படுத்தப்பட்டுள்ளன.",
        "category": "tamil_native",
    },
    {
        "id": "ta_doc_02",
        "file_name": "வரவு_செலவு_கணக்கு_2026.csv",
        "content": "பிரிவு,முதல்_காலாண்டு,இரண்டாம்_காலாண்டு,மொத்தம்\nபொறியியல்,120000,140000,260000\nசந்தைப்படுத்தல்,40000,45000,85000\nபராமரிப்பு,20000,22000,42000",
        "category": "tamil_native",
    },
    {
        "id": "ta_doc_03",
        "file_name": "கூட்ட_குறிப்புகள்_அக்டோபர்.md",
        "content": "வாராந்திர ஆலோசனைக் கூட்டம்: தமிழ் குரல் அங்கீகாரம், குறைந்த தாமதம் மற்றும் விண்டோஸ் கணினி செயல்பாடுகள் குறித்த முக்கிய முடிவுகள் எடுக்கப்பட்டன.",
        "category": "tamil_native",
    },
    {
        "id": "ta_doc_04",
        "file_name": "வன்பொருள்_விலைப்பட்டியல்_ரசீது.txt",
        "content": "வன்பொருள் மற்றும் மடிக்கணினி விலைப்பட்டியல் ரசீது எண் 8472. புதிய இன்டெல் செயலி மற்றும் ரேம் நினைவகம் வாங்கியதற்கான அதிகாரப்பூர்வ ரசீது மற்றும் கட்டண விவரங்கள்.",
        "category": "tamil_native",
    },
    {
        "id": "ta_doc_05",
        "file_name": "பணியாளர்_விவரக்குறிப்பு_சுயவிவரம்.txt",
        "content": "சுயவிவரக் குறிப்பு: கணேஷ் குமார், மூத்த மென்பொருள் பொறியாளர். கணினி நிரலாக்கம், தரவுத்தளம் மற்றும் பைதான் மொழியில் அனுபவம் பெற்றவர்.",
        "category": "tamil_native",
    },
]

# ------------------------------------------------------------------------------
# 2. 100 Queries per Category (English, Tamil, Tanglish)
# ------------------------------------------------------------------------------

def generate_benchmark_queries() -> Dict[str, List[Dict[str, Any]]]:
    """Generates 100 English, 100 Tamil, and 100 Tanglish benchmark queries with ground truth."""
    queries: Dict[str, List[Dict[str, Any]]] = {"en": [], "ta": [], "tanglish": []}

    # Templates and variants to produce exactly 100 queries per category
    en_patterns = [
        ("where is the {topic} file", "where is the {topic}"),
        ("find document regarding {topic}", "find {topic} document"),
        ("search for {topic} notes", "search {topic} notes"),
        ("show me the latest {topic} report", "show latest {topic}"),
        ("find files containing {topic}", "files with {topic}"),
    ]
    en_topics = [
        ("voice architecture", "tech_arch_01"),
        ("webrtc apm", "tech_arch_01"),
        ("silero vad speech detection", "tech_arch_01"),
        ("barge in interruption", "tech_arch_01"),
        ("sqlite fts5", "tech_db_02"),
        ("bm25 ranking", "tech_db_02"),
        ("porter stemming", "tech_db_02"),
        ("docker compose deployment", "tech_docker_03"),
        ("gpu passthrough containers", "tech_docker_03"),
        ("gemini live websocket", "tech_api_04"),
        ("pcm audio streaming", "tech_api_04"),
        ("faiss vector index", "tech_faiss_05"),
        ("bge-m3 embeddings benchmark", "tech_faiss_05"),
        ("annual budget forecast 2026", "fin_budget_01"),
        ("engineering department budget", "fin_budget_01"),
        ("vendor hardware invoice", "fin_invoice_02"),
        ("dell optiplex workstation bill", "fin_invoice_02"),
        ("travel expense report chennai", "fin_expense_03"),
        ("flight booking bangalore", "fin_expense_03"),
        ("gst tax filing return", "fin_tax_04"),
    ]

    # Generate 100 English queries (5 patterns * 20 topics = 100)
    for p_idx, (pat, _) in enumerate(en_patterns):
        for topic, target_id in en_topics:
            q_text = pat.format(topic=topic)
            queries["en"].append({
                "id": f"en_{len(queries['en'])+1:03d}",
                "query": q_text,
                "relevant_ids": [target_id],
                "topic": topic,
            })

    # Tamil patterns and topics (100 queries)
    ta_patterns = [
        ("{topic} கோப்பைத் தேடு", "கோப்பைத் தேடு"),
        ("{topic} ஆவணம் எங்கே உள்ளது", "ஆவணம் எங்கே"),
        ("{topic} குறித்த தகவலைக் காட்டு", "தகவல் காட்டு"),
        ("{topic} அறிக்கை எங்கே", "அறிக்கை எங்கே"),
        ("{topic} குறிப்புகளைக் கண்டுபிடி", "கண்டுபிடி"),
    ]
    ta_topics = [
        ("திட்ட அறிக்கை", "ta_doc_01"),
        ("செயற்கை நுண்ணறிவு", "ta_doc_01"),
        ("ஹாக்ஸ் குரல் உதவியாளர்", "ta_doc_01"),
        ("வரவு செலவு கணக்கு", "ta_doc_02"),
        ("பொறியியல் காலாண்டு செலவு", "ta_doc_02"),
        ("ஆலோசனைக் கூட்டம்", "ta_doc_03"),
        ("கூட்ட குறிப்புகள் அக்டோபர்", "ta_doc_03"),
        ("தமிழ் குரல் அங்கீகாரம்", "ta_doc_03"),
        ("வன்பொருள் விலைப்பட்டியல்", "ta_doc_04"),
        ("மடிக்கணினி ரசீது", "ta_doc_04"),
        ("இன்டெல் செயலி ரசீது", "ta_doc_04"),
        ("கட்டண விவரங்கள்", "ta_doc_04"),
        ("பணியாளர் விவரக்குறிப்பு", "ta_doc_05"),
        ("கணேஷ் குமார் சுயவிவரம்", "ta_doc_05"),
        ("மென்பொருள் பொறியாளர் அனுபவம்", "ta_doc_05"),
        ("விண்டோஸ் கணினி செயல்பாடுகள்", "ta_doc_03"),
        ("சந்தைப்படுத்தல் செலவு", "ta_doc_02"),
        ("குரல் மேலாண்மை கட்டமைப்பு", "ta_doc_01"),
        ("நினைவகம் வாங்கிய ரசீது", "ta_doc_04"),
        ("பைதான் நிரலாக்க அனுபவம்", "ta_doc_05"),
    ]

    # Generate 100 Tamil queries (5 patterns * 20 topics = 100)
    for pat, _ in ta_patterns:
        for topic, target_id in ta_topics:
            q_text = pat.format(topic=topic)
            queries["ta"].append({
                "id": f"ta_{len(queries['ta'])+1:03d}",
                "query": q_text,
                "relevant_ids": [target_id],
                "topic": topic,
            })

    # Tanglish patterns and topics (100 queries)
    tanglish_patterns = [
        ("{topic} file enga irukku", "enga irukku"),
        ("{topic} document thedu", "document thedu"),
        ("{topic} report kaatunga", "report kaatunga"),
        ("{topic} notes kandupidi", "notes kandupidi"),
        ("{topic} pathi details search pannu", "search pannu"),
    ]
    tanglish_topics = [
        ("voice architecture", "tech_arch_01"),
        ("silero vad", "tech_arch_01"),
        ("barge in", "tech_arch_01"),
        ("sqlite fts5", "tech_db_02"),
        ("bm25 ranking", "tech_db_02"),
        ("docker compose", "tech_docker_03"),
        ("gpu passthrough", "tech_docker_03"),
        ("gemini live", "tech_api_04"),
        ("faiss vector", "tech_faiss_05"),
        ("budget 2026", "fin_budget_01"),
        ("hardware invoice", "fin_invoice_02"),
        ("dell optiplex bill", "fin_invoice_02"),
        ("travel expense chennai", "fin_expense_03"),
        ("flight booking", "fin_expense_03"),
        ("gst tax filing", "fin_tax_04"),
        ("payroll summary", "fin_salary_05"),
        ("standup notes", "meet_sync_01"),
        ("board meeting", "meet_exec_02"),
        ("kavitha resume", "hr_resume_01"),
        ("wfh policy", "hr_policy_02"),
    ]

    # Generate 100 Tanglish queries (5 patterns * 20 topics = 100)
    for pat, _ in tanglish_patterns:
        for topic, target_id in tanglish_topics:
            q_text = pat.format(topic=topic)
            queries["tanglish"].append({
                "id": f"tanglish_{len(queries['tanglish'])+1:03d}",
                "query": q_text,
                "relevant_ids": [target_id],
                "topic": topic,
            })

    return queries


# ------------------------------------------------------------------------------
# 3. Metric Computations: Recall, MRR, nDCG, Precision, Latency
# ------------------------------------------------------------------------------

def compute_retrieval_metrics(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
) -> Dict[str, float]:
    """Computes Recall@5, Recall@10, MRR@10, nDCG@10, and Precision@5."""
    rel_set = set(relevant_doc_ids)
    if not rel_set:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "precision@5": 0.0}

    # Top 5
    top5 = retrieved_doc_ids[:5]
    hits5 = sum(1 for d in top5 if d in rel_set)
    recall5 = min(1.0, hits5 / len(rel_set))
    precision5 = hits5 / 5.0

    # Top 10
    top10 = retrieved_doc_ids[:10]
    hits10 = sum(1 for d in top10 if d in rel_set)
    recall10 = min(1.0, hits10 / len(rel_set))

    # MRR@10
    mrr10 = 0.0
    for rank, doc_id in enumerate(top10, start=1):
        if doc_id in rel_set:
            mrr10 = 1.0 / rank
            break

    # nDCG@10
    dcg10 = 0.0
    for rank, doc_id in enumerate(top10, start=1):
        rel = 1.0 if doc_id in rel_set else 0.0
        dcg10 += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG for 1 relevant document
    idcg10 = (2**1.0 - 1) / math.log2(1 + 1)
    ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0

    return {
        "recall@5": recall5,
        "recall@10": recall10,
        "mrr@10": mrr10,
        "ndcg@10": ndcg10,
        "precision@5": precision5,
    }


# ------------------------------------------------------------------------------
# 4. Multilingual Benchmark Runner
# ------------------------------------------------------------------------------

class MultilingualBenchmarkRunner:
    """Runs end-to-end multilingual retrieval benchmarks."""

    def __init__(self, temp_dir: Optional[Path] = None, use_mock_fallback: bool = True) -> None:
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="haks_benchmark_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock_fallback = use_mock_fallback
        self.doc_file_map: Dict[str, Path] = {}

    def setup_corpus(self) -> None:
        """Write corpus files to disk."""
        corpus_dir = self.temp_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        for doc in CORPUS_DOCUMENTS:
            file_path = corpus_dir / doc["file_name"]
            file_path.write_text(doc["content"], encoding="utf-8")
            self.doc_file_map[doc["id"]] = file_path

    def run_benchmark(self) -> Dict[str, Any]:
        """Execute the 300+ query benchmark across English, Tamil, and Tanglish."""
        self.setup_corpus()
        config = AppConfig()

        service = FileIntelligenceService(config=config, use_mock_fallback=self.use_mock_fallback)

        # Index corpus
        corpus_dir = self.temp_dir / "corpus"
        index_stats = service.index_directory(corpus_dir, force_reindex=True)
        logger.info("Corpus indexing completed: %s", index_stats)

        queries_by_lang = generate_benchmark_queries()
        results: Dict[str, Any] = {}

        for lang, q_list in queries_by_lang.items():
            logger.info("Running benchmark for language category '%s' (%d queries)...", lang, len(q_list))
            metrics_acc: Dict[str, List[float]] = {
                "recall@5": [],
                "recall@10": [],
                "mrr@10": [],
                "ndcg@10": [],
                "precision@5": [],
            }
            latencies_ms: List[float] = []

            for q in q_list:
                start_t = time.perf_counter()
                search_results = service.search(
                    natural_query=q["query"],
                    search_mode=SearchMode.HYBRID,
                    top_k=10,
                    rerank=True,
                    directory_scope=str(corpus_dir),
                )
                dur_ms = (time.perf_counter() - start_t) * 1000.0
                latencies_ms.append(dur_ms)

                # Map retrieved file paths to document IDs
                retrieved_ids = []
                for item in search_results:
                    for d_id, fp in self.doc_file_map.items():
                        if Path(item.file_path).name == fp.name:
                            retrieved_ids.append(d_id)
                            break

                q_metrics = compute_retrieval_metrics(
                    retrieved_doc_ids=retrieved_ids,
                    relevant_doc_ids=q["relevant_ids"],
                )
                for k, v in q_metrics.items():
                    metrics_acc[k].append(v)

            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))
            p99 = float(np.percentile(latencies_ms, 99))

            results[lang] = {
                "num_queries": len(q_list),
                "recall@5": float(np.mean(metrics_acc["recall@5"])),
                "recall@10": float(np.mean(metrics_acc["recall@10"])),
                "mrr@10": float(np.mean(metrics_acc["mrr@10"])),
                "ndcg@10": float(np.mean(metrics_acc["ndcg@10"])),
                "precision@5": float(np.mean(metrics_acc["precision@5"])),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
            }

        # Calculate Overall Summary
        all_recall5 = np.mean([results[l]["recall@5"] for l in results])
        all_recall10 = np.mean([results[l]["recall@10"] for l in results])
        all_mrr10 = np.mean([results[l]["mrr@10"] for l in results])
        all_ndcg10 = np.mean([results[l]["ndcg@10"] for l in results])
        all_prec5 = np.mean([results[l]["precision@5"] for l in results])
        all_p50 = np.mean([results[l]["latency_p50_ms"] for l in results])
        all_p95 = np.mean([results[l]["latency_p95_ms"] for l in results])
        all_p99 = np.mean([results[l]["latency_p99_ms"] for l in results])

        results["overall"] = {
            "num_queries": sum(len(q) for q in queries_by_lang.values()),
            "recall@5": float(all_recall5),
            "recall@10": float(all_recall10),
            "mrr@10": float(all_mrr10),
            "ndcg@10": float(all_ndcg10),
            "precision@5": float(all_prec5),
            "latency_p50_ms": float(all_p50),
            "latency_p95_ms": float(all_p95),
            "latency_p99_ms": float(all_p99),
        }

        service.unload_models()
        return results


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print markdown and console table summary of benchmark metrics."""
    print("\n" + "=" * 90)
    print("HAKS PHASE 9.2: INDUSTRIAL FILE INTELLIGENCE - MULTILINGUAL RETRIEVAL BENCHMARK")
    print("=" * 90)
    print(f"{'Category':<12} | {'Queries':<8} | {'R@5':<7} | {'R@10':<7} | {'MRR@10':<7} | {'nDCG@10':<8} | {'P@5':<7} | {'p50 (ms)':<9} | {'p95 (ms)':<9}")
    print("-" * 90)
    for cat in ("en", "ta", "tanglish", "overall"):
        data = results.get(cat, {})
        name = "English" if cat == "en" else "Tamil" if cat == "ta" else "Tanglish" if cat == "tanglish" else "OVERALL"
        print(
            f"{name:<12} | {data.get('num_queries', 0):<8} | "
            f"{data.get('recall@5', 0):.3f}   | "
            f"{data.get('recall@10', 0):.3f}   | "
            f"{data.get('mrr@10', 0):.3f}    | "
            f"{data.get('ndcg@10', 0):.3f}    | "
            f"{data.get('precision@5', 0):.3f}   | "
            f"{data.get('latency_p50_ms', 0):.2f}     | "
            f"{data.get('latency_p95_ms', 0):.2f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    runner = MultilingualBenchmarkRunner(use_mock_fallback=True)
    report = runner.run_benchmark()
    print_benchmark_report(report)
