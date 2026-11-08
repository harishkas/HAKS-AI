# HAKS — Personal AI Assistant

HAKS is an offline-first, privacy-focused personal AI operating assistant architected as a **Modular Monolith** in Python.

HAKS combines local and cloud AI models, dual independent long-term memory systems (Personal Memory and Work Memory), continuous voice conversation in Tamil and English, and permission-governed Windows computer control.

---

## Hardware Target & Specifications

HAKS is calibrated to run on constrained laptop hardware:

| Component | Target Specification |
|---|---|
| **Device** | Dell Inspiron 3576 |
| **CPU** | Intel Core i5-7200U (2 cores, 4 threads @ 2.50 GHz) |
| **RAM** | 8 GB DDR4 (1.6 GB - 2.5 GB free during typical desktop load) |
| **GPU** | AMD Radeon 520 (2 GB dedicated VRAM) + Intel HD Graphics 620 |
| **Storage** | 238 GB SSD (C: ~151 GB, E: ~87 GB) |
| **OS** | Windows 10/11 Home (64-bit) |

### Hardware Reality & Constraints
- **CPU Inference Only**: The AMD Radeon 520 does not support modern ROCm or HIP inference on Windows. All local AI inference executes on the dual-core Intel Core i5-7200U via AVX2 instructions.
- **Strict RAM Ceiling**: With 8 GB total system RAM, operating system processes, background applications, and PySide6 UI consume approximately 4.5 GB - 5.5 GB.
- **Model Size Rule**: Models larger than 3.5 GB (e.g. 7B, 8B, 14B) will trigger aggressive Windows pagefile swapping, freezing desktop responsiveness. Only 1B - 3B quantized models (`Q4_K_M`) are supported for local inference.

---

## System Architecture

HAKS uses a **Modular Monolithic Architecture** with strict layer separation:

1. **Presentation Layer (`ui/`)**: PySide6 / Qt 6 desktop application, companion draggable floating assistant, and system tray.
2. **Application Layer (`core/`)**: Unified application controller (`AppController`), event bus (`ApplicationEvents`), and thread-safe signal dispatchers.
3. **AI Engine Layer (`ai/`)**:
   - `AIEngineInterface`: Common asynchronous abstract interface for all local and cloud engines.
   - `LocalAIEngine`: Hardware-aware local inference manager wrapping Ollama.
   - `OllamaClient`: Low-level asynchronous REST & NDJSON streaming client for Ollama.
   - `CloudAIEngine`: Provider-independent cloud adapter coordinating multiple providers.
   - `CloudProviderAdapter`: Pluggable adapters for Google Gemini, OpenAI, and Anthropic Claude.
   - `CredentialManager`: Secure OS-level key storage using Windows Credential Manager.
   - `AIEngineManager`: Central lifecycle coordinator managing engine selection, streaming, health monitoring, and cancellation.
   - `AIInferenceWorker`: Non-blocking background `QThread` ensuring the Qt GUI maintains 60fps responsiveness during generation.
4. **Memory, Voice & Security Layer (`storage/`)**:
   - **Personal Memory**: Personal history, goals, preferences, and moments stored in `personal_memory.db`.
   - **Work Memory**: Technical discussions, projects, architecture, and code context in `work_memory.db`.
   - **Privacy Boundary**: Memory databases are strictly kept local and are NEVER transmitted to cloud AI engines. Only current conversation messages explicitly selected by the user are sent.
5. **Infrastructure & Storage**: 5 independent SQLite databases operating in WAL mode with foreign-key enforcement.

---

## Directory Structure

```
HAKS/
│
├── app.py                             # Main application entry point & CLI runner
├── pyproject.toml                     # Project packaging, dependencies & pytest config
├── requirements.txt                   # Pinned runtime dependencies
├── README.md                          # System documentation & manual
├── .env.example                       # Environment template without credentials
├── .gitignore                         # Git ignore rules
│
├── ai/                                # Phase 3 AI Engine Integration
│   ├── __init__.py                    # Public AI interface exports
│   ├── models.py                      # AIMessage, AIResponse, AIStreamChunk, AIEngineConfig, AIEngineHealth
│   ├── interface.py                   # AIEngineInterface abstract base class
│   ├── exceptions.py                  # Structured AI error hierarchy
│   ├── credentials.py                 # Windows Credential Manager (keyring) integration
│   ├── ollama_client.py               # Asynchronous Ollama REST & streaming NDJSON client
│   ├── local_engine.py                # LocalAIEngine with 8 GB RAM hardware safeguards
│   ├── cloud_engine.py                # Provider-independent CloudAIEngine
│   ├── manager.py                     # AIEngineManager unified lifecycle & cancellation coordinator
│   ├── worker.py                      # Dedicated QThread worker for non-blocking UI streaming
│   └── providers/                     # Cloud provider adapters
│       ├── __init__.py
│       ├── base.py                    # CloudProviderAdapter abstract base class
│       ├── gemini.py                  # Google Gemini API & SSE adapter
│       ├── openai_compatible.py       # OpenAI / Groq / OpenRouter adapter
│       └── anthropic.py               # Anthropic Claude Messages adapter
│
├── config/                            # Configuration management
│   ├── __init__.py
│   └── settings.py                    # Pydantic AppConfig, paths, database, & AI settings
│
├── core/                              # Core system services
│   ├── __init__.py
│   ├── logger.py                      # Rotating file & console logging
│   ├── events.py                      # PySide6 signal bus (ApplicationEvents)
│   └── app_controller.py              # Central application controller
│
├── storage/                           # SQLite database framework
│   ├── __init__.py
│   ├── database.py                    # Connection management, WAL mode, transaction sessions
│   └── schema.py                      # Schemas for the 5 independent databases
│
├── ui/                                # Desktop presentation layer
│   ├── __init__.py
│   ├── app.py                         # HaksApplication Qt lifecycle manager
│   ├── main_window.py                 # MainWindow with sidebar, top bar, & navigation
│   ├── floating_assistant.py          # Compact floating AI companion
│   ├── system_tray.py                 # Windows notification tray icon
│   ├── styles.py                      # QSS stylesheet & dark design system
│   └── views/                         # Modular UI views
│       ├── chat_view.py               # Interactive Chat with streaming AI & engine selectors
│       ├── ai_engine_view.py          # AI Engine management, diagnostics & key configuration
│       ├── history_view.py            # Conversation history browser
│       ├── personal_memory_view.py    # Isolated Personal Memory browser
│       ├── work_memory_view.py        # Isolated Work Memory browser
│       ├── voice_view.py              # Voice pipeline state visualizer
│       └── settings_view.py           # Application settings & theme switcher
│
├── tests/                             # Automated test suite (59 tests)
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures (headless Qt & isolated storage)
│   ├── test_config.py                 # Configuration tests
│   ├── test_logger.py                 # Logger tests
│   ├── test_database.py               # Database schema, WAL mode, & isolation tests
│   ├── test_app_controller.py         # Controller and state persistence tests
│   ├── test_credentials.py            # Windows Credential Manager tests
│   ├── test_ai_models_and_interface.py# Typed models and interface conformity tests
│   ├── test_ollama_client_and_local_engine.py # Local Ollama detection, generation, streaming, pulling & worker tests
│   ├── test_cloud_engine_and_providers.py     # Cloud adapter tests (Gemini, OpenAI, Groq, OpenRouter, Custom)
│   ├── test_ai_manager_and_cancellation.py    # Engine switching, health, and cancellation tests
│   ├── test_ui_chat_streaming.py      # ChatView streaming, retry flow, stop button, and privacy tests
│   ├── test_ui_foundation.py          # MainWindow instantiation tests
│   └── test_views.py                  # View navigation, AIEngineView controls & state tests
│
├── logs/                              # Application log files (git-ignored)
└── data/                              # SQLite databases (git-ignored)
```

---

## Local AI Setup (Ollama)

### 1. Installing Ollama on Windows
1. Download Ollama for Windows from [https://ollama.com/download](https://ollama.com/download).
2. Run the installer (`OllamaSetup.exe`).
3. Ensure Ollama starts automatically in your Windows taskbar notification area, or start it manually:
   ```cmd
   ollama serve
   ```
4. Verify the local daemon is responding:
   ```powershell
   curl.exe http://127.0.0.1:11434/api/version
   ```

### 2. Recommended Lightweight Quantized Models
On the target Intel Core i5-7200U with 8 GB RAM, pull lightweight models:

```powershell
# Recommended Default: Llama 3.2 1B (Fast, ~1.3 GB RAM footprint)
ollama pull llama3.2:1b

# Alternative: Qwen 2.5 1.5B (Excellent coding & reasoning, ~1.6 GB RAM)
ollama pull qwen2.5:1.5b

# Optional: Llama 3.2 3B (Maximum recommended model for 8 GB RAM)
ollama pull llama3.2:3b
```

> [!CAUTION]
> Do NOT download `llama3:8b`, `mistral:7b`, or `qwen2.5:14b`. On an 8 GB system, these models require 5.5 GB - 10 GB of memory and will cause high disk paging, freezing the computer.

---

## Cloud AI Configuration

HAKS includes a provider-independent Cloud AI adapter supporting:
- **Google Gemini API** (`gemini-1.5-flash`, `gemini-1.5-pro`)
- **OpenAI API** (`gpt-4o-mini`, `gpt-4o`, `o3-mini`, `gpt-4-turbo`)
- **Groq Cloud** (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`)
- **OpenRouter** (`google/gemini-2.0-flash-001`, `meta-llama/llama-3.3-70b-instruct`, `anthropic/claude-3.5-haiku`)
- **Anthropic Claude API** (`claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`)
- **Custom OpenAI-Compatible Endpoints** (LocalAI, vLLM, Ollama-remote, or private inference servers with editable Base URLs)

### 1. Storing API Keys Securely
API keys are **never** stored in plain-text files or configuration scripts. HAKS stores credentials securely in the **Windows Credential Manager** (`keyring` backend `WinVaultKeyring`):

1. Launch HAKS desktop app.
2. Navigate to **AI Engine** in the sidebar.
3. Under **Cloud AI Adapter**, select your provider (e.g., **Google Gemini API**, **Groq Cloud**, etc.).
4. Click **Configure API Key**.
5. Enter your key in the secure password-masked dialog and click OK.
6. The key is encrypted by Windows DPAPI and stored under target `HAKS_AI/<provider>`.

### 2. Environment Variable Override (Optional)
For headless testing or CI environments, keys can also be supplied via environment variables:
```powershell
$env:HAKS_GEMINI_API_KEY = "your_key_here"
$env:HAKS_OPENAI_API_KEY = "your_key_here"
$env:HAKS_GROQ_API_KEY = "your_key_here"
$env:HAKS_OPENROUTER_API_KEY = "your_key_here"
$env:HAKS_ANTHROPIC_API_KEY = "your_key_here"
```

---

## Privacy & Security Guarantees

1. **Zero Silent Fallback**: If Cloud AI fails or is unconfigured, HAKS displays an explicit error message. It never silently switches between cloud and local without explicit user action.
2. **Context Isolation**: Only the recent conversation messages (up to 20 messages) are passed as prompt context. Personal Memory (`personal_memory.db`) and Work Memory (`work_memory.db`) are **never** uploaded or passed to cloud engines.
3. **No Key Exposure**: API keys are masked in UI views (`••••••••AbCd`) and are never written to `logs/haks.log` or diagnostic reports.

---

## Getting Started

### 1. Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.10+ (tested on Python 3.14.6)
- Git for Windows

### 2. Environment Setup
```powershell
# Navigate to project directory
cd "c:\Users\Admin\Music\HAKS AI"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install runtime and testing dependencies
pip install -r requirements.txt
```

### 3. Running the Test Suite (59 Tests)
```powershell
.\.venv\Scripts\pytest.exe
```
Output:
```
============================= 59 passed in 20.42s =============================
```

### 4. Running the Smoke Test
```powershell
.\.venv\Scripts\python.exe app.py --smoke-test
```

### 5. Launching HAKS Desktop Application
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## Troubleshooting Guide

| Issue | Cause | Resolution |
|---|---|---|
| **Ollama: Cannot connect to http://127.0.0.1:11434** | Ollama daemon is not running. | Run `ollama serve` or start Ollama from the Windows Start menu. |
| **Model 'llama3.2:1b' not found** | Model has not been pulled to local storage. | Run `ollama pull llama3.2:1b` in terminal. |
| **Cloud AI: Provider not configured with an API key** | API key is missing from Windows Credential Manager. | Open HAKS UI -> AI Engine -> Click "Configure API Key". |
| **Cloud AI: Authentication failed (HTTP 400/401/403)** | API key is invalid, revoked, or expired. | Check your API key in Google AI Studio or OpenAI dashboard and re-enter. |
| **System sluggishness during inference** | Too large a model was loaded in Ollama. | In HAKS AI Engine view, click "Unload Active Model" and switch to `llama3.2:1b`. |
| **Headless Qt test error (`qt.qpa.plugin`)** | PySide6 cannot find a display. | Set environment variable: `$env:QT_QPA_PLATFORM="offscreen"`. (Handled automatically in `tests/conftest.py`). |

---

## Release Status & Roadmap

- [x] **Phase 1 — Foundation** (Configuration, Logging, SQLite WAL Databases, Basic PySide6 UI)
- [x] **Phase 2 — Desktop UI** (Unified AppController, ChatView, Floating Assistant, History, Themes)
- [x] **Phase 3 — AI Engine Integration** (Ollama Local AI, Cloud AI Provider Adapter, Non-Blocking Streaming, Windows Credential Manager, Cancellation)
- [ ] **Phase 4 — Intelligent AI Routing & Privacy Management** (Rule-based routing, complexity classifier, consent manager, fallback approvals)
- [ ] **Phase 5 — Voice Pipeline** (Silero VAD, Tamil/English STT/TTS, wake-word "Hey HAKS")
- [ ] **Phase 6 — Windows Computer Control & Security** (OS actions, 4-tier risk classification, audit logs)
- [ ] **Phase 7 — Offline & Disaster Recovery** (Automated backups, integrity validation)
- [ ] **Phase 8 — Packaging & Release 0.1** (PyInstaller packaging, final verification)
