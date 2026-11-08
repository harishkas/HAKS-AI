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
4. **Intelligent AI Routing & Privacy Layer (`routing/`)**:
   - `HybridAIRouter`: Multi-factor deterministic classifier evaluating sensitivity, complexity, network availability, and user preferences.
   - `DataRedactor`: Regular expression and Shannon entropy scanner masking API keys, emails, IPs, and payment credentials.
   - `ConsentManager`: Transparent user approval workflow for routing private content to cloud AI.
5. **Continuous Voice Architecture (`voice/`)**:
   - `AudioCaptureService`: Real-time microphone capture via `sounddevice` with RMS level monitoring, hardware enumeration, and mute.
   - `VoiceActivityDetector`: Dynamic noise-floor tracking VAD with pre-speech padding and silence timeout.
   - `WakeWordDetector`: Dual-engine wake detector for "Hey HAKS" with offline local acoustic burst/fricative analysis and optional Picovoice Porcupine custom keyword support.
   - `FasterWhisperSTT`: Quantized CPU int8 bilingual speech recognition (Tamil, English, Mixed) with lazy loading and prompt conditioning.
   - `LanguageAwareTTSRouter`: Script-aware text-to-speech router directing Tamil to high-fidelity Edge Neural voices (`ta-IN-PallaviNeural`, `ta-IN-ValluvarNeural`) and English to Edge or offline Windows SAPI (`pyttsx3`).
   - `AudioPlayerService`: Sub-50ms latency barge-in audio playback decoded via PyAV with instant stop on user interruption.
   - `ContinuousVoiceEngine`: Central state machine (`STANDBY` -> `WAKE_DETECTED` -> `LISTENING` -> `PROCESSING` -> `THINKING` -> `SPEAKING` -> `INTERRUPTED`) with built-in voice commands in English and Tamil.
6. **Memory & Security Layer (`memory/`, `storage/`)**:
   - **Personal Memory**: Personal history, goals, preferences, and moments stored in `personal_memory.db`.
   - **Work Memory**: Technical discussions, projects, architecture, and code context in `work_memory.db`.
   - **Privacy Boundary**: Memory databases are strictly kept local, encrypted at rest via AES-256 (Fernet + PBKDF2), and NEVER transmitted to cloud AI engines.
7. **Windows Computer Control & Administrative Automation (`windows_control/`)**:
   - `WindowsControlManager`: Central coordinator managing full-control execution, system-wide resource discovery, and hardware diagnostics.
   - `SystemHardwareProfiler`: Genuine hardware profiler detecting CPU cores, physical RAM, genuine GPU compute limits, displays, and audio endpoints.
   - `ResourceSafetyCheck`: Pre-flight capacity evaluator enforcing 2.0 GB disk and 1.0 GB RAM safety margins before heavy operations.
   - `SystemResourceDiscoveryService`: Multi-drive inspection (C:, D:, E:), user directories (with OneDrive redirection support), Python runtimes, Git ecosystem, dev tools, and installed software inventory.
   - `FilesystemService`: Permission-aware traversal (`SearchResult`), large file detection, in-place rename, cross-drive copy/move, and Recycle Bin safeguards.
   - `CommandExecutor` & `ElevationManager`: Async CMD/PowerShell execution with streaming output, UAC administrator elevation, and immutable redacted audit logging in `settings.db`.
8. **Infrastructure & Storage**: 5 independent SQLite databases operating in WAL mode with foreign-key enforcement.

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
├── memory/                             # Phase 4 Memory Architecture
│   ├── __init__.py                    # Public memory models and service exports
│   ├── models.py                      # Pydantic models: MemoryItem, MemoryProposal, MemoryBackupBundle
│   ├── encryption.py                  # AES-256 (Fernet + PBKDF2) encryption at rest & key recovery
│   ├── vector_index.py                # 384-dim subword n-gram vectorizer & cosine similarity index
│   ├── proposal_detector.py           # Automatic memory proposal pattern detector
│   ├── proposal_manager.py            # User approval/rejection lifecycle workflow manager
│   ├── personal_memory.py             # Personal Memory domain manager (personal_memory.db)
│   ├── work_memory.py                 # Work Memory domain manager (work_memory.db)
│   └── context_builder.py             # Minimal disclosure relevant memory prompt injector
│
├── routing/                            # Phase 5 Intelligent Routing & Privacy
│   ├── __init__.py                    # Public routing exports
│   ├── models.py                      # PrivacyLevel, ComplexityLevel, RoutingDecision, ConsentRequest
│   ├── privacy_classifier.py          # Deterministic regex, entropy & keyword privacy classifier
│   ├── complexity_analyzer.py         # Multi-factor token, linguistic & AST complexity evaluator
│   ├── router.py                      # HybridAIRouter selecting Local vs Cloud engines
│   ├── consent_manager.py             # User consent tracking & temporary cloud authorization
│   ├── data_redactor.py               # Masking of PII, API tokens, IP addresses, credentials
│   └── exceptions.py                  # Structured routing error classes
│
├── voice/                              # Phase 6 Continuous Voice Architecture
│   ├── __init__.py                    # Public voice exports
│   ├── models.py                      # VoiceState, VoiceCommandType, AudioDevice, TranscriptionResult, SpokenUtterance
│   ├── audio_capture.py               # Real-time sounddevice microphone capture, RMS VU level, mute
│   ├── vad.py                         # Voice Activity Detector with dynamic noise tracking & pre-speech padding
│   ├── player.py                      # PyAV audio decoding & sounddevice output with sub-50ms barge-in stop
│   ├── conversation_engine.py         # Continuous state machine, bilingual commands, AI & memory integration
│   ├── wake_word/                     # Wake word detectors ("Hey HAKS")
│   │   ├── __init__.py
│   │   ├── base.py                    # WakeWordDetectorInterface
│   │   ├── local_detector.py          # Offline acoustic energy burst & fricative /s/ detector
│   │   └── porcupine_detector.py      # Picovoice Porcupine adapter with custom keyword support
│   ├── stt/                           # Speech-to-Text
│   │   ├── __init__.py
│   │   ├── base.py                    # SpeechToTextInterface
│   │   └── faster_whisper_stt.py      # CPU int8 Faster-Whisper with Tamil/English auto-detection & lazy loading
│   └── tts/                           # Text-to-Speech
│       ├── __init__.py
│       ├── base.py                    # TextToSpeechInterface
│       ├── edge_tts_engine.py         # Edge Neural Tamil (Pallavi/Valluvar) & English (Neerja/Jenny)
│       ├── sapi_tts_engine.py         # Offline Windows SAPI engine (pyttsx3)
│       ├── piper_tts_engine.py        # Local ONNX Piper TTS adapter
│       └── router.py                  # LanguageAwareTTSRouter (Unicode Tamil vs English script)
│
├── storage/                           # SQLite database & persistence framework
│   ├── __init__.py
│   ├── database.py                    # Connection management, WAL mode, transaction sessions
│   ├── schema.py                      # Schemas for 5 databases & Phase 4 migrations
│   ├── backup_manager.py              # Independent domain backup, restore, export (JSON/CSV/MD), & wiping
│   └── migration.py                   # Safe storage directory migration with conflict protection
│
├── ui/                                # Desktop presentation layer
│   ├── __init__.py
│   ├── app.py                         # HaksApplication Qt lifecycle manager
│   ├── main_window.py                 # MainWindow with sidebar, top bar, & navigation
│   ├── floating_assistant.py          # Compact floating AI companion
│   ├── system_tray.py                 # Windows notification tray icon
│   ├── styles.py                      # QSS stylesheet & dark design system
│   └── views/                         # Modular UI views
│       ├── chat_view.py               # Interactive Chat with streaming AI & proposal banners
│       ├── ai_engine_view.py          # AI Engine management, diagnostics & key configuration
│       ├── history_view.py            # Conversation history browser
│       ├── personal_memory_view.py    # Isolated Personal Memory browser & proposal approval
│       ├── work_memory_view.py        # Isolated Work Memory browser & project organizer
│       ├── voice_view.py              # Voice pipeline state visualizer
│       ├── windows_control_view.py    # Windows Computer Control dashboard & command center
│       └── settings_view.py           # Application settings, storage migration, backups & themes
│
├── windows_control/                   # Phase 7 Windows Computer Control & Discovery
│   ├── __init__.py                    # Public exports
│   ├── models.py                      # ActionPlan, CommandResult, HardwareProfile, SafetyCheck, SearchResult
│   ├── hardware_profiler.py           # Genuine hardware profiling & operation capacity evaluation
│   ├── resource_discovery.py          # System-wide discovery (drives, user dirs, tools, runtimes, apps)
│   ├── filesystem_service.py          # Multi-drive operations, permission-aware traversal, Recycle Bin
│   ├── manager.py                     # WindowsControlManager central coordinator
│   ├── command_executor.py            # Async subprocess executor with streaming callbacks
│   ├── elevation_manager.py           # Windows UAC elevation via runas & Named Pipe IPC
│   ├── execution_policy.py            # Risk tiers (LOW, SENSITIVE, DESTRUCTIVE, BLOCKED)
│   ├── command_interpreter.py         # Bilingual (English & Tamil) natural language command interpreter
│   ├── audit_logger.py                # Immutable redacted audit log in settings.db
│   ├── process_manager.py             # Process enumeration, termination, and critical OS protection
│   ├── application_manager.py         # Application registry lookups and launcher
│   ├── software_manager.py            # Winget, pip, and installer command builders
│   ├── download_manager.py            # Multi-threaded file downloads with SHA-256 verification
│   ├── gui_automation.py              # Native pywinauto / ctypes desktop GUI automation
│   ├── cmd_service.py                 # Streaming CMD shell service
│   ├── powershell_service.py          # Streaming PowerShell 5.1/7+ shell service
│   └── system_manager.py              # Network ping, port test, services, and hardware specs
│
├── tests/                             # Automated test suite (161 tests)
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
│   ├── test_views.py                  # View navigation, AIEngineView controls & state tests
│   ├── test_memory_encryption.py      # AES-256 encryption at rest, JSON & file encryption tests
│   ├── test_memory_vector_index.py    # Vector indexing, cosine similarity, duplicate detection tests
│   ├── test_memory_proposal_detector.py # Automatic pattern detection & scoring tests
│   ├── test_memory_proposal_workflow.py # Approval/rejection workflow & minimal context builder tests
│   ├── test_personal_and_work_memory_managers.py # CRUD, domain isolation, encrypted backup/restore tests
│   ├── test_storage_architecture.py   # Persistent OS storage, domain exports, wiping, migration tests
│   ├── test_privacy_and_routing.py    # Phase 5 hybrid router, classifiers, consent & redactor tests
│   ├── test_voice_architecture.py     # Phase 6 VAD, wake word, Faster-Whisper, TTS router, & continuous engine tests
│   ├── test_windows_control.py        # Phase 7 full control, UAC elevation, shell, & audit tests (19 tests)
│   └── test_system_wide_windows_control.py # System-wide discovery, hardware profiler & capacity tests (16 tests)
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
- **Google Gemini API** (`gemini-2.5-flash`, `gemini-2.5-pro`)
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

### 3. Running the Test Suite (126 Tests)
```powershell
.\.venv\Scripts\pytest.exe
```
Output:
```
============================= 126 passed in 19.55s =============================
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

## Continuous Voice Assistant & Voice Controls

HAKS includes a bilingual, continuous voice assistant pipeline with full barge-in interruption capability.

### Voice State Flow
```
[STANDBY] ──("Hey HAKS" or Push-to-Talk)──> [WAKE_DETECTED] ──> [LISTENING] ──(VAD Speech End)──> [PROCESSING] ──(Faster-Whisper STT)──> [THINKING] ──(AI & Memory)──> [SPEAKING] ──(Natural End)──> [LISTENING] (Continuous Mode)
                                                                                                                          │
                                                                                                    (User Interrupts) ────┴──> [INTERRUPTED] ──> [LISTENING]
```

### Voice Commands (English & Tamil)
| Command | Tamil Equivalent | Action |
|---|---|---|
| **"Stop HAKS"** / **"Stop"** | **"நிறுத்து"** / **"நில்"** | Stops current conversation and enters STANDBY. |
| **"Sleep HAKS"** | **"தூங்கு"** / **"ஓய்வெடு"** | Enters low-power SLEEP state. |
| **"Wake up HAKS"** | **"எழுந்திரு"** / **"விழி"** | Wakes assistant from SLEEP to STANDBY. |
| **"Cancel"** | **"ரத்து செய்"** / **"வேண்டாம்"** | Aborts current voice turn immediately. |
| **"Repeat"** | **"மீண்டும் சொல்"** / **"மறுபடியும் சொல்"** | Re-synthesizes and plays the last response. |
| **"Stop speaking"** | **"பேசாதே"** / **"பேசுவதை நிறுத்து"** | Instantly silences audio playback and listens. |

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
| **Microphone not detected** | Microphone permissions or default input device disconnected. | Open HAKS UI -> Voice -> select device from "Audio Input Device" dropdown. |

---

## Release Status & Roadmap

- [x] **Phase 1 — Foundation** (Configuration, Logging, SQLite WAL Databases, Basic PySide6 UI)
- [x] **Phase 2 — Desktop UI** (Unified AppController, ChatView, Floating Assistant, History, Themes)
- [x] **Phase 3 — AI Engine Integration** (Ollama Local AI, Cloud AI Provider Adapter, Non-Blocking Streaming, Windows Credential Manager, Cancellation)
- [x] **Phase 4 — Memory Architecture** (Dual isolated memory domains, local vector search, proposal approval workflow, encryption at rest, backup/restore/export, safe storage migration)
- [x] **Phase 5 — Intelligent AI Routing & Privacy Management** (Hybrid rule + lightweight classifier, deterministic privacy policies, consent manager, data redactor, fallback confirmation)
- [x] **Phase 6 — Advanced Bilingual Voice Assistant** (Real-time sounddevice capture, dynamic VAD, "Hey HAKS" wake word, Faster-Whisper CPU int8 STT, multi-engine Edge/SAPI TTS router, barge-in interruption, continuous state machine)
- [ ] **Phase 7 — Windows Computer Control & Security** (OS actions, 4-tier risk classification, audit logs)
- [ ] **Phase 8 — Offline & Disaster Recovery** (Automated backups, integrity validation)
- [ ] **Phase 9 — Packaging & Release 0.1** (PyInstaller packaging, final verification)
