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

---

## System Architecture

HAKS uses a **Modular Monolithic Architecture** with strict layer separation:

1. **Presentation Layer**: PySide6 / Qt 6 desktop application and companion floating interface.
2. **Application Layer**: Application controllers, event bus, and signal dispatchers.
3. **AI Intelligence Layer**: Hybrid Local (Ollama) and Cloud AI adapter with rule-based and local routing.
4. **Memory, Voice & Security Layer**:
   - **Personal Memory**: Personal history, goals, preferences, and moments.
   - **Work Memory**: Technical discussions, projects, architecture, and code context.
   - **Voice Intelligence**: Wake-word ("Hey HAKS"), continuous Tamil/English conversation, VAD, and TTS.
   - **Security**: 4-level risk assessment, least privilege, audit logging, and consent manager.
5. **Infrastructure & Storage**: 5 independent SQLite databases operating in WAL mode with foreign-key enforcement.
6. **Windows OS Integration**: Permission-gated Windows API adapters.

---

## Phase 1 — Foundation (Current Release: v0.1.0)

Phase 1 provides the core infrastructure:

- **Environment & Configuration**: Typed configuration management using Pydantic, supporting dynamic path discovery and environment overrides.
- **Structured Logging**: Rotating file logging (`logs/haks.log`) with thread information, formatting, and console mirroring.
- **Event Bus**: Decoupled, thread-safe PySide6 signal bus (`core/events.py`) preventing worker-thread UI lockups.
- **Storage Layer**: 5 isolated SQLite databases initialized with WAL mode, foreign keys, and independent domain clearing:
  - `data/personal_memory.db`
  - `data/work_memory.db`
  - `data/conversations.db`
  - `data/memory_proposals.db`
  - `data/settings.db`
- **PySide6 Desktop Application**: Responsive dark-themed desktop UI with high-DPI scaling, 8-section navigation sidebar, system dashboard, engine status badges, and status bar.

---

## Directory Structure

```
HAKS/
│
├── app.py                      # Main application entry point & CLI runner
├── pyproject.toml              # Project packaging & tool configuration
├── requirements.txt            # Pinned runtime dependencies
├── README.md                   # System documentation & manual
├── .gitignore                  # Git ignore rules
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py             # Pydantic AppConfig, paths, database, & UI settings
│
├── core/                       # Core system services & primitives
│   ├── __init__.py
│   ├── logger.py               # Rotating file & console logging
│   └── events.py               # PySide6 signal bus (ApplicationEvents)
│
├── storage/                    # SQLite database framework
│   ├── __init__.py
│   ├── database.py             # Connection management, WAL mode, transaction sessions
│   └── schema.py               # Schemas for the 5 independent databases
│
├── ui/                         # Desktop presentation layer
│   ├── __init__.py
│   ├── app.py                  # HaksApplication Qt lifecycle manager
│   ├── main_window.py          # MainWindow with sidebar, top bar, & dashboard
│   └── styles.py               # QSS stylesheet & dark design system
│
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures (headless Qt & isolated storage)
│   ├── test_config.py          # Configuration tests
│   ├── test_logger.py          # Logger tests
│   ├── test_database.py        # Database schema, WAL mode, & isolation tests
│   └── test_ui_foundation.py   # UI instantiation & navigation tests
│
├── logs/                       # Application log files (git-ignored)
└── data/                       # SQLite databases (git-ignored)
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- Git for Windows

### 2. Environment Setup
```powershell
# Navigate to project directory
cd "c:\Users\Admin\Music\HAKS AI"

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install / verify dependencies
pip install -r requirements.txt
```

### 3. Running the Test Suite
```powershell
.\.venv\Scripts\pytest.exe
```
Output:
```
============================== 9 passed in 0.92s ==============================
```

### 4. Running the Smoke Test (Headless CLI Verification)
```powershell
.\.venv\Scripts\python.exe app.py --smoke-test
```

### 5. Launching the Desktop Application
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## Development Execution Plan

- [x] **Phase 1 — Foundation** (Completed & Verified)
- [ ] **Phase 2 — Desktop UI** (Sidebar, Chat Interface, Floating Assistant, System Tray, Themes)
- [ ] **Phase 3 — AI Brain** (Ollama Local Runtime, Cloud Adapter, AI Router, Fallback Approval)
- [ ] **Phase 4 — Memory** (Personal & Work Memory managers, proposals, retrieval, encryption)
- [ ] **Phase 5 — Voice** (Wake-word "Hey HAKS", VAD, Tamil/English STT/TTS, continuous conversation)
- [ ] **Phase 6 — Security** (Windows permission risk levels, consent manager, audit logs)
- [ ] **Phase 7 — Offline & Recovery** (Offline fallback, backup, restore)
- [ ] **Phase 8 — Testing & Packaging** (System stability, packaging, and Release 0.1)
