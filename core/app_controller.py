"""Central Application Controller for HAKS.

Coordinates UI views, conversation state, memory domains, voice state,
and window persistence through a unified controller.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import uuid

from config.settings import AppConfig, get_settings
from core.events import ApplicationEvents, get_app_events
from core.logger import get_logger
from memory.context_builder import MemoryContextBuilder
from memory.models import MemoryDomain, MemoryProposal
from memory.personal_memory import PersonalMemoryManager
from memory.proposal_detector import MemoryProposalDetector
from memory.proposal_manager import MemoryProposalManager
from memory.work_memory import WorkMemoryManager
from storage.backup_manager import BackupManager
from storage.database import DatabaseManager, get_db_manager
from storage.migration import StorageMigrationService
from privacy.classifier import get_privacy_classifier
from privacy.redactor import get_data_redactor
from privacy.consent_manager import get_consent_manager
from routing.complexity import get_complexity_classifier
from routing.models import RoutingDecision, RoutingMode

if TYPE_CHECKING:
    from ai.manager import AIEngineManager
    from ai.models import AIMessage

logger = get_logger("core.app_controller")


class AppController:
    """Unified application controller shared by desktop window and floating assistant."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        events: Optional[ApplicationEvents] = None,
    ) -> None:
        self.config = config or get_settings()
        self.db = db_manager or get_db_manager(self.config)
        self.events = events or get_app_events()

        self._active_conversation_id: Optional[str] = None
        self._voice_state: str = "STANDBY"
        self._is_mic_muted: bool = False
        self._current_theme: str = self.get_setting("theme", self.config.ui.theme)
        self._ai_manager: Optional[Any] = None

        # Phase 4 Memory Architecture Subsystems
        self.personal_memory = PersonalMemoryManager(self.config, self.db)
        self.work_memory = WorkMemoryManager(self.config, self.db)
        self.proposal_detector = MemoryProposalDetector()
        self.proposal_manager = MemoryProposalManager(
            self.config, self.db, self.personal_memory, self.work_memory
        )
        self.context_builder = MemoryContextBuilder(self.personal_memory, self.work_memory)
        self.backup_manager = BackupManager(self.config, self.db, self.personal_memory.encryption)
        self.migration_service = StorageMigrationService(self.config, self.db)

        # Phase 5 Privacy & Intelligent Routing Subsystems
        self.privacy_classifier = get_privacy_classifier()
        self.complexity_classifier = get_complexity_classifier()
        self.redactor = get_data_redactor()
        self.consent_manager = get_consent_manager()
        self._router: Optional[Any] = None

        # Phase 6 Advanced Voice Assistant Engine
        from voice.conversation_engine import ContinuousVoiceEngine
        self.voice_engine = ContinuousVoiceEngine(controller=self, config=self.config)

        # Phase 7 Windows Computer Control & Administrative Automation
        from windows_control.manager import WindowsControlManager
        self.windows_control = WindowsControlManager(config=self.config, db_manager=self.db)

        # Phase 8 Offline Operation & Disaster Recovery Subsystem
        from recovery.offline_manager import OfflineRecoveryManager
        self.recovery_manager = OfflineRecoveryManager(
            config=self.config,
            db_manager=self.db,
            encryption_service=self.personal_memory.encryption,
            events=self.events,
        )
        self.recovery_manager.initialize_on_startup()

        logger.info("AppController initialized. Current theme: %s", self._current_theme)

    # --------------------------------------------------------------------------
    # Settings Persistence
    # --------------------------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        """Retrieve a stored setting from settings.db."""
        try:
            with self.db.session("settings") as conn:
                row = conn.execute(
                    "SELECT value FROM app_settings WHERE key = ?;", (key,)
                ).fetchone()
                if row:
                    return str(row["value"])
        except Exception as exc:
            logger.warning("Could not read setting '%s': %s", key, exc)
        return default

    def set_setting(self, key: str, value: str, section: str = "general") -> None:
        """Persist a setting into settings.db."""
        try:
            with self.db.session("settings") as conn:
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value, section, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        section = excluded.section,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (key, str(value), section),
                )
        except Exception as exc:
            logger.error("Could not write setting '%s': %s", key, exc)

    def get_floating_position(self) -> tuple[int, int]:
        """Retrieve last saved floating window coordinates."""
        raw_x = self.get_setting("floating_pos_x", "100")
        raw_y = self.get_setting("floating_pos_y", "100")
        try:
            return int(raw_x), int(raw_y)
        except ValueError:
            return 100, 100

    def save_floating_position(self, x: int, y: int) -> None:
        """Save floating window coordinates."""
        self.set_setting("floating_pos_x", str(x), section="ui")
        self.set_setting("floating_pos_y", str(y), section="ui")

    def get_always_on_top(self) -> bool:
        """Get always-on-top preference for floating assistant."""
        val = self.get_setting("floating_always_on_top", "true")
        return val.lower() == "true"

    def set_always_on_top(self, enabled: bool) -> None:
        """Save always-on-top preference for floating assistant."""
        self.set_setting("floating_always_on_top", "true" if enabled else "false", section="ui")

    def get_theme(self) -> str:
        """Get current active UI theme name ('dark' or 'light')."""
        return self._current_theme

    def set_theme(self, theme_name: str) -> None:
        """Update and broadcast active theme."""
        if theme_name not in ("dark", "light"):
            theme_name = "light"
        self._current_theme = theme_name
        self.set_setting("theme", theme_name, section="ui")
        self.events.theme_changed.emit(theme_name)
        logger.info("Theme switched to: %s", theme_name)

    # --------------------------------------------------------------------------
    # Conversation Management
    # --------------------------------------------------------------------------

    def get_active_conversation_id(self) -> str:
        """Return the current active conversation ID, creating one if needed."""
        if not self._active_conversation_id:
            # Check if there is an existing recent conversation
            try:
                with self.db.session("conversations") as conn:
                    row = conn.execute(
                        "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1;"
                    ).fetchone()
                    if row:
                        self._active_conversation_id = str(row["id"])
            except Exception as exc:
                logger.warning("Error fetching most recent conversation: %s", exc)

            if not self._active_conversation_id:
                self.create_new_conversation()

        assert self._active_conversation_id is not None
        return self._active_conversation_id

    def create_new_conversation(self, title: str = "New Conversation") -> str:
        """Create a new conversation session."""
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        with self.db.session("conversations") as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, model_used, language)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (conv_id, title, now, now, self.config.ai.local_model, self.config.voice.preferred_language),
            )

        self._active_conversation_id = conv_id
        self.events.conversation_created.emit(conv_id, title)
        self.events.conversation_list_updated.emit()
        logger.info("Created new conversation: %s ('%s')", conv_id, title)
        return conv_id

    def load_conversation(self, conv_id: str, emit_event: bool = False) -> list[dict]:
        """Load messages for a specific conversation session."""
        messages: list[dict] = []
        with self.db.session("conversations") as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC;
                """,
                (conv_id,),
            ).fetchall()

            for r in rows:
                messages.append({
                    "id": str(r["id"]),
                    "conversation_id": conv_id,
                    "role": str(r["role"]),
                    "content": str(r["content"]),
                    "timestamp": str(r["timestamp"]),
                    "metadata": str(r["metadata"] or ""),
                })

        self._active_conversation_id = conv_id
        if emit_event:
            self.events.conversation_loaded.emit(conv_id)
        logger.info("Loaded conversation: %s (%d messages)", conv_id, len(messages))
        return messages

    def select_conversation(self, conv_id: str) -> list[dict]:
        """Activate a conversation and notify UI listeners."""
        return self.load_conversation(conv_id, emit_event=True)

    def get_all_conversations(self) -> list[dict]:
        """Fetch list of all conversations ordered by recent activity."""
        conversations: list[dict] = []
        try:
            with self.db.session("conversations") as conn:
                rows = conn.execute(
                    """
                    SELECT c.id, c.title, c.updated_at, c.model_used,
                           COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC;
                    """
                ).fetchall()

                for r in rows:
                    conversations.append({
                        "id": str(r["id"]),
                        "title": str(r["title"]),
                        "updated_at": str(r["updated_at"]),
                        "model_used": str(r["model_used"] or ""),
                        "message_count": int(r["message_count"]),
                    })
        except Exception as exc:
            logger.error("Error retrieving conversations: %s", exc)
        return conversations

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation and all its messages."""
        with self.db.session("conversations") as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?;", (conv_id,))

        logger.info("Deleted conversation: %s", conv_id)
        if self._active_conversation_id == conv_id:
            self._active_conversation_id = None
            self.create_new_conversation()

        self.events.conversation_list_updated.emit()

    def send_message(self, content: str, role: str = "user", metadata: Optional[dict] = None) -> dict:
        """Record and broadcast a message within the active conversation."""
        conv_id = self.get_active_conversation_id()
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        meta_str = json.dumps(metadata) if metadata else ""

        with self.db.session("conversations") as conn:
            # 1. Insert message
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (msg_id, conv_id, role, content, now, meta_str),
            )

            # 2. Update conversation activity and title if needed
            conv_row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?;", (conv_id,)
            ).fetchone()

            if conv_row and conv_row["title"] == "New Conversation" and role == "user":
                new_title = (content.strip()[:35] + "...") if len(content.strip()) > 35 else content.strip()
                if new_title:
                    conn.execute(
                        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?;",
                        (new_title, now, conv_id),
                    )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                    (now, conv_id),
                )

        msg_data = {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "metadata": metadata or {},
        }

        # Broadcast signal to all UI listeners (Main Window, Floating Assistant)
        self.events.message_received.emit(conv_id, msg_id, role, content)
        self.events.conversation_list_updated.emit()

        # Automatic Memory Proposal Detection (User review required before permanent save)
        if role == "user":
            try:
                proposal = self.proposal_detector.detect_proposal(
                    user_text=content,
                    source_conversation_id=conv_id,
                )
                if proposal:
                    self.proposal_manager.create_proposal(proposal)
                    self.events.memory_proposal_created.emit(proposal.model_dump())
            except Exception as exc:
                logger.error("Error during memory proposal detection: %s", exc)

        return msg_data

    # --------------------------------------------------------------------------
    # AI Engine Operations
    # --------------------------------------------------------------------------

    def get_ai_manager(self) -> Any:
        """Return the singleton AIEngineManager instance."""
        if self._ai_manager is None:
            from ai.manager import get_ai_manager
            self._ai_manager = get_ai_manager(self.config)
            saved_engine = self.get_setting("active_ai_engine", self.config.ai.default_engine)
            try:
                self._ai_manager.set_active_engine(saved_engine)
            except Exception as exc:
                logger.warning("Failed setting initial active AI engine '%s': %s", saved_engine, exc)
            # Synchronize stored hardware limits to local engine
            try:
                limits = self.get_local_model_settings()
                if hasattr(self._ai_manager, "local_engine") and hasattr(self._ai_manager.local_engine, "update_hardware_limits"):
                    self._ai_manager.local_engine.update_hardware_limits(**limits)
            except Exception as exc:
                logger.warning("Failed synchronizing local hardware limits: %s", exc)
        return self._ai_manager

    def get_local_model_settings(self) -> dict[str, Any]:
        """Fetch persisted hardware-aware local model limits from settings.db with defaults."""
        raw_ctx = self.get_setting("local_num_ctx", str(getattr(self.config.ai, "local_num_ctx", 2048)))
        raw_predict = self.get_setting("local_num_predict", str(getattr(self.config.ai, "local_num_predict", 128)))
        raw_temp = self.get_setting("local_temperature", str(getattr(self.config.ai, "local_temperature", 0.3)))
        raw_threads = self.get_setting("local_cpu_threads", str(getattr(self.config.ai, "local_cpu_threads", 2)))
        raw_penalty = self.get_setting("local_repeat_penalty", str(getattr(self.config.ai, "local_repeat_penalty", 1.15)))

        try:
            num_ctx = int(raw_ctx)
        except ValueError:
            num_ctx = 2048
        try:
            num_predict = int(raw_predict)
        except ValueError:
            num_predict = 128
        try:
            temperature = float(raw_temp)
        except ValueError:
            temperature = 0.3
        try:
            cpu_threads = int(raw_threads)
        except ValueError:
            cpu_threads = 2
        try:
            repeat_penalty = float(raw_penalty)
        except ValueError:
            repeat_penalty = 1.15

        return {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "cpu_threads": cpu_threads,
            "repeat_penalty": repeat_penalty,
        }

    def set_local_model_settings(
        self,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        cpu_threads: int,
        repeat_penalty: float = 1.15,
    ) -> None:
        """Persist hardware-aware local model limits and sync to local engine."""
        self.set_setting("local_num_ctx", str(num_ctx), section="ai_local")
        self.set_setting("local_num_predict", str(num_predict), section="ai_local")
        self.set_setting("local_temperature", str(temperature), section="ai_local")
        self.set_setting("local_cpu_threads", str(cpu_threads), section="ai_local")
        self.set_setting("local_repeat_penalty", str(repeat_penalty), section="ai_local")

        # Synchronize with active AI manager's local engine
        if hasattr(self, "_ai_manager") and self._ai_manager is not None:
            local_engine = getattr(self._ai_manager, "local_engine", None)
            if local_engine and hasattr(local_engine, "update_hardware_limits"):
                local_engine.update_hardware_limits(
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    temperature=temperature,
                    cpu_threads=cpu_threads,
                    repeat_penalty=repeat_penalty,
                )

        logger.info(
            "Persisted local model settings: num_ctx=%d, num_predict=%d, temp=%.2f, threads=%d, penalty=%.2f",
            num_ctx, num_predict, temperature, cpu_threads, repeat_penalty
        )


    def get_selected_ai_engine(self) -> str:
        """Get the active AI engine ('local' or 'cloud'). Defaults to 'cloud' (Gemini)."""
        engine = self.get_setting("active_ai_engine", getattr(self.config.ai, "default_engine", "cloud"))
        if not engine or engine not in ("local", "cloud"):
            return "cloud"
        return engine

    def set_selected_ai_engine(self, engine_type: str) -> None:
        """Set and persist active AI engine ('local' or 'cloud')."""
        norm = engine_type.lower().strip()
        if norm not in ("local", "cloud"):
            norm = "local"
        self.set_setting("active_ai_engine", norm, section="ai")
        mgr = self.get_ai_manager()
        mgr.set_active_engine(norm)
        self.events.ai_engine_changed.emit(norm)
        logger.info("Active AI engine persisted as: %s", norm)

    def get_router(self) -> Any:
        """Return the singleton AIEngineRouter instance."""
        if self._router is None:
            from routing.router import AIEngineRouter
            self._router = AIEngineRouter(
                ai_manager=self.get_ai_manager(),
                privacy_classifier=self.privacy_classifier,
                complexity_classifier=self.complexity_classifier,
                redactor=self.redactor,
                consent_manager=self.consent_manager,
                config=self.config,
            )
        return self._router

    def get_routing_mode(self) -> RoutingMode:
        """Get the active routing mode (AUTO, LOCAL, CLOUD, FORCE_LOCAL)."""
        raw_mode = self.get_setting("routing_mode", "AUTO")
        try:
            return RoutingMode(raw_mode)
        except ValueError:
            return RoutingMode.AUTO

    def set_routing_mode(self, mode: RoutingMode) -> None:
        """Persist active routing mode."""
        self.set_setting("routing_mode", mode.value, section="ai")
        logger.info("Routing mode set to: %s", mode.value)

    def route_request(
        self,
        user_prompt: str,
        mode: Optional[RoutingMode] = None,
        memory_context: str = "",
    ) -> RoutingDecision:
        """Compute an authoritative routing decision using the AIEngineRouter."""
        effective_mode = mode or self.get_routing_mode()
        router = self.get_router()
        is_online = self.recovery_manager.is_online if hasattr(self, "recovery_manager") else True
        return router.route(
            user_prompt=user_prompt,
            mode=effective_mode,
            retrieved_memory_context=memory_context,
            is_online=is_online,
        )

    def get_conversation_ai_messages(
        self,
        conv_id: Optional[str] = None,
        limit: int = 15,
        max_total_tokens: Optional[int] = None,
    ) -> list[Any]:
        """Convert stored conversation messages into typed AIMessage objects with token budgeting.

        - Preserves the current user request and essential action results.
        - Truncates oversized tool/action outputs safely.
        - Removes redundant and consecutive duplicate messages.
        - Enforces hardware-aware token budgets to prevent Ollama token-limit errors.
        """
        from ai.models import AIMessage, estimate_tokens
        target_id = conv_id or self.get_active_conversation_id()
        raw_msgs = self.load_conversation(target_id, emit_event=False)
        if not raw_msgs:
            return []

        # Target token budget for history: reserve space for system prompt and generation
        local_settings = self.get_local_model_settings()
        num_ctx = local_settings.get("num_ctx", 2048)
        num_predict = local_settings.get("num_predict", 128)
        if max_total_tokens is None:
            max_total_tokens = max(512, num_ctx - num_predict - 350)

        VALID_AI_ROLES = {"user", "assistant", "system"}
        filtered: list[dict] = []
        for m in raw_msgs:
            role = m["role"]
            content = m["content"]
            if role not in VALID_AI_ROLES or not content or not content.strip():
                continue

            meta = m.get("metadata")
            if isinstance(meta, str) and meta.strip():
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}

            if meta.get("error") or content.startswith("I encountered an error") or content.startswith("⚠️"):
                continue

            filtered.append(m)

        if not filtered:
            return []

        # Always identify the latest user request index
        latest_user_idx = -1
        for i in range(len(filtered) - 1, -1, -1):
            if filtered[i]["role"] == "user":
                latest_user_idx = i
                break

        # Process candidates working backwards from the latest up to limit & token budget
        slice_start = max(0, len(filtered) - limit)
        candidates = filtered[slice_start:]

        selected: list[AIMessage] = []
        current_token_sum = 0

        for i in range(len(candidates) - 1, -1, -1):
            cand = candidates[i]
            role = cand["role"]
            content = cand["content"].strip()
            is_current_user = (slice_start + i == latest_user_idx)

            # Truncate oversized tool/action outputs safely
            if len(content) > 400 and (role == "assistant" or "Output:" in content or "```" in content):
                lines = content.splitlines()
                if len(lines) > 10:
                    summary_head = "\n".join(lines[:6])
                    tail = "\n".join(lines[-3:])
                    content = f"{summary_head}\n[... output truncated for context limit ...]\n{tail}"
                elif len(content) > 500:
                    content = content[:300] + "\n[... output truncated ...]\n" + content[-150:]

            msg_toks = estimate_tokens(content) + 4

            # If adding this message exceeds token budget, and this is NOT the current user message, stop
            if not is_current_user and (current_token_sum + msg_toks > max_total_tokens) and selected:
                logger.debug("Token budget reached (%d tokens), skipping older message", current_token_sum)
                continue

            # Deduplicate consecutive identical messages
            if selected and selected[0].role == role and selected[0].content == content:
                continue

            selected.insert(0, AIMessage(role=role, content=content, timestamp=cand["timestamp"]))
            current_token_sum += msg_toks

        # Safeguard: Verify the current user request was never discarded
        if latest_user_idx != -1:
            latest_user_cand = filtered[latest_user_idx]
            has_current_user = any(
                m.role == "user" and m.timestamp == latest_user_cand["timestamp"]
                for m in selected
            )
            if not has_current_user:
                u_content = latest_user_cand["content"].strip()
                if len(u_content) > 1000:
                    u_content = u_content[:500] + "\n[... truncated ...]\n" + u_content[-300:]
                selected.append(AIMessage(
                    role="user",
                    content=u_content,
                    timestamp=latest_user_cand["timestamp"],
                ))

        return selected


    # --------------------------------------------------------------------------
    # Personal Memory Operations
    # --------------------------------------------------------------------------

    def get_personal_memories(self, search: str = "", category: Optional[str] = None) -> list[dict]:
        """Fetch records from personal_memory.db via PersonalMemoryManager."""
        items = self.personal_memory.get_all_memories(search=search, category=category)
        return [
            {
                "id": m.id,
                "title": m.title,
                "content": m.content,
                "category": m.category,
                "tags": ", ".join(m.tags) if isinstance(m.tags, list) else str(m.tags or ""),
                "created_at": m.created_at,
                "importance": m.importance,
                "source_conversation_id": m.source_conversation_id,
            }
            for m in items
        ]

    def add_personal_memory(
        self,
        title: str,
        content: str,
        category: str = "general",
        tags: str = "",
        importance: int = 1,
        source_conversation_id: Optional[str] = None,
    ) -> str:
        """Add a persistent record to Personal Memory."""
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) else (tags or [])
        mem_id, is_new = self.personal_memory.add_memory(
            title=title,
            content=content,
            category=category,
            tags=tags_list,
            importance=importance,
            source_conversation_id=source_conversation_id,
        )
        self.events.memory_domain_updated.emit("personal")
        return mem_id

    def update_personal_memory(
        self,
        memory_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> bool:
        """Update an existing personal memory record."""
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) else None
        updated = self.personal_memory.update_memory(
            memory_id=memory_id,
            title=title,
            content=content,
            category=category,
            tags=tags_list,
            importance=importance,
        )
        if updated:
            self.events.memory_domain_updated.emit("personal")
        return updated

    def delete_personal_memory(self, memory_id: str) -> None:
        """Delete an individual record from Personal Memory."""
        self.personal_memory.delete_memory(memory_id)
        self.events.memory_domain_updated.emit("personal")

    def clear_personal_memory(self) -> None:
        """Clear Personal Memory completely (never touches Work Memory)."""
        self.personal_memory.clear_domain()
        self.events.memory_domain_updated.emit("personal")

    # --------------------------------------------------------------------------
    # Work Memory Operations
    # --------------------------------------------------------------------------

    def get_work_memories(self, project: str = "", search: str = "") -> list[dict]:
        """Fetch records from work_memory.db via WorkMemoryManager."""
        items = self.work_memory.get_all_memories(project_name=project, search=search)
        return [
            {
                "id": m.id,
                "project_name": m.project_name,
                "title": m.title,
                "content": m.content,
                "tags": ", ".join(m.tags) if isinstance(m.tags, list) else str(m.tags or ""),
                "created_at": m.created_at,
                "importance": m.importance,
                "source_conversation_id": m.source_conversation_id,
            }
            for m in items
        ]

    def add_work_memory(
        self,
        title: str,
        content: str,
        project_name: str = "default",
        tags: str = "",
        importance: int = 1,
        source_conversation_id: Optional[str] = None,
    ) -> str:
        """Add a persistent record to Work Memory."""
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) else (tags or [])
        mem_id, is_new = self.work_memory.add_memory(
            title=title,
            content=content,
            project_name=project_name,
            tags=tags_list,
            importance=importance,
            source_conversation_id=source_conversation_id,
        )
        self.events.memory_domain_updated.emit("work")
        return mem_id

    def update_work_memory(
        self,
        memory_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> bool:
        """Update an existing work memory record."""
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) else None
        updated = self.work_memory.update_memory(
            memory_id=memory_id,
            title=title,
            content=content,
            project_name=project_name,
            tags=tags_list,
            importance=importance,
        )
        if updated:
            self.events.memory_domain_updated.emit("work")
        return updated

    def delete_work_memory(self, memory_id: str) -> None:
        """Delete an individual record from Work Memory."""
        self.work_memory.delete_memory(memory_id)
        self.events.memory_domain_updated.emit("work")

    def clear_work_memory(self) -> None:
        """Clear Work Memory completely (never touches Personal Memory)."""
        self.work_memory.clear_domain()
        self.events.memory_domain_updated.emit("work")

    def get_work_projects(self) -> list[str]:
        """Fetch all unique project names from Work Memory."""
        return self.work_memory.get_projects()

    # --------------------------------------------------------------------------
    # Memory Proposal Workflow Operations
    # --------------------------------------------------------------------------

    def get_pending_proposals(self) -> list[dict]:
        """Fetch all memory proposals awaiting user approval."""
        proposals = self.proposal_manager.get_pending_proposals()
        return [p.model_dump() for p in proposals]

    def approve_proposal(
        self,
        proposal_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: int = 2,
    ) -> tuple[bool, Optional[str]]:
        """Approve a memory proposal, persisting it to Personal or Work memory."""
        success, mem_id = self.proposal_manager.approve_proposal(
            proposal_id=proposal_id,
            title=title,
            content=content,
            category=category,
            project_name=project_name,
            tags=tags,
            importance=importance,
        )
        if success:
            prop = self.proposal_manager.get_proposal(proposal_id)
            if prop:
                self.events.memory_domain_updated.emit(prop.domain.value)
        return success, mem_id

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a memory proposal, discarding it without saving."""
        success = self.proposal_manager.reject_proposal(proposal_id)
        if success:
            self.events.memory_domain_updated.emit("proposals")
        return success

    # --------------------------------------------------------------------------
    # Memory Export, Backup & Restore
    # --------------------------------------------------------------------------

    def export_memory_domain(self, domain: str, file_path: Path, encrypt: bool = False) -> Path:
        """Export a domain (personal or work) to a backup file."""
        if domain == "personal":
            return self.personal_memory.export_bundle(file_path, encrypt=encrypt)
        elif domain == "work":
            return self.work_memory.export_bundle(file_path, encrypt=encrypt)
        raise ValueError(f"Unknown domain for export: {domain}")

    def restore_memory_domain(self, domain: str, file_path: Path, is_encrypted: bool = False) -> int:
        """Restore a domain (personal or work) from a backup bundle."""
        if domain == "personal":
            count = self.personal_memory.restore_bundle(file_path, is_encrypted=is_encrypted)
            self.events.memory_domain_updated.emit("personal")
            return count
        elif domain == "work":
            count = self.work_memory.restore_bundle(file_path, is_encrypted=is_encrypted)
            self.events.memory_domain_updated.emit("work")
            return count
        raise ValueError(f"Unknown domain for restore: {domain}")
        self.db.clear_domain("work_memory")
        self.events.memory_domain_updated.emit("work")

    # --------------------------------------------------------------------------
    # Voice & Audio State
    # --------------------------------------------------------------------------

    def get_voice_state(self) -> str:
        """Return current voice pipeline state."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            return self.voice_engine.state.value
        return self._voice_state

    def set_voice_state(self, state: str) -> None:
        """Update and broadcast voice state."""
        self._voice_state = state
        if hasattr(self, "voice_engine") and self.voice_engine:
            try:
                from voice.models import VoiceState
                self.voice_engine._set_state(VoiceState(state))
            except Exception:
                pass
        self.events.voice_state_changed.emit(state)
        logger.info("Voice state changed to: %s", state)

    def is_mic_muted(self) -> bool:
        """Return microphone mute status."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            return self.voice_engine.audio_capture.is_muted
        return self._is_mic_muted

    def toggle_mic_mute(self) -> bool:
        """Toggle microphone mute and emit state."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self._is_mic_muted = self.voice_engine.toggle_mute()
        else:
            self._is_mic_muted = not self._is_mic_muted
            self.events.voice_mute_toggled.emit(self._is_mic_muted)
        logger.info("Microphone mute set to: %s", self._is_mic_muted)
        return self._is_mic_muted

    def start_voice_assistant(self) -> bool:
        """Start continuous voice assistant capture."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            return self.voice_engine.start()
        return False

    def stop_voice_assistant(self) -> None:
        """Stop voice assistant capture."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.stop()

    def trigger_push_to_talk(self) -> None:
        """Trigger push-to-talk in voice engine."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.trigger_push_to_talk()

    def interrupt_voice(self) -> None:
        """Interrupt voice playback (barge-in)."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine._handle_barge_in_interruption()

    def get_voice_mode(self) -> str:
        """Return active voice mode ('gemini_live', 'pipeline', 'offline')."""
        saved = self.get_setting("voice_mode", self.config.voice.voice_mode)
        return saved

    def set_voice_mode(self, mode: str) -> None:
        """Update and persist active voice mode."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.set_voice_mode(mode)
        else:
            self.config.voice.voice_mode = mode
            self.set_setting("voice_mode", mode, section="voice")

    def set_noise_suppression_enabled(self, enabled: bool) -> None:
        """Enable or disable real-time background noise suppression."""
        self.config.voice.enable_noise_suppression = enabled
        self.set_setting("enable_noise_suppression", str(enabled).lower(), section="voice")
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.pipeline.enable_noise_suppression = enabled

    def set_echo_cancellation_enabled(self, enabled: bool) -> None:
        """Enable or disable acoustic echo cancellation."""
        self.config.voice.enable_echo_cancellation = enabled
        self.set_setting("enable_echo_cancellation", str(enabled).lower(), section="voice")
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.pipeline.enable_aec = enabled

    def set_noise_suppression_strength(self, strength: str) -> None:
        """Configure noise suppression aggressiveness ('low', 'medium', 'high', 'ultra')."""
        self.config.voice.noise_suppression_strength = strength
        self.set_setting("noise_suppression_strength", strength, section="voice")
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.pipeline.noise_suppressor.set_strength(strength)

    def set_mic_gain(self, gain: float) -> None:
        """Configure microphone input gain (0.2x to 3.0x)."""
        self.config.voice.mic_input_gain = gain
        self.set_setting("mic_input_gain", str(gain), section="voice")
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.pipeline.dc_normalizer.set_gain(gain)

    def set_human_speech_sensitivity(self, sensitivity: float) -> None:
        """Configure human speech detection sensitivity (0.1 to 1.0)."""
        self.config.voice.human_speech_sensitivity = sensitivity
        self.set_setting("human_speech_sensitivity", str(sensitivity), section="voice")
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.pipeline.vad_detector.set_sensitivity(sensitivity)

    def set_voice_interruption_sensitivity(self, sensitivity: float) -> None:
        """Configure voice interruption (barge-in) sensitivity (0.1 to 1.0)."""
        self.config.voice.voice_interruption_sensitivity = sensitivity
        self.set_setting("voice_interruption_sensitivity", str(sensitivity), section="voice")

    def calibrate_noise_floor(self) -> None:
        """Start dynamic background noise calibration."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.audio_capture.calibrate_noise_floor()

    def finish_noise_calibration(self) -> float:
        """Finish dynamic background noise calibration."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            floor = self.voice_engine.audio_capture.finish_noise_calibration()
            if hasattr(self, "events") and hasattr(self.events, "voice_noise_floor_updated"):
                self.events.voice_noise_floor_updated.emit(floor)
            return floor
        return 0.005

    # --------------------------------------------------------------------------
    # Storage Architecture & Persistence Management
    # --------------------------------------------------------------------------

    def get_storage_path(self) -> Path:
        """Return current persistent data directory."""
        return self.config.paths.data_dir

    def migrate_storage(self, new_path: Path, overwrite_approved: bool = False) -> tuple[bool, str]:
        """Safely migrate data directory to a new location."""
        success, msg = self.migration_service.migrate_data_directory(new_path, overwrite_approved=overwrite_approved)
        if success:
            # Re-initialize memory managers pointing to the new location
            self.personal_memory = PersonalMemoryManager(self.config, self.db)
            self.work_memory = WorkMemoryManager(self.config, self.db)
            self.proposal_manager = MemoryProposalManager(
                self.config, self.db, self.personal_memory, self.work_memory
            )
            self.context_builder = MemoryContextBuilder(self.personal_memory, self.work_memory)
            self.backup_manager = BackupManager(self.config, self.db, self.personal_memory.encryption)
            self.set_setting("storage_data_directory", str(new_path))
            self.events.memory_domain_updated.emit("all")
        return success, msg

    def backup_domain(self, domain: str, encrypt: bool = True) -> Path:
        """Create an independent backup for a domain."""
        return self.backup_manager.backup_domain(domain, encrypt=encrypt)

    def restore_domain(self, backup_path: Path, domain: str) -> bool:
        """Restore an independent domain from backup."""
        success = self.backup_manager.restore_domain(backup_path, domain)
        if success:
            if domain == "personal_memory":
                self.personal_memory = PersonalMemoryManager(self.config, self.db)
                self.events.memory_domain_updated.emit("personal")
            elif domain == "work_memory":
                self.work_memory = WorkMemoryManager(self.config, self.db)
                self.events.memory_domain_updated.emit("work")
            elif domain == "conversations":
                self.events.conversation_created.emit("restored")
        return success

    def export_domain(self, domain: str, export_format: str, output_path: Path) -> Path:
        """Export domain data to Markdown, JSON, or CSV."""
        fmt = export_format.lower()
        if fmt in ("markdown", "md"):
            return self.backup_manager.export_domain_markdown(domain, output_path)
        elif fmt == "csv":
            return self.backup_manager.export_domain_csv(domain, output_path)
        else:
            return self.backup_manager.export_domain_json(domain, output_path, decrypt=True)

    def wipe_domain(self, domain: str, confirm_phrase: str) -> bool:
        """Securely wipe an individual domain with confirmation guard."""
        success = self.backup_manager.wipe_domain(domain, confirm_phrase)
        if success:
            if domain == "personal_memory":
                self.personal_memory.vector_index.clear()
                self.events.memory_domain_updated.emit("personal")
            elif domain == "work_memory":
                self.work_memory.vector_index.clear()
                self.events.memory_domain_updated.emit("work")
        return success

    def sync_vector_indexes(self) -> dict[str, Any]:
        """Perform self-healing vector index synchronization across memory domains."""
        p_sync = self.personal_memory.sync_vector_index()
        w_sync = self.work_memory.sync_vector_index()
        return {
            "personal": p_sync,
            "work": w_sync,
        }

    def get_recovery_key(self) -> str:
        """Export master memory recovery key."""
        return self.personal_memory.encryption.export_recovery_key()

    def recover_encryption_key(self, recovery_key: str) -> bool:
        """Restore master memory encryption cipher using recovery key."""
        return self.personal_memory.encryption.recover_with_recovery_key(recovery_key)

    # --------------------------------------------------------------------------
    # Phase 7 Windows Computer Control Integration
    # --------------------------------------------------------------------------

    def get_windows_control(self) -> Any:
        """Return the WindowsControlManager subsystem."""
        return self.windows_control

    async def reason_windows_action(
        self,
        instruction: str,
        source: str = "chat",
        working_dir: Optional[str] = None,
    ) -> Any:
        """Reason about a natural-language Windows control instruction using Gemini."""
        if not hasattr(self, "windows_control") or self.windows_control is None:
            raise RuntimeError("Windows Control Manager is not available.")
        ai_mgr = self.get_ai_manager()
        return await self.windows_control.reason_with_gemini(
            instruction=instruction,
            ai_manager=ai_mgr,
            source=source,
            working_dir=working_dir,
        )

    async def execute_windows_instruction(
        self,
        instruction: str,
        source: str = "chat",
        working_dir: Optional[str] = None,
        bypass_confirmation: bool = False,
    ) -> Any:
        """Route conversational request through computer control engine."""
        return await self.windows_control.execute_instruction(
            instruction=instruction,
            source=source,
            working_dir=working_dir,
            bypass_confirmation=bypass_confirmation,
        )

    def trigger_emergency_stop(self) -> int:
        """Emergency Stop: immediately abort all active computer control tasks and voice playback."""
        logger.warning("Emergency Stop requested across all HAKS subsystems.")
        count = self.windows_control.emergency_stop()
        if hasattr(self, "voice_engine") and self.voice_engine:
            self.voice_engine.player.stop()
        if hasattr(self, "events") and self.events:
            self.events.windows_emergency_stop_triggered.emit()
        return count

    def get_hardware_profile(self, refresh: bool = False) -> Any:
        """Retrieve complete hardware profile including CPU, RAM, GPU, display, audio, and storage."""
        return self.windows_control.get_complete_hardware_profile(refresh=refresh)

    def evaluate_operation_capacity(
        self,
        operation: str,
        target_path: Optional[str] = None,
        required_disk_gb: float = 0.0,
        required_ram_gb: float = 0.0,
        requires_gpu: bool = False,
    ) -> Any:
        """Pre-flight check before heavy operations (model pull, big build, software download)."""
        return self.windows_control.check_resource_safety(
            operation=operation,
            target_path=target_path,
            required_disk_gb=required_disk_gb,
            required_ram_gb=required_ram_gb,
            requires_gpu=requires_gpu,
        )

    def discover_system_resources(
        self,
        include_apps: bool = True,
        include_processes: bool = True,
    ) -> Any:
        """Discover drives, user folders, dev tools, Python runtimes, Git, and installed apps."""
        return self.windows_control.discover_system_resources(
            include_apps=include_apps,
            include_processes=include_processes,
        )

    def search_system_files(
        self,
        query: str,
        start_dir: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 100,
    ) -> Any:
        """Search files across drives with permission-aware directory tracking."""
        return self.windows_control.search_system_files(
            query=query,
            start_dir=start_dir,
            extension=extension,
            max_results=max_results,
        )

    def list_system_drives(self) -> Any:
        """List all detected Windows drives with capacity and status."""
        return self.windows_control.list_system_drives()

    # --------------------------------------------------------------------------
    # Phase 8 Offline Operation & Disaster Recovery Integration
    # --------------------------------------------------------------------------

    def get_recovery_manager(self) -> Any:
        """Return the OfflineRecoveryManager subsystem."""
        return self.recovery_manager

    def is_online(self) -> bool:
        """Return current network reachability."""
        return self.recovery_manager.is_online

    def create_full_backup(self, encrypt: bool = True, notes: str = "") -> Path:
        """Create a complete full-system backup bundle."""
        return self.recovery_manager.create_full_backup(encrypt=encrypt, notes=notes)

    def restore_full_backup(
        self,
        backup_path: Path,
        target_dir: Optional[Path] = None,
        preserve_current: bool = True,
    ) -> Any:
        """Restore full system from backup with pre-recovery snapshot safeguard."""
        res = self.recovery_manager.restore_system(
            backup_path=backup_path,
            target_dir=target_dir,
            preserve_current=preserve_current,
        )
        if res.success and target_dir is None:
            # Refresh internal domain managers
            self.personal_memory = PersonalMemoryManager(self.config, self.db)
            self.work_memory = WorkMemoryManager(self.config, self.db)
            self.events.memory_domain_updated.emit("all")
        return res

    def run_system_integrity_check(self) -> dict[str, Any]:
        """Perform comprehensive SQLite integrity verification across all 5 databases."""
        return self.recovery_manager.run_integrity_check()

    def rebuild_all_vector_indexes(self) -> dict[str, Any]:
        """Verify and self-heal memory vector indexes from SQLite records."""
        res = self.recovery_manager.rebuild_vector_indexes()
        self.events.memory_domain_updated.emit("all")
        return res

    def get_system_health_overview(self) -> Any:
        """Return comprehensive health summary for UI."""
        return self.recovery_manager.get_system_health_overview()

    def shutdown(self) -> None:
        """Clean shutdown across all HAKS subsystems."""
        if hasattr(self, "voice_engine") and self.voice_engine:
            try:
                self.voice_engine.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down voice engine: %s", exc)
        if hasattr(self, "recovery_manager") and self.recovery_manager:
            try:
                self.recovery_manager.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down recovery manager: %s", exc)
        if hasattr(self, "windows_control") and self.windows_control:
            try:
                self.windows_control.emergency_stop()
            except Exception as exc:
                logger.warning("Error stopping windows control during shutdown: %s", exc)


_controller_instance: Optional[AppController] = None


def get_app_controller(config: Optional[AppConfig] = None) -> AppController:
    """Return singleton AppController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = AppController(config)
    return _controller_instance
