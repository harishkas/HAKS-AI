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
from storage.database import DatabaseManager, get_db_manager

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
            theme_name = "dark"
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
        return self._ai_manager

    def get_selected_ai_engine(self) -> str:
        """Get the active AI engine ('local' or 'cloud')."""
        return self.get_setting("active_ai_engine", self.config.ai.default_engine)

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

    def get_conversation_ai_messages(self, conv_id: Optional[str] = None, limit: int = 20) -> list[Any]:
        """Convert stored conversation messages into typed AIMessage objects for AI context.

        Privacy guarantee: Sends ONLY conversation messages.
        Does NOT automatically append Personal or Work Memory databases.
        """
        from ai.models import AIMessage
        target_id = conv_id or self.get_active_conversation_id()
        raw_msgs = self.load_conversation(target_id, emit_event=False)
        recent = raw_msgs[-limit:] if len(raw_msgs) > limit else raw_msgs
        ai_msgs = []
        for m in recent:
            ai_msgs.append(
                AIMessage(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m["timestamp"],
                )
            )
        return ai_msgs

    # --------------------------------------------------------------------------
    # Personal Memory Operations
    # --------------------------------------------------------------------------

    def get_personal_memories(self, search: str = "") -> list[dict]:
        """Fetch records from personal_memory.db."""
        memories = []
        try:
            with self.db.session("personal_memory") as conn:
                query = """
                    SELECT id, title, content, category, tags, created_at, importance
                    FROM personal_memories
                    WHERE is_archived = 0
                """
                params = []
                if search:
                    query += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
                    term = f"%{search}%"
                    params.extend([term, term, term])
                query += " ORDER BY created_at DESC;"

                for r in conn.execute(query, params).fetchall():
                    memories.append({
                        "id": str(r["id"]),
                        "title": str(r["title"]),
                        "content": str(r["content"]),
                        "category": str(r["category"] or "general"),
                        "tags": str(r["tags"] or ""),
                        "created_at": str(r["created_at"]),
                        "importance": int(r["importance"]),
                    })
        except Exception as exc:
            logger.error("Error fetching personal memories: %s", exc)
        return memories

    def add_personal_memory(
        self, title: str, content: str, category: str = "general", tags: str = "", importance: int = 1
    ) -> str:
        """Add a persistent record to Personal Memory."""
        mem_id = f"pm_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        with self.db.session("personal_memory") as conn:
            conn.execute(
                """
                INSERT INTO personal_memories (id, title, content, category, tags, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (mem_id, title, content, category, tags, importance, now, now),
            )

        self.events.memory_domain_updated.emit("personal")
        logger.info("Added Personal Memory: %s ('%s')", mem_id, title)
        return mem_id

    def delete_personal_memory(self, memory_id: str) -> None:
        """Delete an individual record from Personal Memory."""
        with self.db.session("personal_memory") as conn:
            conn.execute("DELETE FROM personal_memories WHERE id = ?;", (memory_id,))

        self.events.memory_domain_updated.emit("personal")
        logger.info("Deleted Personal Memory record: %s", memory_id)

    def clear_personal_memory(self) -> None:
        """Clear Personal Memory completely (never touches Work Memory)."""
        self.db.clear_domain("personal_memory")
        self.events.memory_domain_updated.emit("personal")

    # --------------------------------------------------------------------------
    # Work Memory Operations
    # --------------------------------------------------------------------------

    def get_work_memories(self, project: str = "", search: str = "") -> list[dict]:
        """Fetch records from work_memory.db."""
        memories = []
        try:
            with self.db.session("work_memory") as conn:
                query = """
                    SELECT id, project_name, title, content, tags, created_at, importance
                    FROM work_memories
                    WHERE is_archived = 0
                """
                params = []
                if project:
                    query += " AND project_name = ?"
                    params.append(project)
                if search:
                    query += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
                    term = f"%{search}%"
                    params.extend([term, term, term])
                query += " ORDER BY created_at DESC;"

                for r in conn.execute(query, params).fetchall():
                    memories.append({
                        "id": str(r["id"]),
                        "project_name": str(r["project_name"]),
                        "title": str(r["title"]),
                        "content": str(r["content"]),
                        "tags": str(r["tags"] or ""),
                        "created_at": str(r["created_at"]),
                        "importance": int(r["importance"]),
                    })
        except Exception as exc:
            logger.error("Error fetching work memories: %s", exc)
        return memories

    def add_work_memory(
        self, title: str, content: str, project_name: str = "default", tags: str = "", importance: int = 1
    ) -> str:
        """Add a persistent record to Work Memory."""
        mem_id = f"wm_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        with self.db.session("work_memory") as conn:
            conn.execute(
                """
                INSERT INTO work_memories (id, project_name, title, content, tags, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (mem_id, project_name, title, content, tags, importance, now, now),
            )

        self.events.memory_domain_updated.emit("work")
        logger.info("Added Work Memory: %s ('%s' under '%s')", mem_id, title, project_name)
        return mem_id

    def delete_work_memory(self, memory_id: str) -> None:
        """Delete an individual record from Work Memory."""
        with self.db.session("work_memory") as conn:
            conn.execute("DELETE FROM work_memories WHERE id = ?;", (memory_id,))

        self.events.memory_domain_updated.emit("work")
        logger.info("Deleted Work Memory record: %s", memory_id)

    def clear_work_memory(self) -> None:
        """Clear Work Memory completely (never touches Personal Memory)."""
        self.db.clear_domain("work_memory")
        self.events.memory_domain_updated.emit("work")

    # --------------------------------------------------------------------------
    # Voice & Audio State
    # --------------------------------------------------------------------------

    def get_voice_state(self) -> str:
        """Return current voice pipeline state."""
        return self._voice_state

    def set_voice_state(self, state: str) -> None:
        """Update and broadcast voice state."""
        self._voice_state = state
        self.events.voice_state_changed.emit(state)
        logger.info("Voice state changed to: %s", state)

    def is_mic_muted(self) -> bool:
        """Return microphone mute status."""
        return self._is_mic_muted

    def toggle_mic_mute(self) -> bool:
        """Toggle microphone mute and emit state."""
        self._is_mic_muted = not self._is_mic_muted
        self.events.voice_mute_toggled.emit(self._is_mic_muted)
        logger.info("Microphone mute set to: %s", self._is_mic_muted)
        return self._is_mic_muted


_controller_instance: Optional[AppController] = None


def get_app_controller(config: Optional[AppConfig] = None) -> AppController:
    """Return singleton AppController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = AppController(config)
    return _controller_instance
