"""Database schema definitions for HAKS SQLite databases.

Defines tables, indexes, and constraints for the 5 independent storage files:
- personal_memory.db
- work_memory.db
- conversations.db
- memory_proposals.db
- settings.db
"""

# Schema for personal_memory.db
PERSONAL_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS personal_memories (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance INTEGER DEFAULT 1,
    is_archived INTEGER DEFAULT 0,
    source_conversation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_personal_memories_category ON personal_memories(category);
CREATE INDEX IF NOT EXISTS idx_personal_memories_created_at ON personal_memories(created_at);
"""

# Schema for work_memory.db
WORK_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_memories (
    id TEXT PRIMARY KEY,
    project_name TEXT DEFAULT 'default',
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    importance INTEGER DEFAULT 1,
    is_archived INTEGER DEFAULT 0,
    source_conversation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_work_memories_project ON work_memories(project_name);
CREATE INDEX IF NOT EXISTS idx_work_memories_created_at ON work_memories(created_at);
"""

# Schema for conversations.db
CONVERSATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    language TEXT DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens INTEGER DEFAULT 0,
    metadata TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
"""

# Schema for memory_proposals.db
MEMORY_PROPOSALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_proposals (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK(domain IN ('personal', 'work')),
    source_conversation_id TEXT,
    proposal_text TEXT NOT NULL,
    suggested_title TEXT,
    suggested_category TEXT,
    suggested_tags TEXT,
    confidence REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_proposals_status ON memory_proposals(status);
CREATE INDEX IF NOT EXISTS idx_memory_proposals_domain ON memory_proposals(domain);
"""

# Schema for settings.db
SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    section TEXT DEFAULT 'general',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action TEXT NOT NULL,
    risk_level INTEGER NOT NULL,
    target TEXT,
    approved INTEGER NOT NULL,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_risk_level ON audit_logs(risk_level);
"""

DATABASE_SCHEMAS: dict[str, str] = {
    "personal_memory": PERSONAL_MEMORY_SCHEMA,
    "work_memory": WORK_MEMORY_SCHEMA,
    "conversations": CONVERSATIONS_SCHEMA,
    "memory_proposals": MEMORY_PROPOSALS_SCHEMA,
    "settings": SETTINGS_SCHEMA,
}
