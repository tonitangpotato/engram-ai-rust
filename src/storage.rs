//! SQLite storage backend for Engram.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Result as SqlResult};
use std::path::Path;

use crate::types::{MemoryLayer, MemoryRecord, MemoryType};

/// SQLite-backed memory storage with FTS5 search.
pub struct Storage {
    conn: Connection,
}

impl Storage {
    /// Open or create a SQLite database at the given path.
    ///
    /// Use `:memory:` for an in-memory database.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;

        // Enable WAL mode for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // Create schema
        Self::create_schema(&conn)?;

        Ok(Self { conn })
    }

    fn create_schema(conn: &Connection) -> SqlResult<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                working_strength REAL NOT NULL DEFAULT 1.0,
                core_strength REAL NOT NULL DEFAULT 0.0,
                importance REAL NOT NULL DEFAULT 0.3,
                pinned INTEGER NOT NULL DEFAULT 0,
                consolidation_count INTEGER NOT NULL DEFAULT 0,
                last_consolidated TEXT,
                source TEXT DEFAULT '',
                contradicts TEXT DEFAULT '',
                contradicted_by TEXT DEFAULT '',
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS access_log (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                accessed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hebbian_links (
                source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                strength REAL NOT NULL DEFAULT 1.0,
                coactivation_count INTEGER NOT NULL DEFAULT 0,
                temporal_forward INTEGER NOT NULL DEFAULT 0,
                temporal_backward INTEGER NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'bidirectional',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );

            CREATE INDEX IF NOT EXISTS idx_access_log_mid ON access_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_hebbian_source ON hebbian_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_hebbian_target ON hebbian_links(target_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);

            -- FTS5 for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, content=memories, content_rowid=rowid
            );

            -- FTS triggers to keep index in sync
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            "#,
        )?;
        Ok(())
    }

    /// Add a new memory to storage.
    pub fn add(&mut self, record: &MemoryRecord) -> Result<(), rusqlite::Error> {
        let tx = self.conn.transaction()?;

        let metadata_json = record
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        tx.execute(
            r#"
            INSERT INTO memories (
                id, content, memory_type, layer, created_at,
                working_strength, core_strength, importance, pinned,
                consolidation_count, last_consolidated, source,
                contradicts, contradicted_by, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                record.id,
                record.content,
                record.memory_type.to_string(),
                record.layer.to_string(),
                record.created_at.to_rfc3339(),
                record.working_strength,
                record.core_strength,
                record.importance,
                record.pinned as i32,
                record.consolidation_count,
                record.last_consolidated.map(|dt| dt.to_rfc3339()),
                record.source,
                record.contradicts.as_ref().unwrap_or(&String::new()),
                record.contradicted_by.as_ref().unwrap_or(&String::new()),
                metadata_json,
            ],
        )?;

        // Record initial access
        tx.execute(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            params![record.id, record.created_at.to_rfc3339()],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Get a memory by ID.
    pub fn get(&self, id: &str) -> Result<Option<MemoryRecord>, rusqlite::Error> {
        let access_times = self.get_access_times(id)?;

        self.conn
            .query_row("SELECT * FROM memories WHERE id = ?", params![id], |row| {
                self.row_to_record(row, access_times.clone())
            })
            .optional()
    }

    /// Get all memories.
    pub fn all(&self) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;

        rows.collect()
    }

    /// Update an existing memory.
    pub fn update(&mut self, record: &MemoryRecord) -> Result<(), rusqlite::Error> {
        let metadata_json = record
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        self.conn.execute(
            r#"
            UPDATE memories SET
                content = ?, memory_type = ?, layer = ?,
                working_strength = ?, core_strength = ?, importance = ?,
                pinned = ?, consolidation_count = ?, last_consolidated = ?,
                source = ?, contradicts = ?, contradicted_by = ?, metadata = ?
            WHERE id = ?
            "#,
            params![
                record.content,
                record.memory_type.to_string(),
                record.layer.to_string(),
                record.working_strength,
                record.core_strength,
                record.importance,
                record.pinned as i32,
                record.consolidation_count,
                record.last_consolidated.map(|dt| dt.to_rfc3339()),
                record.source,
                record.contradicts.as_ref().unwrap_or(&String::new()),
                record.contradicted_by.as_ref().unwrap_or(&String::new()),
                metadata_json,
                record.id,
            ],
        )?;
        Ok(())
    }

    /// Delete a memory by ID.
    pub fn delete(&mut self, id: &str) -> Result<(), rusqlite::Error> {
        self.conn
            .execute("DELETE FROM memories WHERE id = ?", params![id])?;
        Ok(())
    }

    /// Record an access for a memory.
    pub fn record_access(&mut self, id: &str) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            params![id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Get all access timestamps for a memory.
    pub fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT accessed_at FROM access_log WHERE memory_id = ? ORDER BY accessed_at",
        )?;

        let rows = stmt.query_map(params![id], |row| {
            let ts: String = row.get(0)?;
            Ok(DateTime::parse_from_rfc3339(&ts)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()))
        })?;

        rows.collect()
    }

    /// Full-text search using FTS5.
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        // Clean query: remove FTS5 special characters and punctuation
        let cleaned: String = query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        let words: Vec<&str> = cleaned.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Build simple OR query for better matching
        let fts_query = words.join(" OR ");

        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.* FROM memories m
            JOIN memories_fts f ON m.rowid = f.rowid
            WHERE memories_fts MATCH ?
            ORDER BY rank LIMIT ?
            "#,
        )?;

        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
            let id: String = row.get(0)?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;

        rows.collect()
    }

    /// Search memories by type.
    pub fn search_by_type(
        &self,
        memory_type: MemoryType,
    ) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM memories WHERE memory_type = ?")?;

        let rows = stmt.query_map(params![memory_type.to_string()], |row| {
            let id: String = row.get(0)?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;

        rows.collect()
    }

    /// Get Hebbian neighbors for a memory.
    pub fn get_hebbian_neighbors(&self, memory_id: &str) -> Result<Vec<String>, rusqlite::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT target_id FROM hebbian_links WHERE source_id = ? AND strength > 0")?;

        let rows = stmt.query_map(params![memory_id], |row| row.get(0))?;
        rows.collect()
    }

    /// Record co-activation for Hebbian learning.
    pub fn record_coactivation(
        &mut self,
        id1: &str,
        id2: &str,
        threshold: i32,
    ) -> Result<bool, rusqlite::Error> {
        let (id1, id2) = if id1 < id2 { (id1, id2) } else { (id2, id1) };

        // Check existing link
        let existing: Option<(f64, i32)> = self.conn
            .query_row(
                "SELECT strength, coactivation_count FROM hebbian_links WHERE source_id = ? AND target_id = ?",
                params![id1, id2],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        match existing {
            Some((strength, count)) if strength > 0.0 => {
                // Link already formed, strengthen it
                let new_strength = (strength + 0.1).min(1.0);
                self.conn.execute(
                    "UPDATE hebbian_links SET strength = ?, coactivation_count = coactivation_count + 1 WHERE source_id = ? AND target_id = ?",
                    params![new_strength, id1, id2],
                )?;
                // Also update reverse link
                self.conn.execute(
                    "UPDATE hebbian_links SET strength = ?, coactivation_count = coactivation_count + 1 WHERE source_id = ? AND target_id = ?",
                    params![new_strength, id2, id1],
                )?;
                Ok(false)
            }
            Some((_, count)) => {
                // Tracking phase, increment count
                let new_count = count + 1;
                if new_count >= threshold {
                    // Threshold reached, form link
                    self.conn.execute(
                        "UPDATE hebbian_links SET strength = 1.0, coactivation_count = ? WHERE source_id = ? AND target_id = ?",
                        params![new_count, id1, id2],
                    )?;
                    // Create reverse link
                    self.conn.execute(
                        "INSERT OR REPLACE INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at) VALUES (?, ?, 1.0, ?, ?)",
                        params![id2, id1, new_count, Utc::now().to_rfc3339()],
                    )?;
                    Ok(true)
                } else {
                    self.conn.execute(
                        "UPDATE hebbian_links SET coactivation_count = ? WHERE source_id = ? AND target_id = ?",
                        params![new_count, id1, id2],
                    )?;
                    Ok(false)
                }
            }
            None => {
                // First co-activation, create tracking record
                self.conn.execute(
                    "INSERT INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at) VALUES (?, ?, 0.0, 1, ?)",
                    params![id1, id2, Utc::now().to_rfc3339()],
                )?;
                Ok(false)
            }
        }
    }

    /// Decay all Hebbian links by a factor.
    pub fn decay_hebbian_links(&mut self, factor: f64) -> Result<usize, rusqlite::Error> {
        // Decay all links
        self.conn.execute(
            "UPDATE hebbian_links SET strength = strength * ? WHERE strength > 0",
            params![factor],
        )?;

        // Prune very weak links
        let pruned = self.conn.execute(
            "DELETE FROM hebbian_links WHERE strength > 0 AND strength < 0.1",
            [],
        )?;

        Ok(pruned)
    }

    fn row_to_record(
        &self,
        row: &rusqlite::Row,
        access_times: Vec<DateTime<Utc>>,
    ) -> SqlResult<MemoryRecord> {
        let memory_type_str: String = row.get(2)?;
        let layer_str: String = row.get(3)?;
        let created_at_str: String = row.get(4)?;
        let last_consolidated_str: Option<String> = row.get(10)?;
        let metadata_str: Option<String> = row.get(14)?;

        let memory_type = match memory_type_str.as_str() {
            "factual" => MemoryType::Factual,
            "episodic" => MemoryType::Episodic,
            "relational" => MemoryType::Relational,
            "emotional" => MemoryType::Emotional,
            "procedural" => MemoryType::Procedural,
            "opinion" => MemoryType::Opinion,
            "causal" => MemoryType::Causal,
            _ => MemoryType::Factual,
        };

        let layer = match layer_str.as_str() {
            "core" => MemoryLayer::Core,
            "working" => MemoryLayer::Working,
            "archive" => MemoryLayer::Archive,
            _ => MemoryLayer::Working,
        };

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let last_consolidated = last_consolidated_str
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let contradicts_str: String = row.get(12)?;
        let contradicted_by_str: String = row.get(13)?;

        let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

        Ok(MemoryRecord {
            id: row.get(0)?,
            content: row.get(1)?,
            memory_type,
            layer,
            created_at,
            access_times,
            working_strength: row.get(5)?,
            core_strength: row.get(6)?,
            importance: row.get(7)?,
            pinned: row.get::<_, i32>(8)? != 0,
            consolidation_count: row.get(9)?,
            last_consolidated,
            source: row.get(11)?,
            contradicts: if contradicts_str.is_empty() {
                None
            } else {
                Some(contradicts_str)
            },
            contradicted_by: if contradicted_by_str.is_empty() {
                None
            } else {
                Some(contradicted_by_str)
            },
            metadata,
        })
    }
}
