//! SQLite storage backend for Engram.

use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{params, Connection, OptionalExtension, Result as SqlResult};
use std::path::Path;

use crate::types::{AclEntry, CrossLink, HebbianLink, MemoryLayer, MemoryRecord, MemoryType, Permission};

use std::sync::OnceLock;

/// Global jieba instance (loaded once, ~150ms first use, then instant).
fn jieba() -> &'static jieba_rs::Jieba {
    static JIEBA: OnceLock<jieba_rs::Jieba> = OnceLock::new();
    JIEBA.get_or_init(jieba_rs::Jieba::new)
}

/// Tokenize text for FTS5 indexing.
/// Uses jieba for Chinese word segmentation + CJK/ASCII boundary splitting.
/// e.g. "RustClaw是一个记忆系统" → "RustClaw 是 一个 记忆 系统"
/// e.g. "用Rust写agent框架" → "用 Rust 写 agent 框架"
fn tokenize_cjk_boundaries(text: &str) -> String {
    if !text.chars().any(is_cjk_char) {
        return text.to_string(); // Fast path: no CJK, skip jieba
    }
    
    // Use jieba to segment Chinese text
    let words = jieba().cut(text, true); // true = HMM mode for better accuracy
    
    // Join with spaces, then ensure CJK/ASCII boundaries have spaces
    let joined = words.join(" ");
    
    // Clean up: remove duplicate spaces
    let mut result = String::with_capacity(joined.len());
    let mut prev_space = false;
    for ch in joined.chars() {
        if ch == ' ' {
            if !prev_space {
                result.push(ch);
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result
}

/// Check if a character is CJK (Chinese/Japanese/Korean).
fn is_cjk_char(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul
    )
}

/// Convert a `DateTime<Utc>` to a Unix float (seconds since epoch).
fn datetime_to_f64(dt: &DateTime<Utc>) -> f64 {
    dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 / 1_000_000_000.0
}

/// Convert a Unix float (seconds since epoch) to `DateTime<Utc>`.
fn f64_to_datetime(ts: f64) -> DateTime<Utc> {
    let secs = ts.floor() as i64;
    let nanos = ((ts - secs as f64) * 1_000_000_000.0).max(0.0) as u32;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .unwrap_or_else(Utc::now)
}

/// Get the current time as a Unix float (seconds since epoch).
fn now_f64() -> f64 {
    datetime_to_f64(&Utc::now())
}

/// Convert raw bytes to Vec<f32> (little-endian).
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Embedding statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingStats {
    pub total_memories: usize,
    pub embedded_count: usize,
    pub model: Option<String>,
    pub dimensions: Option<usize>,
}

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
        
        // Run migrations for v2 features (namespace, ACL)
        Self::migrate_v2(&conn)?;
        
        // Run migrations for embeddings
        Self::migrate_embeddings(&conn)?;
        
        // Rebuild FTS with CJK tokenization if needed
        Self::rebuild_fts_if_needed(&conn)?;
        
        Ok(Self { conn })
    }
    
    /// Get a reference to the underlying database connection.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn create_schema(conn: &Connection) -> SqlResult<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                created_at REAL NOT NULL,
                working_strength REAL NOT NULL DEFAULT 1.0,
                core_strength REAL NOT NULL DEFAULT 0.0,
                importance REAL NOT NULL DEFAULT 0.3,
                pinned INTEGER NOT NULL DEFAULT 0,
                consolidation_count INTEGER NOT NULL DEFAULT 0,
                last_consolidated REAL,
                source TEXT DEFAULT '',
                contradicts TEXT DEFAULT '',
                contradicted_by TEXT DEFAULT '',
                metadata TEXT,
                namespace TEXT NOT NULL DEFAULT 'default'
            );

            CREATE TABLE IF NOT EXISTS access_log (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                accessed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hebbian_links (
                source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                strength REAL NOT NULL DEFAULT 1.0,
                coactivation_count INTEGER NOT NULL DEFAULT 0,
                temporal_forward INTEGER NOT NULL DEFAULT 0,
                temporal_backward INTEGER NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'bidirectional',
                created_at REAL NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (source_id, target_id)
            );
            
            CREATE TABLE IF NOT EXISTS engram_acl (
                agent_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                permission TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (agent_id, namespace)
            );

            -- Schema metadata
            CREATE TABLE IF NOT EXISTS engram_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO engram_meta VALUES ('schema_version', '1');

            -- Entity tables (canonical schema)
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                namespace TEXT NOT NULL DEFAULT 'default',
                created_at REAL NOT NULL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                role TEXT NOT NULL DEFAULT 'mention',
                PRIMARY KEY (memory_id, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_access_log_mid ON access_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_hebbian_source ON hebbian_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_hebbian_target ON hebbian_links(target_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_hebbian_namespace ON hebbian_links(namespace);
            CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace);
            CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);

            -- FTS5 for full-text search (manually managed, not via triggers,
            -- so we can pre-process content for CJK/ASCII boundary tokenization)
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content
            );
            "#,
        )?;
        Ok(())
    }
    
    /// Migrate existing databases to v2 schema (add namespace, ACL table).
    fn migrate_v2(conn: &Connection) -> SqlResult<()> {
        // Check if namespace column exists in memories table
        let has_namespace: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('memories') WHERE name='namespace'",
            [],
            |row| row.get(0),
        )?;
        
        if !has_namespace {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
                [],
            )?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)",
                [],
            )?;
        }
        
        // Check if namespace column exists in hebbian_links table
        let has_hebbian_namespace: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('hebbian_links') WHERE name='namespace'",
            [],
            |row| row.get(0),
        )?;
        
        if !has_hebbian_namespace {
            conn.execute(
                "ALTER TABLE hebbian_links ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
                [],
            )?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hebbian_namespace ON hebbian_links(namespace)",
                [],
            )?;
        }
        
        // Create ACL table if not exists (idempotent)
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS engram_acl (
                agent_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                permission TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (agent_id, namespace)
            );
            "#,
        )?;
        
        Ok(())
    }
    
    /// Migrate to add embeddings table.
    fn migrate_embeddings(conn: &Connection) -> SqlResult<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )?;
        
        Ok(())
    }

    /// Rebuild FTS index with CJK tokenization if not already done.
    /// Uses engram_meta 'fts_cjk_version' to track migration state.
    fn rebuild_fts_if_needed(conn: &Connection) -> SqlResult<()> {
        const FTS_CJK_VERSION: &str = "1";
        
        let current: Option<String> = conn
            .query_row(
                "SELECT value FROM engram_meta WHERE key = 'fts_cjk_version'",
                [],
                |row| row.get(0),
            )
            .ok();
        
        if current.as_deref() == Some(FTS_CJK_VERSION) {
            return Ok(()); // Already up to date
        }
        
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        if count == 0 {
            conn.execute(
                "INSERT OR REPLACE INTO engram_meta VALUES ('fts_cjk_version', ?1)",
                params![FTS_CJK_VERSION],
            )?;
            return Ok(());
        }
        
        // Rebuild: clear FTS and re-insert all with tokenization
        conn.execute("DELETE FROM memories_fts", [])?;
        
        let mut stmt = conn.prepare("SELECT rowid, content FROM memories")?;
        let rows: Vec<(i64, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        
        for (rowid, content) in &rows {
            let tokenized = tokenize_cjk_boundaries(content);
            conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?1, ?2)",
                params![rowid, tokenized],
            )?;
        }
        
        conn.execute(
            "INSERT OR REPLACE INTO engram_meta VALUES ('fts_cjk_version', ?1)",
            params![FTS_CJK_VERSION],
        )?;
        
        eprintln!("[engram] Rebuilt FTS index with CJK tokenization for {} memories", rows.len());
        Ok(())
    }

    /// Add a new memory to storage.
    pub fn add(&mut self, record: &MemoryRecord, namespace: &str) -> Result<(), rusqlite::Error> {
        let tx = self.conn.transaction()?;
        
        let metadata_json = record.metadata.as_ref().map(|m| serde_json::to_string(m).ok()).flatten();
        
        tx.execute(
            r#"
            INSERT INTO memories (
                id, content, memory_type, layer, created_at,
                working_strength, core_strength, importance, pinned,
                consolidation_count, last_consolidated, source,
                contradicts, contradicted_by, metadata, namespace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                record.id,
                record.content,
                record.memory_type.to_string(),
                record.layer.to_string(),
                datetime_to_f64(&record.created_at),
                record.working_strength,
                record.core_strength,
                record.importance,
                record.pinned as i32,
                record.consolidation_count,
                record.last_consolidated.map(|dt| datetime_to_f64(&dt)),
                record.source,
                record.contradicts.as_ref().unwrap_or(&String::new()),
                record.contradicted_by.as_ref().unwrap_or(&String::new()),
                metadata_json,
                namespace,
            ],
        )?;
        
        // Record initial access
        tx.execute(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            params![record.id, datetime_to_f64(&record.created_at)],
        )?;
        
        // Insert into FTS with CJK/ASCII boundary tokenization
        let tokenized = tokenize_cjk_boundaries(&record.content);
        let rowid: i64 = tx.query_row(
            "SELECT rowid FROM memories WHERE id = ?",
            params![record.id],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
            params![rowid, tokenized],
        )?;
        
        tx.commit()?;
        Ok(())
    }

    /// Get a memory by ID.
    pub fn get(&self, id: &str) -> Result<Option<MemoryRecord>, rusqlite::Error> {
        let access_times = self.get_access_times(id)?;
        
        self.conn
            .query_row(
                "SELECT * FROM memories WHERE id = ?",
                params![id],
                |row| self.row_to_record(row, access_times.clone()),
            )
            .optional()
    }

    /// Get all memories.
    pub fn all(&self) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let mut stmt = self.conn.prepare("SELECT * FROM memories")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get("id")?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;
        
        rows.collect()
    }

    /// Update an existing memory.
    pub fn update(&mut self, record: &MemoryRecord) -> Result<(), rusqlite::Error> {
        let metadata_json = record.metadata.as_ref().map(|m| serde_json::to_string(m).ok()).flatten();
        
        // Get rowid for FTS update
        let rowid: i64 = self.conn.query_row(
            "SELECT rowid FROM memories WHERE id = ?",
            params![record.id],
            |row| row.get(0),
        )?;
        
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
                record.last_consolidated.map(|dt| datetime_to_f64(&dt)),
                record.source,
                record.contradicts.as_ref().unwrap_or(&String::new()),
                record.contradicted_by.as_ref().unwrap_or(&String::new()),
                metadata_json,
                record.id,
            ],
        )?;
        
        // Update FTS with CJK tokenization
        let _ = self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", params![rowid]);
        let tokenized = tokenize_cjk_boundaries(&record.content);
        let _ = self.conn.execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
            params![rowid, tokenized],
        );
        
        Ok(())
    }

    /// Delete a memory by ID.
    pub fn delete(&mut self, id: &str) -> Result<(), rusqlite::Error> {
        // Delete FTS entry (standalone table, delete by rowid)
        let rowid: Result<i64, _> = self.conn.query_row(
            "SELECT rowid FROM memories WHERE id = ?",
            params![id],
            |row| row.get(0),
        );
        if let Ok(rowid) = rowid {
            let _ = self.conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?",
                params![rowid],
            );
        }
        self.conn.execute("DELETE FROM memories WHERE id = ?", params![id])?;
        Ok(())
    }
    
    /// Update just the content and metadata of a memory.
    ///
    /// Used by update_memory to change content while preserving other fields.
    pub fn update_content(
        &mut self,
        id: &str,
        new_content: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), rusqlite::Error> {
        let metadata_json = metadata.map(|m| serde_json::to_string(&m).ok()).flatten();
        
        // Get rowid before updating
        let rowid: i64 = self.conn.query_row(
            "SELECT rowid FROM memories WHERE id = ?",
            params![id],
            |row| row.get(0),
        )?;
        
        self.conn.execute(
            "UPDATE memories SET content = ?, metadata = ? WHERE id = ?",
            params![new_content, metadata_json, id],
        )?;
        
        // Update FTS index manually (no triggers, need CJK tokenization)
        let _ = self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", params![rowid]);
        let tokenized = tokenize_cjk_boundaries(new_content);
        let _ = self.conn.execute(
            "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
            params![rowid, tokenized],
        );
        
        Ok(())
    }
    
    /// Get all memories of a specific type, optionally filtered by namespace.
    pub fn search_by_type_ns(
        &self,
        memory_type: MemoryType,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let ns = namespace.unwrap_or("default");
        
        if ns == "*" {
            let mut stmt = self.conn.prepare(
                "SELECT * FROM memories WHERE memory_type = ? ORDER BY importance DESC LIMIT ?"
            )?;
            
            let rows = stmt.query_map(params![memory_type.to_string(), limit as i64], |row| {
                let id: String = row.get("id")?;
                let access_times = self.get_access_times(&id).unwrap_or_default();
                self.row_to_record(row, access_times)
            })?;
            
            rows.collect()
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT * FROM memories WHERE memory_type = ? AND namespace = ? ORDER BY importance DESC LIMIT ?"
            )?;
            
            let rows = stmt.query_map(params![memory_type.to_string(), ns, limit as i64], |row| {
                let id: String = row.get("id")?;
                let access_times = self.get_access_times(&id).unwrap_or_default();
                self.row_to_record(row, access_times)
            })?;
            
            rows.collect()
        }
    }

    /// Record an access for a memory.
    pub fn record_access(&mut self, id: &str) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at) VALUES (?, ?)",
            params![id, now_f64()],
        )?;
        Ok(())
    }

    /// Get all access timestamps for a memory.
    pub fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, rusqlite::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT accessed_at FROM access_log WHERE memory_id = ? ORDER BY accessed_at")?;
        
        let rows = stmt.query_map(params![id], |row| {
            let ts: f64 = row.get(0)?;
            Ok(f64_to_datetime(ts))
        })?;
        
        rows.collect()
    }

    /// Full-text search using FTS5.
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        // Tokenize CJK boundaries, then clean
        let tokenized = tokenize_cjk_boundaries(query);
        let cleaned: String = tokenized
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
            let id: String = row.get("id")?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;
        
        rows.collect()
    }

    /// Search memories by type.
    pub fn search_by_type(&self, memory_type: MemoryType) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM memories WHERE memory_type = ?")?;
        
        let rows = stmt.query_map(params![memory_type.to_string()], |row| {
            let id: String = row.get("id")?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;
        
        rows.collect()
    }

    /// Get Hebbian neighbors for a memory.
    pub fn get_hebbian_neighbors(&self, memory_id: &str) -> Result<Vec<String>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT target_id FROM hebbian_links WHERE source_id = ? AND strength > 0"
        )?;
        
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
                        params![id2, id1, new_count, now_f64()],
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
                    params![id1, id2, now_f64()],
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
        // Use column names instead of indices to handle DBs with extra columns (e.g. Python's summary/tokens)
        let memory_type_str: String = row.get("memory_type")?;
        let layer_str: String = row.get("layer")?;
        let created_at_f64: f64 = row.get("created_at")?;
        let last_consolidated_f64: Option<f64> = row.get("last_consolidated")?;
        let metadata_str: Option<String> = row.get("metadata")?;
        
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
        
        let created_at = f64_to_datetime(created_at_f64);
        
        let last_consolidated = last_consolidated_f64.map(f64_to_datetime);
        
        let contradicts_str: String = row.get("contradicts")?;
        let contradicted_by_str: String = row.get("contradicted_by")?;
        
        let metadata = metadata_str
            .and_then(|s| serde_json::from_str(&s).ok());
        
        Ok(MemoryRecord {
            id: row.get("id")?,
            content: row.get("content")?,
            memory_type,
            layer,
            created_at,
            access_times,
            working_strength: row.get("working_strength")?,
            core_strength: row.get("core_strength")?,
            importance: row.get("importance")?,
            pinned: row.get::<_, i32>("pinned")? != 0,
            consolidation_count: row.get("consolidation_count")?,
            last_consolidated,
            source: row.get("source")?,
            contradicts: if contradicts_str.is_empty() { None } else { Some(contradicts_str) },
            contradicted_by: if contradicted_by_str.is_empty() { None } else { Some(contradicted_by_str) },
            metadata,
        })
    }
    
    /// Get the namespace of a memory by ID.
    pub fn get_namespace(&self, id: &str) -> Result<Option<String>, rusqlite::Error> {
        self.conn
            .query_row(
                "SELECT namespace FROM memories WHERE id = ?",
                params![id],
                |row| row.get(0),
            )
            .optional()
    }
    
    /// Full-text search using FTS5, filtered by namespace.
    /// 
    /// If namespace is None, search in "default" namespace.
    /// If namespace is Some("*"), search across all namespaces.
    pub fn search_fts_ns(
        &self,
        query: &str,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        // Tokenize CJK boundaries, then clean
        let tokenized = tokenize_cjk_boundaries(query);
        let cleaned: String = tokenized
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        
        let words: Vec<&str> = cleaned.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }
        
        // Build simple OR query for better matching
        let fts_query = words.join(" OR ");
        
        let ns = namespace.unwrap_or("default");
        
        if ns == "*" {
            // Search all namespaces
            let mut stmt = self.conn.prepare(
                r#"
                SELECT m.* FROM memories m
                JOIN memories_fts f ON m.rowid = f.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank LIMIT ?
                "#,
            )?;
            
            let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
                let id: String = row.get("id")?;
                let access_times = self.get_access_times(&id).unwrap_or_default();
                self.row_to_record(row, access_times)
            })?;
            
            rows.collect()
        } else {
            // Search specific namespace
            let mut stmt = self.conn.prepare(
                r#"
                SELECT m.* FROM memories m
                JOIN memories_fts f ON m.rowid = f.rowid
                WHERE memories_fts MATCH ? AND m.namespace = ?
                ORDER BY rank LIMIT ?
                "#,
            )?;
            
            let rows = stmt.query_map(params![fts_query, ns, limit as i64], |row| {
                let id: String = row.get("id")?;
                let access_times = self.get_access_times(&id).unwrap_or_default();
                self.row_to_record(row, access_times)
            })?;
            
            rows.collect()
        }
    }
    
    /// Get all memories in a specific namespace.
    pub fn all_in_namespace(&self, namespace: Option<&str>) -> Result<Vec<MemoryRecord>, rusqlite::Error> {
        let ns = namespace.unwrap_or("default");
        
        if ns == "*" {
            return self.all();
        }
        
        let mut stmt = self.conn.prepare("SELECT * FROM memories WHERE namespace = ?")?;
        let rows = stmt.query_map(params![ns], |row| {
            let id: String = row.get("id")?;
            let access_times = self.get_access_times(&id).unwrap_or_default();
            self.row_to_record(row, access_times)
        })?;
        
        rows.collect()
    }
    
    // === Embedding Methods ===
    
    /// Store embedding for a memory.
    ///
    /// Serializes the embedding as raw f32 bytes (little-endian) for compact storage.
    pub fn store_embedding(
        &mut self,
        memory_id: &str,
        embedding: &[f32],
        model: &str,
        dimensions: usize,
    ) -> Result<(), rusqlite::Error> {
        // Serialize Vec<f32> as raw bytes (4 bytes per f32, little-endian)
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        let now = chrono::Utc::now().to_rfc3339();
        
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, model, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
            params![memory_id, bytes, model, dimensions as i64, now],
        )?;
        
        Ok(())
    }
    
    /// Get embedding for a memory.
    ///
    /// Deserializes raw f32 bytes back to Vec<f32>.
    pub fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, rusqlite::Error> {
        let result: Option<Vec<u8>> = self.conn
            .query_row(
                "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
                params![memory_id],
                |row| row.get(0),
            )
            .optional()?;
        
        Ok(result.map(|bytes| bytes_to_f32_vec(&bytes)))
    }
    
    /// Get all embeddings for similarity search.
    ///
    /// Returns (memory_id, embedding) pairs.
    pub fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT memory_id, embedding FROM memory_embeddings"
        )?;
        
        let rows = stmt.query_map([], |row| {
            let memory_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            Ok((memory_id, bytes_to_f32_vec(&bytes)))
        })?;
        
        rows.collect()
    }
    
    /// Get embeddings for a specific namespace.
    pub fn get_embeddings_in_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, Vec<f32>)>, rusqlite::Error> {
        let ns = namespace.unwrap_or("default");
        
        if ns == "*" {
            return self.get_all_embeddings();
        }
        
        let mut stmt = self.conn.prepare(
            r#"
            SELECT e.memory_id, e.embedding FROM memory_embeddings e
            JOIN memories m ON e.memory_id = m.id
            WHERE m.namespace = ?
            "#
        )?;
        
        let rows = stmt.query_map(params![ns], |row| {
            let memory_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            Ok((memory_id, bytes_to_f32_vec(&bytes)))
        })?;
        
        rows.collect()
    }
    
    /// Delete embedding for a memory.
    pub fn delete_embedding(&mut self, memory_id: &str) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?",
            params![memory_id],
        )?;
        Ok(())
    }
    
    /// Get memory IDs that don't have embeddings yet.
    pub fn get_memories_without_embeddings(&self) -> Result<Vec<String>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.id FROM memories m
            LEFT JOIN memory_embeddings e ON m.id = e.memory_id
            WHERE e.memory_id IS NULL
            "#
        )?;
        
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect()
    }
    
    /// Get embedding statistics.
    pub fn embedding_stats(&self) -> Result<EmbeddingStats, rusqlite::Error> {
        let total_memories: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |row| row.get(0),
        )?;
        
        let embedded_count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_embeddings",
            [],
            |row| row.get(0),
        )?;
        
        let model: Option<String> = self.conn.query_row(
            "SELECT model FROM memory_embeddings LIMIT 1",
            [],
            |row| row.get(0),
        ).optional()?;
        
        let dimensions: Option<usize> = self.conn.query_row(
            "SELECT dimensions FROM memory_embeddings LIMIT 1",
            [],
            |row| row.get::<_, i64>(0).map(|d| d as usize),
        ).optional()?;
        
        Ok(EmbeddingStats {
            total_memories,
            embedded_count,
            model,
            dimensions,
        })
    }
    
    // === ACL Methods ===
    
    /// Grant a permission to an agent for a namespace.
    pub fn grant_permission(
        &mut self,
        agent_id: &str,
        namespace: &str,
        permission: Permission,
        granted_by: &str,
    ) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO engram_acl (agent_id, namespace, permission, granted_by, created_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
            params![
                agent_id,
                namespace,
                permission.to_string(),
                granted_by,
                now_f64(),
            ],
        )?;
        Ok(())
    }
    
    /// Revoke a permission from an agent for a namespace.
    pub fn revoke_permission(&mut self, agent_id: &str, namespace: &str) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "DELETE FROM engram_acl WHERE agent_id = ? AND namespace = ?",
            params![agent_id, namespace],
        )?;
        Ok(())
    }
    
    /// Check if an agent has a specific permission for a namespace.
    /// 
    /// Permission hierarchy: admin > write > read
    /// Wildcard namespace ("*") grants access to all namespaces.
    pub fn check_permission(
        &self,
        agent_id: &str,
        namespace: &str,
        required: Permission,
    ) -> Result<bool, rusqlite::Error> {
        // Check for direct namespace permission
        let direct: Option<String> = self.conn
            .query_row(
                "SELECT permission FROM engram_acl WHERE agent_id = ? AND namespace = ?",
                params![agent_id, namespace],
                |row| row.get(0),
            )
            .optional()?;
        
        if let Some(perm_str) = direct {
            if let Ok(perm) = perm_str.parse::<Permission>() {
                return Ok(Self::permission_allows(perm, required));
            }
        }
        
        // Check for wildcard namespace permission
        let wildcard: Option<String> = self.conn
            .query_row(
                "SELECT permission FROM engram_acl WHERE agent_id = ? AND namespace = '*'",
                params![agent_id],
                |row| row.get(0),
            )
            .optional()?;
        
        if let Some(perm_str) = wildcard {
            if let Ok(perm) = perm_str.parse::<Permission>() {
                return Ok(Self::permission_allows(perm, required));
            }
        }
        
        // Default: check if this is the agent's own namespace or global namespace
        // Global namespace ("global") is readable by everyone
        if namespace == "global" && matches!(required, Permission::Read) {
            return Ok(true);
        }
        
        // Default write to own namespace
        if namespace == agent_id && matches!(required, Permission::Write | Permission::Read) {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Check if granted permission allows required permission.
    fn permission_allows(granted: Permission, required: Permission) -> bool {
        match required {
            Permission::Read => granted.can_read(),
            Permission::Write => granted.can_write(),
            Permission::Admin => granted.is_admin(),
        }
    }
    
    /// List all permissions for an agent.
    pub fn list_permissions(&self, agent_id: &str) -> Result<Vec<AclEntry>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT agent_id, namespace, permission, granted_by, created_at FROM engram_acl WHERE agent_id = ?"
        )?;
        
        let rows = stmt.query_map(params![agent_id], |row| {
            let perm_str: String = row.get(2)?;
            let created_at_f64: f64 = row.get(4)?;
            
            Ok(AclEntry {
                agent_id: row.get(0)?,
                namespace: row.get(1)?,
                permission: perm_str.parse().unwrap_or(Permission::Read),
                granted_by: row.get(3)?,
                created_at: f64_to_datetime(created_at_f64),
            })
        })?;
        
        rows.collect()
    }
    
    /// Get Hebbian neighbors for a memory, optionally filtered by namespace.
    pub fn get_hebbian_neighbors_ns(
        &self,
        memory_id: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<String>, rusqlite::Error> {
        match namespace {
            Some("*") | None => {
                // All namespaces (original behavior)
                self.get_hebbian_neighbors(memory_id)
            }
            Some(ns) => {
                let mut stmt = self.conn.prepare(
                    "SELECT target_id FROM hebbian_links WHERE source_id = ? AND strength > 0 AND namespace = ?"
                )?;
                
                let rows = stmt.query_map(params![memory_id, ns], |row| row.get(0))?;
                rows.collect()
            }
        }
    }
    
    /// Record co-activation with namespace tracking.
    pub fn record_coactivation_ns(
        &mut self,
        id1: &str,
        id2: &str,
        threshold: i32,
        namespace: &str,
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
                        "INSERT OR REPLACE INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at, namespace) VALUES (?, ?, 1.0, ?, ?, ?)",
                        params![id2, id1, new_count, now_f64(), namespace],
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
                    "INSERT INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at, namespace) VALUES (?, ?, 0.0, 1, ?, ?)",
                    params![id1, id2, now_f64(), namespace],
                )?;
                Ok(false)
            }
        }
    }
    
    // === Cross-Namespace Hebbian Methods (Phase 3) ===
    
    /// Record cross-namespace co-activation.
    ///
    /// When memories from different namespaces are recalled together,
    /// this creates a Hebbian link that spans namespaces.
    pub fn record_cross_namespace_coactivation(
        &mut self,
        id1: &str,
        ns1: &str,
        id2: &str,
        ns2: &str,
        threshold: i32,
    ) -> Result<bool, rusqlite::Error> {
        // Only create cross-namespace links when namespaces differ
        if ns1 == ns2 {
            return self.record_coactivation_ns(id1, id2, threshold, ns1);
        }
        
        // Ensure consistent ordering
        let (id1, id2, ns1, ns2) = if (ns1, id1) < (ns2, id2) {
            (id1, id2, ns1, ns2)
        } else {
            (id2, id1, ns2, ns1)
        };
        
        // Check existing link
        let existing: Option<(f64, i32)> = self.conn
            .query_row(
                "SELECT strength, coactivation_count FROM hebbian_links WHERE source_id = ? AND target_id = ?",
                params![id1, id2],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        
        // Use "cross" as special namespace marker for cross-namespace links
        let cross_ns = format!("{}:{}", ns1, ns2);
        
        match existing {
            Some((strength, _count)) if strength > 0.0 => {
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
                        "INSERT OR REPLACE INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at, namespace) VALUES (?, ?, 1.0, ?, ?, ?)",
                        params![id2, id1, new_count, now_f64(), &cross_ns],
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
                    "INSERT INTO hebbian_links (source_id, target_id, strength, coactivation_count, created_at, namespace) VALUES (?, ?, 0.0, 1, ?, ?)",
                    params![id1, id2, now_f64(), &cross_ns],
                )?;
                Ok(false)
            }
        }
    }
    
    /// Discover cross-namespace Hebbian links between two namespaces.
    ///
    /// Returns all Hebbian links where source is in namespace_a and target
    /// is in namespace_b (or vice versa).
    pub fn discover_cross_links(
        &self,
        namespace_a: &str,
        namespace_b: &str,
    ) -> Result<Vec<HebbianLink>, rusqlite::Error> {
        // Find links with cross-namespace marker
        let cross_ns_1 = format!("{}:{}", namespace_a, namespace_b);
        let cross_ns_2 = format!("{}:{}", namespace_b, namespace_a);
        
        let mut stmt = self.conn.prepare(
            r#"
            SELECT h.source_id, h.target_id, h.strength, h.coactivation_count, 
                   h.direction, h.created_at, h.namespace,
                   m1.namespace as source_ns, m2.namespace as target_ns
            FROM hebbian_links h
            LEFT JOIN memories m1 ON h.source_id = m1.id
            LEFT JOIN memories m2 ON h.target_id = m2.id
            WHERE h.strength > 0 AND (h.namespace = ? OR h.namespace = ?)
            ORDER BY h.strength DESC
            "#,
        )?;
        
        let rows = stmt.query_map(params![cross_ns_1, cross_ns_2], |row| {
            let created_at_f64: f64 = row.get(5)?;
            let source_ns: Option<String> = row.get(7)?;
            let target_ns: Option<String> = row.get(8)?;
            
            Ok(HebbianLink {
                source_id: row.get(0)?,
                target_id: row.get(1)?,
                strength: row.get(2)?,
                coactivation_count: row.get(3)?,
                direction: row.get(4)?,
                created_at: f64_to_datetime(created_at_f64),
                source_ns,
                target_ns,
            })
        })?;
        
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
    
    /// Get all cross-namespace links for a memory.
    pub fn get_cross_namespace_neighbors(
        &self,
        memory_id: &str,
    ) -> Result<Vec<CrossLink>, rusqlite::Error> {
        // Get source memory's namespace
        let source_ns = self.get_namespace(memory_id)?;
        
        let mut stmt = self.conn.prepare(
            r#"
            SELECT h.source_id, h.target_id, h.strength, m.namespace, m.content
            FROM hebbian_links h
            JOIN memories m ON h.target_id = m.id
            WHERE h.source_id = ? AND h.strength > 0
            "#,
        )?;
        
        let source_ns_str = source_ns.clone().unwrap_or_else(|| "default".to_string());
        
        let rows = stmt.query_map(params![memory_id], |row| {
            let target_ns: String = row.get(3)?;
            let content: String = row.get(4)?;
            
            Ok(CrossLink {
                source_id: row.get(0)?,
                source_ns: source_ns_str.clone(),
                target_id: row.get(1)?,
                target_ns,
                strength: row.get(2)?,
                description: Some(content),
            })
        })?;
        
        // Filter to only cross-namespace links
        let source_ns_val = source_ns.unwrap_or_else(|| "default".to_string());
        Ok(rows
            .filter_map(|r| r.ok())
            .filter(|link| link.target_ns != source_ns_val)
            .collect())
    }
    
    /// Get all cross-namespace links in the database.
    pub fn get_all_cross_links(&self) -> Result<Vec<CrossLink>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT h.source_id, h.target_id, h.strength, 
                   m1.namespace as source_ns, m2.namespace as target_ns,
                   m2.content as target_content
            FROM hebbian_links h
            JOIN memories m1 ON h.source_id = m1.id
            JOIN memories m2 ON h.target_id = m2.id
            WHERE h.strength > 0 AND m1.namespace != m2.namespace
            ORDER BY h.strength DESC
            "#,
        )?;
        
        let rows = stmt.query_map([], |row| {
            Ok(CrossLink {
                source_id: row.get(0)?,
                target_id: row.get(1)?,
                strength: row.get(2)?,
                source_ns: row.get(3)?,
                target_ns: row.get(4)?,
                description: row.get(5)?,
            })
        })?;
        
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}
