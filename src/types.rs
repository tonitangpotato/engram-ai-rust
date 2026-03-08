//! Core memory data types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Memory type classification following neuroscience categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    /// Factual knowledge: "SaltyHall uses Supabase"
    Factual,
    /// Episodic events: "On Feb 2 we shipped 10 features"
    Episodic,
    /// Relational knowledge: "potato prefers action over discussion"
    Relational,
    /// Emotional memories: "potato said I kinda like you"
    Emotional,
    /// Procedural knowledge: "Use www.moltbook.com not moltbook.com"
    Procedural,
    /// Opinions: "I think graph+text hybrid is best"
    Opinion,
    /// Causal relationships: "changing auth.py → downstream tests fail"
    Causal,
}

impl MemoryType {
    /// Default importance value for this memory type.
    pub fn default_importance(&self) -> f64 {
        match self {
            MemoryType::Factual => 0.3,
            MemoryType::Episodic => 0.4,
            MemoryType::Relational => 0.6,
            MemoryType::Emotional => 0.9,
            MemoryType::Procedural => 0.5,
            MemoryType::Opinion => 0.3,
            MemoryType::Causal => 0.7,
        }
    }

    /// Default decay rate (mu parameter) for this memory type.
    /// Lower = decays slower = lasts longer.
    pub fn default_decay_rate(&self) -> f64 {
        match self {
            MemoryType::Factual => 0.03,
            MemoryType::Episodic => 0.10,
            MemoryType::Relational => 0.02,
            MemoryType::Emotional => 0.01,
            MemoryType::Procedural => 0.01,
            MemoryType::Opinion => 0.05,
            MemoryType::Causal => 0.02,
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryType::Factual => write!(f, "factual"),
            MemoryType::Episodic => write!(f, "episodic"),
            MemoryType::Relational => write!(f, "relational"),
            MemoryType::Emotional => write!(f, "emotional"),
            MemoryType::Procedural => write!(f, "procedural"),
            MemoryType::Opinion => write!(f, "opinion"),
            MemoryType::Causal => write!(f, "causal"),
        }
    }
}

/// Memory consolidation layer (Memory Chain Model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryLayer {
    /// Core: always loaded, distilled knowledge
    Core,
    /// Working: recent daily notes (7 days)
    Working,
    /// Archive: old, searched on demand
    Archive,
}

impl fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryLayer::Core => write!(f, "core"),
            MemoryLayer::Working => write!(f, "working"),
            MemoryLayer::Archive => write!(f, "archive"),
        }
    }
}

/// A single memory entry with all metadata for cognitive models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Unique memory ID (8-char UUID prefix)
    pub id: String,
    /// Memory content (natural language)
    pub content: String,
    /// Memory type
    pub memory_type: MemoryType,
    /// Current layer
    pub layer: MemoryLayer,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// All access timestamps (for ACT-R base-level activation)
    pub access_times: Vec<DateTime<Utc>>,
    
    /// Working memory strength (hippocampal trace, fast decay)
    pub working_strength: f64,
    /// Core memory strength (neocortical trace, slow decay)
    pub core_strength: f64,
    
    /// Importance/emotional modulation (0-1)
    pub importance: f64,
    /// Pinned memories never decay
    pub pinned: bool,
    
    /// Number of consolidation cycles
    pub consolidation_count: i32,
    /// Last consolidation timestamp
    pub last_consolidated: Option<DateTime<Utc>>,
    
    /// Source identifier
    pub source: String,
    
    /// Contradiction links
    pub contradicts: Option<String>,
    pub contradicted_by: Option<String>,
    
    /// Optional structured metadata (JSON)
    pub metadata: Option<serde_json::Value>,
}

impl MemoryRecord {
    /// Age in hours since creation.
    pub fn age_hours(&self) -> f64 {
        let now = Utc::now();
        (now - self.created_at).num_seconds() as f64 / 3600.0
    }

    /// Age in days since creation.
    pub fn age_days(&self) -> f64 {
        self.age_hours() / 24.0
    }
}

/// Search result with activation score and confidence.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub record: MemoryRecord,
    pub activation: f64,
    pub confidence: f64,
    pub confidence_label: String,
}

/// Memory system statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub by_type: std::collections::HashMap<String, TypeStats>,
    pub by_layer: std::collections::HashMap<String, LayerStats>,
    pub pinned: usize,
    pub uptime_hours: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStats {
    pub count: usize,
    pub avg_strength: f64,
    pub avg_importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub count: usize,
    pub avg_working: f64,
    pub avg_core: f64,
}
