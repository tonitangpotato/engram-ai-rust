//! # IronClaw-Engram: Neuroscience-Grounded Memory for IronClaw Agents
//!
//! IronClaw-Engram is a Rust port of [Engram](https://github.com/tonitangpotato/engram-ai),
//! a memory system for AI agents based on cognitive science models, optimized for
//! integration with [IronClaw](https://github.com/nearai/ironclaw).
//!
//! ## Core Cognitive Models
//!
//! - **ACT-R Activation**: Retrieval based on frequency, recency, and spreading activation
//! - **Memory Chain Model**: Dual-trace consolidation (hippocampus → neocortex)
//! - **Ebbinghaus Forgetting**: Exponential decay with spaced repetition
//! - **Hebbian Learning**: Co-activation forms associative links
//! - **STDP**: Temporal patterns infer causal relationships
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use engramai::{Memory, MemoryType};
//!
//! let mut mem = Memory::new("./agent.db", None)?;
//!
//! // Store memories
//! mem.add(
//!     "potato prefers action over discussion",
//!     MemoryType::Relational,
//!     Some(0.7),
//!     None,
//!     None,
//! )?;
//!
//! // Recall with ACT-R activation
//! let results = mem.recall("what does potato prefer?", 5, None, None)?;
//! for r in results {
//!     println!("[{}] {}", r.confidence_label, r.record.content);
//! }
//!
//! // Consolidate (run "sleep" cycle)
//! mem.consolidate(1.0)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Configuration Presets
//!
//! ```rust
//! use engramai::MemoryConfig;
//!
//! // Chatbot: slow decay, high replay
//! let config = MemoryConfig::chatbot();
//!
//! // Task agent: fast decay, low replay
//! let config = MemoryConfig::task_agent();
//!
//! // Personal assistant: very slow core decay
//! let config = MemoryConfig::personal_assistant();
//!
//! // Researcher: minimal forgetting
//! let config = MemoryConfig::researcher();
//! ```

pub mod config;
pub mod memory;
pub mod models;
pub mod storage;
pub mod types;

pub use config::MemoryConfig;
pub use memory::Memory;
pub use types::{AclEntry, MemoryLayer, MemoryRecord, MemoryStats, MemoryType, Permission, RecallResult};
