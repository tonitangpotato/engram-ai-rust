//! Engram CLI — command-line interface for multi-agent memory.
//!
//! Usage:
//!   engram store "content" --ns trading --type factual --importance 0.8
//!   engram recall "query" --ns "*" --limit 5 --json
//!   engram stats --ns trading
//!   engram consolidate --ns trading
//!   engram grant agent-id --ns namespace --perm read
//!   engram revoke agent-id --ns namespace

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use engramai::{Memory, MemoryConfig, MemoryType, Permission};

/// Engram — Neuroscience-grounded memory system for AI agents.
#[derive(Parser)]
#[command(name = "engram")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to SQLite database file
    #[arg(short, long, env = "ENGRAM_DB", default_value = "engram.db")]
    database: PathBuf,
    
    /// Agent ID for this session (used for ACL)
    #[arg(short, long, env = "ENGRAM_AGENT_ID")]
    agent_id: Option<String>,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a new memory
    Store {
        /// Memory content
        content: String,
        
        /// Namespace to store in
        #[arg(long, short = 'n', default_value = "default")]
        ns: String,
        
        /// Memory type
        #[arg(long, short = 't', default_value = "factual")]
        r#type: MemoryTypeArg,
        
        /// Importance score (0.0-1.0)
        #[arg(long, short = 'i')]
        importance: Option<f64>,
        
        /// Source identifier
        #[arg(long, short = 's')]
        source: Option<String>,
    },
    
    /// Recall memories by query
    Recall {
        /// Search query
        query: String,
        
        /// Namespace to search (use "*" for all)
        #[arg(long, short = 'n', default_value = "default")]
        ns: String,
        
        /// Maximum number of results
        #[arg(long, short = 'l', default_value = "5")]
        limit: usize,
        
        /// Minimum confidence threshold
        #[arg(long, short = 'c')]
        min_confidence: Option<f64>,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Show memory statistics
    Stats {
        /// Namespace to show stats for (use "*" for all)
        #[arg(long, short = 'n')]
        ns: Option<String>,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Run memory consolidation cycle
    Consolidate {
        /// Namespace to consolidate (omit for all)
        #[arg(long, short = 'n')]
        ns: Option<String>,
        
        /// Simulated days of consolidation
        #[arg(long, short = 'd', default_value = "1.0")]
        days: f64,
    },
    
    /// Grant access permission to an agent
    Grant {
        /// Agent ID to grant permission to
        agent_id: String,
        
        /// Namespace to grant access to
        #[arg(long, short = 'n')]
        ns: String,
        
        /// Permission level (read, write, admin)
        #[arg(long, short = 'p', default_value = "read")]
        perm: PermissionArg,
    },
    
    /// Revoke access permission from an agent
    Revoke {
        /// Agent ID to revoke permission from
        agent_id: String,
        
        /// Namespace to revoke access from
        #[arg(long, short = 'n')]
        ns: String,
    },
    
    /// List permissions for an agent
    Permissions {
        /// Agent ID to list permissions for
        agent_id: String,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Pin a memory (prevent decay)
    Pin {
        /// Memory ID to pin
        memory_id: String,
    },
    
    /// Unpin a memory (allow decay)
    Unpin {
        /// Memory ID to unpin
        memory_id: String,
    },
    
    /// Delete a specific memory
    Forget {
        /// Memory ID to delete
        memory_id: String,
    },
    
    /// Apply reward signal to recent memories
    Reward {
        /// Feedback text (positive/negative sentiment detected)
        feedback: String,
        
        /// Number of recent memories to affect
        #[arg(long, short = 'n', default_value = "3")]
        recent: usize,
    },
}

#[derive(Clone, ValueEnum)]
enum MemoryTypeArg {
    Factual,
    Episodic,
    Relational,
    Emotional,
    Procedural,
    Opinion,
    Causal,
}

impl From<MemoryTypeArg> for MemoryType {
    fn from(arg: MemoryTypeArg) -> Self {
        match arg {
            MemoryTypeArg::Factual => MemoryType::Factual,
            MemoryTypeArg::Episodic => MemoryType::Episodic,
            MemoryTypeArg::Relational => MemoryType::Relational,
            MemoryTypeArg::Emotional => MemoryType::Emotional,
            MemoryTypeArg::Procedural => MemoryType::Procedural,
            MemoryTypeArg::Opinion => MemoryType::Opinion,
            MemoryTypeArg::Causal => MemoryType::Causal,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum PermissionArg {
    Read,
    Write,
    Admin,
}

impl From<PermissionArg> for Permission {
    fn from(arg: PermissionArg) -> Self {
        match arg {
            PermissionArg::Read => Permission::Read,
            PermissionArg::Write => Permission::Write,
            PermissionArg::Admin => Permission::Admin,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    let db_path = cli.database.to_str().ok_or("invalid database path")?;
    let mut mem = Memory::new(db_path, Some(MemoryConfig::default()))?;
    
    if let Some(agent_id) = &cli.agent_id {
        mem.set_agent_id(agent_id);
    }
    
    match cli.command {
        Commands::Store { content, ns, r#type, importance, source } => {
            let id = mem.add_to_namespace(
                &content,
                r#type.into(),
                importance,
                source.as_deref(),
                None,
                Some(&ns),
            )?;
            println!("{}", id);
        }
        
        Commands::Recall { query, ns, limit, min_confidence, json } => {
            let ns_opt = if ns == "default" { None } else { Some(ns.as_str()) };
            let results = mem.recall_from_namespace(&query, limit, None, min_confidence, ns_opt)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                if results.is_empty() {
                    println!("No memories found.");
                } else {
                    for r in &results {
                        println!("[{}] ({:.2}) {}", r.record.id, r.confidence, r.record.content);
                        if !r.record.source.is_empty() {
                            println!("    source: {}", r.record.source);
                        }
                    }
                }
            }
        }
        
        Commands::Stats { ns, json } => {
            let stats = mem.stats_ns(ns.as_deref())?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("Total memories: {}", stats.total_memories);
                println!("Pinned: {}", stats.pinned);
                println!("Uptime: {:.2} hours", stats.uptime_hours);
                println!("\nBy type:");
                for (type_name, type_stats) in &stats.by_type {
                    println!("  {}: {} (avg strength: {:.3}, avg importance: {:.3})",
                        type_name, type_stats.count, type_stats.avg_strength, type_stats.avg_importance);
                }
                println!("\nBy layer:");
                for (layer_name, layer_stats) in &stats.by_layer {
                    println!("  {}: {} (avg working: {:.3}, avg core: {:.3})",
                        layer_name, layer_stats.count, layer_stats.avg_working, layer_stats.avg_core);
                }
            }
        }
        
        Commands::Consolidate { ns, days } => {
            mem.consolidate_namespace(days, ns.as_deref())?;
            println!("Consolidation complete ({} days simulated)", days);
        }
        
        Commands::Grant { agent_id, ns, perm } => {
            let perm_str = match perm {
                PermissionArg::Read => "read",
                PermissionArg::Write => "write",
                PermissionArg::Admin => "admin",
            };
            mem.grant(&agent_id, &ns, perm.into())?;
            println!("Granted {} permission to {} on namespace {}", perm_str, agent_id, ns);
        }
        
        Commands::Revoke { agent_id, ns } => {
            mem.revoke(&agent_id, &ns)?;
            println!("Revoked permission from {} on namespace {}", agent_id, ns);
        }
        
        Commands::Permissions { agent_id, json } => {
            let perms = mem.list_permissions(&agent_id)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&perms)?);
            } else {
                if perms.is_empty() {
                    println!("No permissions found for {}", agent_id);
                } else {
                    println!("Permissions for {}:", agent_id);
                    for p in &perms {
                        println!("  {} on {} (granted by {} at {})",
                            p.permission, p.namespace, p.granted_by, p.created_at);
                    }
                }
            }
        }
        
        Commands::Pin { memory_id } => {
            mem.pin(&memory_id)?;
            println!("Pinned memory {}", memory_id);
        }
        
        Commands::Unpin { memory_id } => {
            mem.unpin(&memory_id)?;
            println!("Unpinned memory {}", memory_id);
        }
        
        Commands::Forget { memory_id } => {
            mem.forget(Some(&memory_id), None)?;
            println!("Deleted memory {}", memory_id);
        }
        
        Commands::Reward { feedback, recent } => {
            mem.reward(&feedback, recent)?;
            println!("Applied reward signal to {} recent memories", recent);
        }
    }
    
    Ok(())
}
