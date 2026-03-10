//! Engram CLI — command-line interface for multi-agent memory.
//!
//! Usage:
//!   engram store "content" --ns trading --type factual --importance 0.8
//!   engram recall "query" --ns "*" --limit 5 --json
//!   engram stats --ns trading
//!   engram consolidate --ns trading
//!   engram grant agent-id --ns namespace --perm read
//!   engram revoke agent-id --ns namespace
//!   engram bus trends
//!   engram bus suggest
//!   engram bus log-outcome check_email --positive

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use engramai::{Memory, MemoryConfig, MemoryType, Permission, EmotionalBus};

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
    
    /// Workspace directory for Emotional Bus (SOUL.md, HEARTBEAT.md, etc.)
    #[arg(short, long, env = "ENGRAM_WORKSPACE")]
    workspace: Option<PathBuf>,
    
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
        
        /// Emotional valence (-1.0 to 1.0)
        #[arg(long, short = 'e')]
        emotion: Option<f64>,
        
        /// Domain for emotional tracking
        #[arg(long)]
        domain: Option<String>,
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
    
    /// Emotional Bus commands
    Bus {
        #[command(subcommand)]
        action: BusAction,
    },
    
    // === Phase 3: Cross-Agent Intelligence ===
    
    /// Subscribe to namespace notifications
    Subscribe {
        /// Agent ID to subscribe
        agent_id: String,
        
        /// Namespace to watch ("*" for all)
        #[arg(long, short = 'n')]
        ns: String,
        
        /// Minimum importance to trigger notification (0.0-1.0)
        #[arg(long, short = 'i', default_value = "0.8")]
        min_importance: f64,
    },
    
    /// Unsubscribe from namespace notifications
    Unsubscribe {
        /// Agent ID to unsubscribe
        agent_id: String,
        
        /// Namespace to stop watching
        #[arg(long, short = 'n')]
        ns: String,
    },
    
    /// Check pending notifications for an agent
    Notifications {
        /// Agent ID to check notifications for
        agent_id: String,
        
        /// Just peek without marking as read
        #[arg(long)]
        peek: bool,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Discover cross-namespace associations
    CrossLinks {
        /// First namespace
        #[arg(long)]
        ns_a: String,
        
        /// Second namespace
        #[arg(long)]
        ns_b: String,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Recall with cross-namespace associations
    RecallAssoc {
        /// Search query
        query: String,
        
        /// Namespace to search (use "*" for all with cross-links)
        #[arg(long, short = 'n', default_value = "*")]
        ns: String,
        
        /// Maximum number of results
        #[arg(long, short = 'l', default_value = "5")]
        limit: usize,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// List subscriptions for an agent
    Subscriptions {
        /// Agent ID to list subscriptions for
        agent_id: String,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
}

#[derive(Subcommand)]
enum BusAction {
    /// Show emotional trends by domain
    Trends {
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Show suggested SOUL/HEARTBEAT updates
    Suggest {
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Log a behavior outcome
    LogOutcome {
        /// Action name (e.g., "check_email", "run_consolidation")
        action: String,
        
        /// Mark outcome as positive
        #[arg(long, conflicts_with = "negative")]
        positive: bool,
        
        /// Mark outcome as negative
        #[arg(long, conflicts_with = "positive")]
        negative: bool,
    },
    
    /// Show behavior statistics
    BehaviorStats {
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
    },
    
    /// Record an emotional event
    RecordEmotion {
        /// Domain (e.g., "coding", "communication")
        domain: String,
        
        /// Emotional valence (-1.0 to 1.0)
        #[arg(long, short = 'v')]
        valence: f64,
    },
    
    /// Check drive alignment for content
    Alignment {
        /// Content to check alignment for
        content: String,
        
        /// Output as JSON
        #[arg(long, short = 'j')]
        json: bool,
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
    
    // Create Memory with or without Emotional Bus
    let mut mem = if let Some(ref workspace) = cli.workspace {
        let ws_path = workspace.to_str().ok_or("invalid workspace path")?;
        Memory::with_emotional_bus(db_path, ws_path, Some(MemoryConfig::default()))?
    } else {
        Memory::new(db_path, Some(MemoryConfig::default()))?
    };
    
    if let Some(agent_id) = &cli.agent_id {
        mem.set_agent_id(agent_id);
    }
    
    match cli.command {
        Commands::Store { content, ns, r#type, importance, source, emotion, domain } => {
            // If emotion is provided, use add_with_emotion
            let id = if let (Some(em), Some(dom)) = (emotion, domain.as_ref()) {
                mem.add_with_emotion(
                    &content,
                    r#type.into(),
                    importance,
                    source.as_deref(),
                    None,
                    Some(&ns),
                    em,
                    dom,
                )?
            } else {
                mem.add_to_namespace(
                    &content,
                    r#type.into(),
                    importance,
                    source.as_deref(),
                    None,
                    Some(&ns),
                )?
            };
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
        
        Commands::Bus { action } => {
            // Bus commands require workspace
            let workspace = cli.workspace.as_ref()
                .ok_or("Emotional Bus commands require --workspace")?;
            let ws_path = workspace.to_str().ok_or("invalid workspace path")?;
            
            // Create bus directly if not already attached
            let bus = EmotionalBus::new(ws_path, mem.connection())?;
            
            match action {
                BusAction::Trends { json } => {
                    let trends = bus.get_trends(mem.connection())?;
                    
                    if json {
                        println!("{}", serde_json::to_string_pretty(&trends)?);
                    } else {
                        if trends.is_empty() {
                            println!("No emotional trends recorded yet.");
                        } else {
                            println!("Emotional Trends:");
                            for trend in &trends {
                                let flag = if trend.needs_soul_update() { " ⚠️ needs update" } else { "" };
                                println!("  {}: {:.2} avg over {} events{}",
                                    trend.domain, trend.valence, trend.count, flag);
                            }
                        }
                    }
                }
                
                BusAction::Suggest { json } => {
                    let soul_updates = bus.suggest_soul_updates(mem.connection())?;
                    let heartbeat_updates = bus.suggest_heartbeat_updates(mem.connection())?;
                    
                    if json {
                        let combined = serde_json::json!({
                            "soul_updates": soul_updates,
                            "heartbeat_updates": heartbeat_updates,
                        });
                        println!("{}", serde_json::to_string_pretty(&combined)?);
                    } else {
                        if soul_updates.is_empty() && heartbeat_updates.is_empty() {
                            println!("No suggested updates at this time.");
                        } else {
                            if !soul_updates.is_empty() {
                                println!("SOUL.md Suggestions:");
                                for s in &soul_updates {
                                    println!("  [{}/{}] {}", s.domain, s.action, s.content);
                                }
                            }
                            if !heartbeat_updates.is_empty() {
                                println!("\nHEARTBEAT.md Suggestions:");
                                for h in &heartbeat_updates {
                                    println!("  [{}] {} (score: {:.0}%, {} attempts)",
                                        h.suggestion, h.action, h.stats.score * 100.0, h.stats.total);
                                }
                            }
                        }
                    }
                }
                
                BusAction::LogOutcome { action, positive, negative } => {
                    let outcome = if positive {
                        true
                    } else if negative {
                        false
                    } else {
                        return Err("Must specify --positive or --negative".into());
                    };
                    
                    bus.log_behavior(mem.connection(), &action, outcome)?;
                    let outcome_str = if outcome { "positive" } else { "negative" };
                    println!("Logged {} outcome for '{}'", outcome_str, action);
                }
                
                BusAction::BehaviorStats { json } => {
                    let stats = bus.get_behavior_stats(mem.connection())?;
                    
                    if json {
                        println!("{}", serde_json::to_string_pretty(&stats)?);
                    } else {
                        if stats.is_empty() {
                            println!("No behavior statistics recorded yet.");
                        } else {
                            println!("Behavior Statistics:");
                            for s in &stats {
                                let flag = if s.should_deprioritize() { " ⚠️ deprioritize" } else { "" };
                                println!("  {}: {:.0}% success ({}/{} positive){}",
                                    s.action, s.score * 100.0, s.positive, s.total, flag);
                            }
                        }
                    }
                }
                
                BusAction::RecordEmotion { domain, valence } => {
                    bus.process_interaction(mem.connection(), "", valence, &domain)?;
                    println!("Recorded emotion {:.2} for domain '{}'", valence, domain);
                }
                
                BusAction::Alignment { content, json } => {
                    let score = bus.alignment_score(&content);
                    let boost = bus.align_importance(&content);
                    let aligned = bus.find_aligned(&content);
                    
                    if json {
                        let result = serde_json::json!({
                            "score": score,
                            "importance_boost": boost,
                            "aligned_drives": aligned,
                        });
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    } else {
                        println!("Alignment score: {:.2}", score);
                        println!("Importance boost: {:.2}x", boost);
                        if !aligned.is_empty() {
                            println!("Aligned drives:");
                            for (name, s) in &aligned {
                                println!("  {}: {:.2}", name, s);
                            }
                        }
                    }
                }
            }
        }
        
        // === Phase 3: Cross-Agent Intelligence ===
        
        Commands::Subscribe { agent_id, ns, min_importance } => {
            mem.subscribe(&agent_id, &ns, min_importance)?;
            println!("Subscribed {} to namespace '{}' (min_importance: {:.2})", 
                agent_id, ns, min_importance);
        }
        
        Commands::Unsubscribe { agent_id, ns } => {
            let removed = mem.unsubscribe(&agent_id, &ns)?;
            if removed {
                println!("Unsubscribed {} from namespace '{}'", agent_id, ns);
            } else {
                println!("No subscription found for {} on namespace '{}'", agent_id, ns);
            }
        }
        
        Commands::Notifications { agent_id, peek, json } => {
            let notifs = if peek {
                mem.peek_notifications(&agent_id)?
            } else {
                mem.check_notifications(&agent_id)?
            };
            
            if json {
                println!("{}", serde_json::to_string_pretty(&notifs)?);
            } else {
                if notifs.is_empty() {
                    println!("No pending notifications for {}", agent_id);
                } else {
                    println!("Notifications for {} ({}):", agent_id, notifs.len());
                    for n in &notifs {
                        println!("  [{}:{}] ({:.2}) {}", 
                            n.namespace, n.memory_id, n.importance, 
                            if n.content.len() > 60 {
                                format!("{}...", &n.content[..60])
                            } else {
                                n.content.clone()
                            }
                        );
                    }
                    if peek {
                        println!("\n(peeked - not marked as read)");
                    }
                }
            }
        }
        
        Commands::CrossLinks { ns_a, ns_b, json } => {
            let links = mem.discover_cross_links(&ns_a, &ns_b)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&links)?);
            } else {
                if links.is_empty() {
                    println!("No cross-namespace links found between '{}' and '{}'", ns_a, ns_b);
                } else {
                    println!("Cross-namespace links between '{}' and '{}' ({}):", ns_a, ns_b, links.len());
                    for link in &links {
                        println!("  {} ↔ {} (strength: {:.2}, coactivations: {})",
                            link.source_id, link.target_id, link.strength, link.coactivation_count);
                    }
                }
            }
        }
        
        Commands::RecallAssoc { query, ns, limit, json } => {
            let ns_opt = if ns == "default" { None } else { Some(ns.as_str()) };
            let result = mem.recall_with_associations(&query, ns_opt, limit)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                if result.memories.is_empty() {
                    println!("No memories found.");
                } else {
                    println!("Memories ({}):", result.memories.len());
                    for r in &result.memories {
                        println!("  [{}] ({:.2}) {}", r.record.id, r.confidence, r.record.content);
                    }
                    
                    if !result.cross_links.is_empty() {
                        println!("\nCross-namespace associations ({}):", result.cross_links.len());
                        for link in &result.cross_links {
                            let desc = link.description.as_ref()
                                .map(|d| if d.len() > 40 { format!("{}...", &d[..40]) } else { d.clone() })
                                .unwrap_or_default();
                            println!("  {}:{} → {}:{} ({:.2}) {}", 
                                link.source_ns, link.source_id, 
                                link.target_ns, link.target_id,
                                link.strength, desc);
                        }
                    }
                }
            }
        }
        
        Commands::Subscriptions { agent_id, json } => {
            let subs = mem.list_subscriptions(&agent_id)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&subs)?);
            } else {
                if subs.is_empty() {
                    println!("No subscriptions for {}", agent_id);
                } else {
                    println!("Subscriptions for {} ({}):", agent_id, subs.len());
                    for sub in &subs {
                        println!("  {} (min_importance: {:.2}, since: {})",
                            sub.namespace, sub.min_importance, sub.created_at.format("%Y-%m-%d %H:%M"));
                    }
                }
            }
        }
    }
    
    Ok(())
}
