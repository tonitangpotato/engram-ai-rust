//! Main Memory API — simplified interface to Engram's cognitive models.

use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

use crate::bus::EmotionalBus;
use crate::config::MemoryConfig;
use crate::models::{effective_strength, retrieval_activation, run_consolidation_cycle};
use crate::storage::Storage;
use crate::bus::{SubscriptionManager, Subscription, Notification};
use crate::models::hebbian::{MemoryWithNamespace, record_cross_namespace_coactivation};
use crate::types::{AclEntry, CrossLink, HebbianLink, LayerStats, MemoryLayer, MemoryRecord, MemoryStats, MemoryType, Permission, RecallResult, RecallWithAssociationsResult, TypeStats};

/// Main interface to the Engram memory system.
///
/// Wraps the neuroscience math models behind a clean API.
/// All complexity is hidden — you just add, recall, and consolidate.
pub struct Memory {
    storage: Storage,
    config: MemoryConfig,
    created_at: chrono::DateTime<Utc>,
    /// Agent ID for this memory instance (used for ACL checks)
    agent_id: Option<String>,
    /// Optional Emotional Bus for drive alignment and emotional tracking
    emotional_bus: Option<EmotionalBus>,
}

impl Memory {
    /// Initialize Engram memory system.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to SQLite database file. Created if it doesn't exist.
    ///           Use `:memory:` for in-memory (non-persistent) operation.
    /// * `config` - MemoryConfig with tunable parameters. None = literature defaults.
    pub fn new(path: &str, config: Option<MemoryConfig>) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = Storage::new(path)?;
        let config = config.unwrap_or_default();
        let created_at = Utc::now();

        Ok(Self {
            storage,
            config,
            created_at,
            agent_id: None,
            emotional_bus: None,
        })
    }
    
    /// Create a Memory instance with an Emotional Bus attached.
    ///
    /// The Emotional Bus connects memory to workspace files (SOUL.md, HEARTBEAT.md)
    /// for drive alignment and emotional feedback loops.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to SQLite database file
    /// * `workspace_dir` - Path to the agent workspace directory
    /// * `config` - Optional MemoryConfig
    pub fn with_emotional_bus(
        path: &str,
        workspace_dir: &str,
        config: Option<MemoryConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = Storage::new(path)?;
        let config = config.unwrap_or_default();
        let created_at = Utc::now();
        
        // Create Emotional Bus using storage's connection
        let emotional_bus = Some(EmotionalBus::new(workspace_dir, storage.connection())?);
        
        Ok(Self {
            storage,
            config,
            created_at,
            agent_id: None,
            emotional_bus,
        })
    }
    
    /// Get a reference to the Emotional Bus, if attached.
    pub fn emotional_bus(&self) -> Option<&EmotionalBus> {
        self.emotional_bus.as_ref()
    }
    
    /// Get a mutable reference to the Emotional Bus, if attached.
    pub fn emotional_bus_mut(&mut self) -> Option<&mut EmotionalBus> {
        self.emotional_bus.as_mut()
    }
    
    /// Get a reference to the underlying storage connection.
    pub fn connection(&self) -> &rusqlite::Connection {
        self.storage.connection()
    }
    
    /// Set the agent ID for this memory instance.
    /// 
    /// This is used for ACL checks when storing and recalling memories.
    /// Each agent should identify itself before performing operations.
    pub fn set_agent_id(&mut self, id: &str) {
        self.agent_id = Some(id.to_string());
    }
    
    /// Get the current agent ID.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    /// Store a new memory. Returns memory ID.
    ///
    /// The memory is encoded with initial working_strength=1.0 (strong
    /// hippocampal trace) and core_strength=0.0 (no neocortical trace yet).
    /// Consolidation cycles will gradually transfer it to core.
    ///
    /// # Arguments
    ///
    /// * `content` - The memory content (natural language)
    /// * `memory_type` - Memory type classification
    /// * `importance` - 0-1 importance score (None = auto from type)
    /// * `source` - Source identifier (e.g., filename, conversation ID)
    /// * `metadata` - Optional structured metadata (e.g., for causal memories)
    pub fn add(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: Option<f64>,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.add_to_namespace(content, memory_type, importance, source, metadata, None)
    }
    
    /// Store a new memory in a specific namespace. Returns memory ID.
    ///
    /// # Arguments
    ///
    /// * `content` - The memory content (natural language)
    /// * `memory_type` - Memory type classification
    /// * `importance` - 0-1 importance score (None = auto from type)
    /// * `source` - Source identifier (e.g., filename, conversation ID)
    /// * `metadata` - Optional structured metadata (e.g., for causal memories)
    /// * `namespace` - Namespace to store in (None = "default")
    pub fn add_to_namespace(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: Option<f64>,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
        namespace: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let ns = namespace.unwrap_or("default");
        let id = format!("{}", Uuid::new_v4())[..8].to_string();
        let base_importance = importance.unwrap_or_else(|| memory_type.default_importance());
        
        // Apply drive alignment boost if Emotional Bus is attached
        let importance = if let Some(ref bus) = self.emotional_bus {
            let boost = bus.align_importance(content);
            (base_importance * boost).min(1.0) // Cap at 1.0
        } else {
            base_importance
        };

        let record = MemoryRecord {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            layer: MemoryLayer::Working,
            created_at: Utc::now(),
            access_times: vec![Utc::now()],
            working_strength: 1.0,
            core_strength: 0.0,
            importance,
            pinned: false,
            consolidation_count: 0,
            last_consolidated: None,
            source: source.unwrap_or("").to_string(),
            contradicts: None,
            contradicted_by: None,
            metadata,
        };

        self.storage.add(&record, ns)?;
        Ok(id)
    }
    
    /// Store a new memory with emotional tracking.
    ///
    /// This method both stores the memory and records the emotional valence
    /// in the Emotional Bus for trend tracking. Requires an Emotional Bus
    /// to be attached.
    ///
    /// # Arguments
    ///
    /// * `content` - The memory content
    /// * `memory_type` - Memory type classification
    /// * `importance` - 0-1 importance score (None = auto)
    /// * `source` - Source identifier
    /// * `metadata` - Optional metadata
    /// * `namespace` - Namespace to store in
    /// * `emotion` - Emotional valence (-1.0 to 1.0)
    /// * `domain` - Domain for emotional tracking
    pub fn add_with_emotion(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: Option<f64>,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
        namespace: Option<&str>,
        emotion: f64,
        domain: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Store the memory (with importance boost from alignment)
        let id = self.add_to_namespace(content, memory_type, importance, source, metadata, namespace)?;
        
        // Record emotion if bus is attached
        if let Some(ref bus) = self.emotional_bus {
            bus.process_interaction(self.storage.connection(), content, emotion, domain)?;
        }
        
        Ok(id)
    }

    /// Retrieve relevant memories using ACT-R activation-based retrieval.
    ///
    /// Unlike simple cosine similarity, this uses:
    /// - Base-level activation (frequency × recency, power law)
    /// - Spreading activation from context keywords
    /// - Importance modulation (emotional memories are more accessible)
    ///
    /// Results include a confidence score (metacognitive monitoring)
    /// that tells you how "trustworthy" each retrieval is.
    ///
    /// # Arguments
    ///
    /// * `query` - Natural language query
    /// * `limit` - Maximum number of results
    /// * `context` - Additional context keywords to boost relevant memories
    /// * `min_confidence` - Minimum confidence threshold (0-1)
    pub fn recall(
        &mut self,
        query: &str,
        limit: usize,
        context: Option<Vec<String>>,
        min_confidence: Option<f64>,
    ) -> Result<Vec<RecallResult>, Box<dyn std::error::Error>> {
        self.recall_from_namespace(query, limit, context, min_confidence, None)
    }
    
    /// Retrieve relevant memories from a specific namespace.
    ///
    /// # Arguments
    ///
    /// * `query` - Natural language query
    /// * `limit` - Maximum number of results
    /// * `context` - Additional context keywords to boost relevant memories
    /// * `min_confidence` - Minimum confidence threshold (0-1)
    /// * `namespace` - Namespace to search (None = "default", Some("*") = all namespaces)
    pub fn recall_from_namespace(
        &mut self,
        query: &str,
        limit: usize,
        context: Option<Vec<String>>,
        min_confidence: Option<f64>,
        namespace: Option<&str>,
    ) -> Result<Vec<RecallResult>, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let context = context.unwrap_or_default();
        let min_conf = min_confidence.unwrap_or(0.0);
        let ns = namespace.unwrap_or("default");

        // Get candidate memories via FTS (namespace-aware)
        let candidates = self.storage.search_fts_ns(query, limit * 3, Some(ns))?;

        // Score each candidate with ACT-R activation
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|record| {
                let activation = retrieval_activation(
                    &record,
                    &context,
                    now,
                    self.config.actr_decay,
                    self.config.context_weight,
                    self.config.importance_weight,
                    self.config.contradiction_penalty,
                );
                (record, activation)
            })
            .filter(|(_, act)| *act > f64::NEG_INFINITY)
            .collect();

        // Sort by activation descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k and compute confidence
        let results: Vec<_> = scored
            .into_iter()
            .take(limit)
            .map(|(record, activation)| {
                let confidence = self.compute_confidence(&record, activation);
                let confidence_label = confidence_label(confidence);

                RecallResult {
                    record,
                    activation,
                    confidence,
                    confidence_label,
                }
            })
            .filter(|r| r.confidence >= min_conf)
            .collect();

        // Record access for all retrieved memories (ACT-R learning)
        for result in &results {
            self.storage.record_access(&result.record.id)?;
        }

        // Hebbian learning: record co-activation (namespace-aware)
        if self.config.hebbian_enabled && results.len() >= 2 {
            let memory_ids: Vec<_> = results.iter().map(|r| r.record.id.clone()).collect();
            crate::models::record_coactivation_ns(
                &mut self.storage,
                &memory_ids,
                self.config.hebbian_threshold,
                ns,
            )?;
        }

        Ok(results)
    }

    /// Run a consolidation cycle ("sleep replay").
    ///
    /// This is the core of memory maintenance. Based on Murre & Chessa's
    /// Memory Chain Model, it:
    ///
    /// 1. Decays working_strength (hippocampal traces fade)
    /// 2. Transfers knowledge to core_strength (neocortical consolidation)
    /// 3. Replays archived memories (prevents catastrophic forgetting)
    /// 4. Rebalances layers (promote strong → core, demote weak → archive)
    ///
    /// Call this periodically — once per "day" of agent operation,
    /// or after significant learning sessions.
    ///
    /// # Arguments
    ///
    /// * `days` - Simulated time step in days (1.0 = one day of consolidation)
    pub fn consolidate(&mut self, days: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.consolidate_namespace(days, None)
    }
    
    /// Run a consolidation cycle for a specific namespace.
    ///
    /// # Arguments
    ///
    /// * `days` - Simulated time step in days (1.0 = one day of consolidation)
    /// * `namespace` - Namespace to consolidate (None = all namespaces)
    pub fn consolidate_namespace(
        &mut self,
        days: f64,
        namespace: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        run_consolidation_cycle(&mut self.storage, days, &self.config, namespace)?;

        // Decay Hebbian links
        if self.config.hebbian_enabled {
            self.storage.decay_hebbian_links(self.config.hebbian_decay)?;
        }

        Ok(())
    }

    /// Forget a specific memory or prune all below threshold.
    ///
    /// If memory_id is given, removes that specific memory.
    /// Otherwise, prunes all memories whose effective_strength
    /// is below threshold (moves them to archive).
    pub fn forget(
        &mut self,
        memory_id: Option<&str>,
        threshold: Option<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threshold = threshold.unwrap_or(self.config.forget_threshold);

        if let Some(id) = memory_id {
            self.storage.delete(id)?;
        } else {
            // Prune all weak memories
            let now = Utc::now();
            let all = self.storage.all()?;
            for record in all {
                if !record.pinned && effective_strength(&record, now) < threshold {
                    if record.layer != MemoryLayer::Archive {
                        let mut updated = record;
                        updated.layer = MemoryLayer::Archive;
                        self.storage.update(&updated)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Process user feedback as a dopaminergic reward signal.
    ///
    /// Detects positive/negative sentiment and applies reward modulation
    /// to recently accessed memories.
    pub fn reward(&mut self, feedback: &str, recent_n: usize) -> Result<(), Box<dyn std::error::Error>> {
        let polarity = detect_feedback_polarity(feedback);

        if polarity == 0.0 {
            return Ok(()); // Neutral feedback
        }

        // Get recently accessed memories
        let all = self.storage.all()?;
        let _now = Utc::now();
        let mut recent: Vec<_> = all
            .into_iter()
            .filter(|r| !r.access_times.is_empty())
            .collect();
        recent.sort_by_key(|r| std::cmp::Reverse(r.access_times.last().cloned()));

        // Apply reward to top-N recent
        for mut record in recent.into_iter().take(recent_n) {
            if polarity > 0.0 {
                // Positive feedback: boost working strength
                record.working_strength += self.config.reward_magnitude * polarity;
                record.working_strength = record.working_strength.min(2.0);
            } else {
                // Negative feedback: suppress working strength
                record.working_strength *= 1.0 + polarity * 0.1; // polarity is negative
                record.working_strength = record.working_strength.max(0.0);
            }
            self.storage.update(&record)?;
        }

        Ok(())
    }

    /// Global synaptic downscaling — normalize all memory weights.
    ///
    /// Based on Tononi & Cirelli's Synaptic Homeostasis Hypothesis.
    pub fn downscale(&mut self, factor: Option<f64>) -> Result<usize, Box<dyn std::error::Error>> {
        let factor = factor.unwrap_or(self.config.downscale_factor);
        let all = self.storage.all()?;
        let mut count = 0;

        for mut record in all {
            if !record.pinned {
                record.working_strength *= factor;
                record.core_strength *= factor;
                self.storage.update(&record)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Memory system statistics.
    pub fn stats(&self) -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let all = self.storage.all()?;
        let now = Utc::now();

        let mut by_type: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut by_layer: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut pinned = 0;

        for record in &all {
            by_type
                .entry(record.memory_type.to_string())
                .or_default()
                .push(record);
            by_layer
                .entry(record.layer.to_string())
                .or_default()
                .push(record);
            if record.pinned {
                pinned += 1;
            }
        }

        let type_stats: HashMap<String, TypeStats> = by_type
            .into_iter()
            .map(|(type_name, records)| {
                let count = records.len();
                let avg_strength = records
                    .iter()
                    .map(|r| effective_strength(r, now))
                    .sum::<f64>()
                    / count as f64;
                let avg_importance = records.iter().map(|r| r.importance).sum::<f64>() / count as f64;

                (
                    type_name,
                    TypeStats {
                        count,
                        avg_strength,
                        avg_importance,
                    },
                )
            })
            .collect();

        let layer_stats: HashMap<String, LayerStats> = by_layer
            .into_iter()
            .map(|(layer_name, records)| {
                let count = records.len();
                let avg_working = records.iter().map(|r| r.working_strength).sum::<f64>() / count as f64;
                let avg_core = records.iter().map(|r| r.core_strength).sum::<f64>() / count as f64;

                (
                    layer_name,
                    LayerStats {
                        count,
                        avg_working,
                        avg_core,
                    },
                )
            })
            .collect();

        let uptime_hours = (now - self.created_at).num_seconds() as f64 / 3600.0;

        Ok(MemoryStats {
            total_memories: all.len(),
            by_type: type_stats,
            by_layer: layer_stats,
            pinned,
            uptime_hours,
        })
    }

    /// Pin a memory — it won't decay or be pruned.
    pub fn pin(&mut self, memory_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut record) = self.storage.get(memory_id)? {
            record.pinned = true;
            self.storage.update(&record)?;
        }
        Ok(())
    }

    /// Unpin a memory — it will resume normal decay.
    pub fn unpin(&mut self, memory_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut record) = self.storage.get(memory_id)? {
            record.pinned = false;
            self.storage.update(&record)?;
        }
        Ok(())
    }

    /// Get Hebbian links for a specific memory.
    pub fn hebbian_links(&self, memory_id: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(self.storage.get_hebbian_neighbors(memory_id)?)
    }
    
    /// Get Hebbian links for a specific memory, filtered by namespace.
    pub fn hebbian_links_ns(
        &self,
        memory_id: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(self.storage.get_hebbian_neighbors_ns(memory_id, namespace)?)
    }
    
    // === ACL Methods ===
    
    /// Grant a permission to an agent for a namespace.
    /// 
    /// Only agents with admin permission on the namespace (or wildcard admin)
    /// can grant permissions. If no agent_id is set, uses "system" as grantor.
    pub fn grant(
        &mut self,
        agent_id: &str,
        namespace: &str,
        permission: Permission,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let grantor = self.agent_id.clone().unwrap_or_else(|| "system".to_string());
        self.storage.grant_permission(agent_id, namespace, permission, &grantor)?;
        Ok(())
    }
    
    /// Revoke a permission from an agent for a namespace.
    pub fn revoke(&mut self, agent_id: &str, namespace: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.storage.revoke_permission(agent_id, namespace)?;
        Ok(())
    }
    
    /// Check if an agent has a specific permission for a namespace.
    pub fn check_permission(
        &self,
        agent_id: &str,
        namespace: &str,
        permission: Permission,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(self.storage.check_permission(agent_id, namespace, permission)?)
    }
    
    /// List all permissions for an agent.
    pub fn list_permissions(&self, agent_id: &str) -> Result<Vec<AclEntry>, Box<dyn std::error::Error>> {
        Ok(self.storage.list_permissions(agent_id)?)
    }
    
    /// Get statistics for a specific namespace.
    pub fn stats_ns(&self, namespace: Option<&str>) -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let all = self.storage.all_in_namespace(namespace)?;
        let now = Utc::now();

        let mut by_type: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut by_layer: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut pinned = 0;

        for record in &all {
            by_type
                .entry(record.memory_type.to_string())
                .or_default()
                .push(record);
            by_layer
                .entry(record.layer.to_string())
                .or_default()
                .push(record);
            if record.pinned {
                pinned += 1;
            }
        }

        let type_stats: HashMap<String, TypeStats> = by_type
            .into_iter()
            .map(|(type_name, records)| {
                let count = records.len();
                let avg_strength = records
                    .iter()
                    .map(|r| effective_strength(r, now))
                    .sum::<f64>()
                    / count as f64;
                let avg_importance = records.iter().map(|r| r.importance).sum::<f64>() / count as f64;

                (
                    type_name,
                    TypeStats {
                        count,
                        avg_strength,
                        avg_importance,
                    },
                )
            })
            .collect();

        let layer_stats: HashMap<String, LayerStats> = by_layer
            .into_iter()
            .map(|(layer_name, records)| {
                let count = records.len();
                let avg_working = records.iter().map(|r| r.working_strength).sum::<f64>() / count as f64;
                let avg_core = records.iter().map(|r| r.core_strength).sum::<f64>() / count as f64;

                (
                    layer_name,
                    LayerStats {
                        count,
                        avg_working,
                        avg_core,
                    },
                )
            })
            .collect();

        let uptime_hours = (now - self.created_at).num_seconds() as f64 / 3600.0;

        Ok(MemoryStats {
            total_memories: all.len(),
            by_type: type_stats,
            by_layer: layer_stats,
            pinned,
            uptime_hours,
        })
    }
    
    // === Phase 3: Cross-Agent Intelligence ===
    
    /// Recall memories with cross-namespace associations.
    ///
    /// When using namespace="*", this also returns Hebbian links that span
    /// across different namespaces, enabling cross-domain intelligence.
    ///
    /// # Arguments
    ///
    /// * `query` - Natural language query
    /// * `namespace` - Namespace to search ("*" for all)
    /// * `limit` - Maximum number of results
    pub fn recall_with_associations(
        &mut self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<RecallWithAssociationsResult, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let ns = namespace.unwrap_or("default");
        
        // Get candidate memories via FTS
        let candidates = self.storage.search_fts_ns(query, limit * 3, Some(ns))?;
        
        // Score each candidate with ACT-R activation
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|record| {
                let activation = retrieval_activation(
                    &record,
                    &[],
                    now,
                    self.config.actr_decay,
                    self.config.context_weight,
                    self.config.importance_weight,
                    self.config.contradiction_penalty,
                );
                (record, activation)
            })
            .filter(|(_, act)| *act > f64::NEG_INFINITY)
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        let results: Vec<_> = scored
            .into_iter()
            .take(limit)
            .map(|(record, activation)| {
                let confidence = self.compute_confidence(&record, activation);
                let confidence_label = confidence_label(confidence);
                
                RecallResult {
                    record,
                    activation,
                    confidence,
                    confidence_label,
                }
            })
            .collect();
        
        // Record access for all retrieved memories
        for result in &results {
            self.storage.record_access(&result.record.id)?;
        }
        
        // Collect cross-namespace associations
        let mut cross_links = Vec::new();
        
        // For wildcard namespace queries, also collect cross-namespace Hebbian neighbors
        if ns == "*" && results.len() >= 2 {
            // Get namespaces for all retrieved memories
            let mut memories_with_ns: Vec<MemoryWithNamespace> = Vec::new();
            
            for result in &results {
                if let Some(mem_ns) = self.storage.get_namespace(&result.record.id)? {
                    memories_with_ns.push(MemoryWithNamespace {
                        id: result.record.id.clone(),
                        namespace: mem_ns,
                    });
                }
            }
            
            // Record cross-namespace co-activation
            if self.config.hebbian_enabled {
                record_cross_namespace_coactivation(
                    &mut self.storage,
                    &memories_with_ns,
                    self.config.hebbian_threshold,
                )?;
            }
            
            // Collect cross-links from all retrieved memories
            for result in &results {
                let links = self.storage.get_cross_namespace_neighbors(&result.record.id)?;
                cross_links.extend(links);
            }
            
            // Deduplicate by (source_id, target_id)
            cross_links.sort_by(|a, b| {
                (&a.source_id, &a.target_id).cmp(&(&b.source_id, &b.target_id))
            });
            cross_links.dedup_by(|a, b| a.source_id == b.source_id && a.target_id == b.target_id);
            
            // Sort by strength descending
            cross_links.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        }
        
        Ok(RecallWithAssociationsResult {
            memories: results,
            cross_links,
        })
    }
    
    /// Discover cross-namespace Hebbian links between two namespaces.
    ///
    /// Returns all Hebbian associations that span across the given namespaces.
    /// ACL-aware: only returns links between namespaces the agent can read.
    pub fn discover_cross_links(
        &self,
        namespace_a: &str,
        namespace_b: &str,
    ) -> Result<Vec<HebbianLink>, Box<dyn std::error::Error>> {
        // ACL check if agent_id is set
        if let Some(ref agent_id) = self.agent_id {
            let can_read_a = self.storage.check_permission(agent_id, namespace_a, Permission::Read)?;
            let can_read_b = self.storage.check_permission(agent_id, namespace_b, Permission::Read)?;
            
            if !can_read_a || !can_read_b {
                return Ok(vec![]); // No access to one or both namespaces
            }
        }
        
        Ok(self.storage.discover_cross_links(namespace_a, namespace_b)?)
    }
    
    /// Get all cross-namespace associations for a memory.
    pub fn get_cross_associations(
        &self,
        memory_id: &str,
    ) -> Result<Vec<CrossLink>, Box<dyn std::error::Error>> {
        Ok(self.storage.get_cross_namespace_neighbors(memory_id)?)
    }
    
    // === Subscription/Notification Methods ===
    
    /// Subscribe to notifications for a namespace.
    ///
    /// The agent will receive notifications when new memories are stored
    /// with importance >= min_importance in the specified namespace.
    pub fn subscribe(
        &self,
        agent_id: &str,
        namespace: &str,
        min_importance: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mgr = SubscriptionManager::new(self.storage.connection())?;
        mgr.subscribe(agent_id, namespace, min_importance)?;
        Ok(())
    }
    
    /// Unsubscribe from a namespace.
    pub fn unsubscribe(
        &self,
        agent_id: &str,
        namespace: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mgr = SubscriptionManager::new(self.storage.connection())?;
        Ok(mgr.unsubscribe(agent_id, namespace)?)
    }
    
    /// List subscriptions for an agent.
    pub fn list_subscriptions(
        &self,
        agent_id: &str,
    ) -> Result<Vec<Subscription>, Box<dyn std::error::Error>> {
        let mgr = SubscriptionManager::new(self.storage.connection())?;
        Ok(mgr.list_subscriptions(agent_id)?)
    }
    
    /// Check for notifications since last check.
    ///
    /// Returns new memories that exceed the subscription thresholds.
    /// Updates the cursor so the same notifications aren't returned twice.
    pub fn check_notifications(
        &self,
        agent_id: &str,
    ) -> Result<Vec<Notification>, Box<dyn std::error::Error>> {
        let mgr = SubscriptionManager::new(self.storage.connection())?;
        Ok(mgr.check_notifications(agent_id)?)
    }
    
    /// Peek at notifications without updating cursor.
    pub fn peek_notifications(
        &self,
        agent_id: &str,
    ) -> Result<Vec<Notification>, Box<dyn std::error::Error>> {
        let mgr = SubscriptionManager::new(self.storage.connection())?;
        Ok(mgr.peek_notifications(agent_id)?)
    }

    fn compute_confidence(&self, record: &MemoryRecord, activation: f64) -> f64 {
        // Simple confidence heuristic: normalize activation + importance
        let normalized_activation = (activation + 10.0) / 20.0; // Rough normalization
        let confidence = (normalized_activation.max(0.0).min(1.0) * 0.7) + (record.importance * 0.3);
        confidence.max(0.0).min(1.0)
    }
}

fn confidence_label(confidence: f64) -> String {
    match confidence {
        c if c >= 0.8 => "high".to_string(),
        c if c >= 0.5 => "medium".to_string(),
        c if c >= 0.2 => "low".to_string(),
        _ => "very low".to_string(),
    }
}

fn detect_feedback_polarity(feedback: &str) -> f64 {
    let lower = feedback.to_lowercase();
    let positive = ["good", "great", "excellent", "correct", "right", "yes", "nice", "perfect"];
    let negative = ["bad", "wrong", "incorrect", "no", "error", "mistake", "poor"];

    let pos_count = positive.iter().filter(|&w| lower.contains(w)).count();
    let neg_count = negative.iter().filter(|&w| lower.contains(w)).count();

    if pos_count > neg_count {
        1.0
    } else if neg_count > pos_count {
        -1.0
    } else {
        0.0
    }
}
