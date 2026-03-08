//! ACT-R activation-based retrieval.
//!
//! The core equation from Anderson's ACT-R theory:
//!     A_i = B_i + Σ(W_j · S_ji) + ε
//!
//! Where:
//!     B_i = base-level activation (frequency × recency)
//!     Σ(W_j · S_ji) = spreading activation from current context
//!     ε = noise (stochastic retrieval, not implemented here)
//!
//! Base-level activation (power law of practice and recency):
//!     B_i = ln(Σ_k  t_k^(-d))
//!
//! Where t_k = time since k-th access, d = decay parameter (~0.5)

use chrono::{DateTime, Utc};

use crate::types::MemoryRecord;

/// ACT-R base-level activation.
///
/// B_i = ln(Σ_k (now - t_k)^(-d))
///
/// Higher when accessed more often and more recently.
/// Returns -inf if no accesses (unretrievable).
pub fn base_level_activation(record: &MemoryRecord, now: DateTime<Utc>, decay: f64) -> f64 {
    if record.access_times.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut total = 0.0;
    for t_k in &record.access_times {
        let age_secs = (now - *t_k).num_seconds() as f64;
        let age_secs = age_secs.max(0.001); // Avoid division by zero
        total += age_secs.powf(-decay);
    }

    if total <= 0.0 {
        return f64::NEG_INFINITY;
    }

    total.ln()
}

/// Simple spreading activation from current context.
///
/// In full ACT-R, this uses semantic similarity between context elements
/// and memory chunks. Here we use keyword overlap as a proxy.
///
/// Σ(W_j · S_ji) ≈ weight × (overlap / total_keywords)
pub fn spreading_activation(record: &MemoryRecord, context_keywords: &[String], weight: f64) -> f64 {
    if context_keywords.is_empty() {
        return 0.0;
    }

    let content_lower = record.content.to_lowercase();
    let matches = context_keywords
        .iter()
        .filter(|kw| content_lower.contains(&kw.to_lowercase()))
        .count();

    weight * (matches as f64 / context_keywords.len() as f64)
}

/// Full retrieval activation score.
///
/// A_i = B_i + context_match + importance_boost - contradiction_penalty
///
/// Combines ACT-R base-level with context spreading activation
/// and emotional/importance modulation.
pub fn retrieval_activation(
    record: &MemoryRecord,
    context_keywords: &[String],
    now: DateTime<Utc>,
    base_decay: f64,
    context_weight: f64,
    importance_weight: f64,
    contradiction_penalty: f64,
) -> f64 {
    let base = base_level_activation(record, now, base_decay);

    if base == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    let context = spreading_activation(record, context_keywords, context_weight);

    // Importance modulation (amygdala analog)
    let importance_boost = record.importance * importance_weight;

    // Contradiction penalty
    let penalty = if record.contradicted_by.is_some() {
        contradiction_penalty
    } else {
        0.0
    };

    base + context + importance_boost - penalty
}
