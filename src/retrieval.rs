//! Advanced retrieval pipeline: RRF fusion + MMR diversity filtering.
//!
//! Inspired by memory-lancedb-pro's pipeline but grounded in cognitive science:
//!
//! 1. **Candidate generation**: FTS + vector search (parallel channels)
//! 2. **RRF fusion**: Reciprocal Rank Fusion merges ranked lists without score calibration
//! 3. **Cognitive scoring**: ACT-R activation blended with RRF score
//! 4. **MMR diversity**: Maximal Marginal Relevance suppresses near-duplicates
//! 5. **Noise filtering**: Skip low-value content at retrieval time

use std::collections::{HashMap, HashSet};

use crate::types::MemoryRecord;

/// RRF constant (k=60 is standard from the original paper).
const RRF_K: f64 = 60.0;

/// Retrieval pipeline configuration.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Weight for cognitive (ACT-R) score in final blend.
    pub cognitive_weight: f64,
    /// Weight for semantic (vector) score in final blend.
    pub semantic_weight: f64,
    /// Weight for keyword (FTS) score in final blend.
    pub keyword_weight: f64,
    /// MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity.
    pub mmr_lambda: f64,
    /// Minimum content length to not be considered noise.
    pub min_content_length: usize,
    /// Candidate pool multiplier (fetch limit * multiplier candidates).
    pub candidate_multiplier: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            cognitive_weight: 0.50,
            semantic_weight: 0.35,
            keyword_weight: 0.15,
            mmr_lambda: 0.7,
            min_content_length: 4,
            candidate_multiplier: 4,
        }
    }
}

/// A scored candidate in the retrieval pipeline.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub record: MemoryRecord,
    /// Blended final score.
    pub score: f64,
    /// Individual component scores for observability.
    pub components: ScoreComponents,
}

#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    pub rrf_score: f64,
    pub cognitive_score: f64,
    pub semantic_score: f64,
    pub keyword_rank: usize,
    pub vector_rank: usize,
}

/// Noise words/patterns that indicate low-value content.
const NOISE_PATTERNS: &[&str] = &[
    "ok",
    "okay",
    "sure",
    "yes",
    "no",
    "thanks",
    "thank you",
    "hi",
    "hello",
    "hey",
    "bye",
    "goodbye",
    "hmm",
    "hm",
    "um",
    "uh",
    "ah",
    "got it",
    "sounds good",
    "makes sense",
];

/// Check if content is noise (greetings, acknowledgments, etc.).
pub fn is_noise(content: &str) -> bool {
    let lower = content.trim().to_lowercase();
    if lower.len() < 3 {
        return true;
    }
    NOISE_PATTERNS.iter().any(|p| lower == *p)
}

/// Reciprocal Rank Fusion: merge multiple ranked lists into a single ranking.
///
/// RRF(d) = Sigma_{r in R} 1 / (k + rank_r(d))
///
/// Where k=60 (standard), rank_r(d) is the rank of document d in ranked list r.
/// Documents not present in a list get rank = list_length + 1.
pub fn rrf_fuse(ranked_lists: &[Vec<String>], k: f64) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for list in ranked_lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }
    }

    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Maximal Marginal Relevance selection.
///
/// Iteratively selects the candidate that maximizes:
///   MMR = lambda * relevance(d) - (1 - lambda) * max_sim(d, selected)
///
/// This balances relevance with diversity -- suppressing near-duplicates.
pub fn mmr_select(
    candidates: &[ScoredCandidate],
    limit: usize,
    lambda: f64,
) -> Vec<ScoredCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut selected: Vec<ScoredCandidate> = Vec::with_capacity(limit);
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();

    // Normalize scores to [0, 1] for fair MMR comparison
    let max_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f64::INFINITY, f64::min);
    let score_range = (max_score - min_score).max(1e-10);

    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f64::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            let cand = &candidates[cand_idx];
            let norm_score = (cand.score - min_score) / score_range;

            // Compute max similarity to already-selected items
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|s| content_similarity(&cand.record.content, &s.record.content))
                    .fold(f64::NEG_INFINITY, f64::max)
            };

            let mmr = lambda * norm_score - (1.0 - lambda) * max_sim;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = pos;
            }
        }

        let chosen_idx = remaining.remove(best_idx);
        selected.push(candidates[chosen_idx].clone());
    }

    selected
}

/// Jaccard similarity between two content strings (word-level).
fn content_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();
    let words_b: HashSet<&str> = b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let union = words_a.union(&words_b).count() as f64;
    intersection / union
}

/// Run the full retrieval pipeline.
///
/// 1. RRF-fuse FTS + cognitive ranked lists
/// 2. Score with cognitive model blend
/// 3. MMR-select for diversity
/// 4. Filter noise
///
/// Accepts pre-fetched FTS results to avoid redundant queries.
/// The architecture supports adding vector search later by extending
/// the `fts_results` parameter to include vector candidates.
pub fn retrieve(
    fts_results: Vec<MemoryRecord>,
    limit: usize,
    cognitive_scores: &HashMap<String, f64>,
    config: &RetrievalConfig,
) -> Vec<ScoredCandidate> {
    let fts_ids: Vec<String> = fts_results.iter().map(|r| r.id.clone()).collect();

    // 1. RRF fusion (FTS + cognitive ranked lists)
    let mut ranked_lists: Vec<Vec<String>> = vec![fts_ids.clone()];
    if !cognitive_scores.is_empty() {
        let mut cog_ranked: Vec<_> = cognitive_scores.iter().collect();
        cog_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked_lists.push(cog_ranked.into_iter().map(|(id, _)| id.clone()).collect());
    }

    let rrf_ranked = rrf_fuse(&ranked_lists, RRF_K);

    // 2. Build scored candidates
    let mut candidate_map: HashMap<String, MemoryRecord> = HashMap::new();
    for r in fts_results {
        candidate_map.entry(r.id.clone()).or_insert(r);
    }

    let keyword_ranks: HashMap<&str, usize> = fts_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i + 1))
        .collect();

    let mut candidates: Vec<ScoredCandidate> = Vec::new();

    for (id, rrf_score) in &rrf_ranked {
        let record = match candidate_map.remove(id) {
            Some(r) => r,
            None => continue,
        };

        // Skip noise
        if is_noise(&record.content) || record.content.len() < config.min_content_length {
            continue;
        }

        let cog_score = cognitive_scores.get(id).copied().unwrap_or(0.0);

        // Normalize cognitive score via sigmoid
        let cog_norm = 1.0 / (1.0 + (-cog_score).exp());

        // Blend cognitive with RRF keyword score
        let first_rrf = rrf_ranked[0].1.max(1e-10);
        let final_score = 0.6 * cog_norm + 0.4 * rrf_score / first_rrf;

        candidates.push(ScoredCandidate {
            record,
            score: final_score,
            components: ScoreComponents {
                rrf_score: *rrf_score,
                cognitive_score: cog_score,
                semantic_score: 0.0,
                keyword_rank: keyword_ranks.get(id.as_str()).copied().unwrap_or(0),
                vector_rank: 0,
            },
        });
    }

    // Sort by score descending before MMR
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. MMR diversity selection
    mmr_select(&candidates, limit, config.mmr_lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let list1 = vec!["a".into(), "b".into(), "c".into()];
        let list2 = vec!["b".into(), "c".into(), "d".into()];
        let result = rrf_fuse(&[list1, list2], 60.0);

        // "b" appears at rank 2 in list1, rank 1 in list2 -> highest combined score
        assert_eq!(result[0].0, "b");
        // "c" appears in both -> second
        assert_eq!(result[1].0, "c");
    }

    #[test]
    fn test_content_similarity() {
        let sim = content_similarity("rust programming language", "rust programming tutorial");
        assert!(sim > 0.3); // "rust" and "programming" overlap

        let sim2 = content_similarity("hello world", "cooking recipes");
        assert!(sim2 < 0.1);
    }

    #[test]
    fn test_noise_detection() {
        assert!(is_noise("ok"));
        assert!(is_noise("thanks"));
        assert!(is_noise("hi"));
        assert!(is_noise(""));
        assert!(!is_noise("The cat sat on the mat"));
        assert!(!is_noise("Python uses dynamic typing"));
    }

    #[test]
    fn test_mmr_diversity() {
        use crate::types::{MemoryLayer, MemoryType};
        use chrono::Utc;

        let make = |id: &str, content: &str, score: f64| ScoredCandidate {
            record: MemoryRecord {
                id: id.to_string(),
                content: content.to_string(),
                memory_type: MemoryType::Factual,
                layer: MemoryLayer::Working,
                created_at: Utc::now(),
                access_times: vec![Utc::now()],
                working_strength: 1.0,
                core_strength: 0.0,
                importance: 0.5,
                pinned: false,
                consolidation_count: 0,
                last_consolidated: None,
                source: String::new(),
                contradicts: None,
                contradicted_by: None,
                metadata: None,
            },
            score,
            components: ScoreComponents::default(),
        };

        let candidates = vec![
            make(
                "1",
                "rust is a systems programming language for safety",
                0.9,
            ),
            make(
                "2",
                "rust is a systems programming language for speed",
                0.85,
            ), // near-dup (6/7 overlap)
            make("3", "cooking italian pasta recipes at home", 0.7), // diverse
        ];

        let selected = mmr_select(&candidates, 2, 0.5);
        assert_eq!(selected.len(), 2);
        // Should pick "1" (highest score) and "3" (most diverse), not "2" (duplicate of 1)
        assert_eq!(selected[0].record.id, "1");
        assert_eq!(selected[1].record.id, "3");
    }
}
