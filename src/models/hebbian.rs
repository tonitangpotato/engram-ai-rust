//! Hebbian learning — co-activation forms memory links.
//!
//! "Neurons that fire together, wire together."
//!
//! When memories are recalled together repeatedly, they form Hebbian links.
//! These links create an associative network independent of explicit entity
//! tagging — purely emergent from usage patterns.

use crate::storage::Storage;

/// Record co-activation for a set of memory IDs.
///
/// When multiple memories are retrieved together (e.g., in a single recall),
/// each pair gets their coactivation_count incremented. When the count
/// reaches the threshold, a Hebbian link is automatically formed.
pub fn record_coactivation(
    storage: &mut Storage,
    memory_ids: &[String],
    threshold: i32,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    if memory_ids.len() < 2 {
        return Ok(vec![]);
    }

    let mut new_links = vec![];

    // Generate all pairs
    for i in 0..memory_ids.len() {
        for j in (i + 1)..memory_ids.len() {
            let id1 = &memory_ids[i];
            let id2 = &memory_ids[j];

            let formed = storage.record_coactivation(id1, id2, threshold)?;
            if formed {
                new_links.push((id1.clone(), id2.clone()));
            }
        }
    }

    Ok(new_links)
}
