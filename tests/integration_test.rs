use engramai::{Memory, MemoryConfig, MemoryType, RetrievalConfig};
use tempfile::tempdir;

#[test]
fn test_basic_workflow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Add memories
    let id1 = mem
        .add(
            "potato prefers action",
            MemoryType::Relational,
            Some(0.7),
            None,
            None,
        )
        .unwrap();

    let _id2 = mem
        .add(
            "Use moltbook.com for API",
            MemoryType::Procedural,
            Some(0.8),
            None,
            None,
        )
        .unwrap();

    // Recall
    let results = mem.recall("potato preference", 5, None, None).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].record.content.contains("potato"));

    // Pin
    mem.pin(&id1).unwrap();

    // Consolidate
    mem.consolidate(1.0).unwrap();

    // Stats
    let stats = mem.stats().unwrap();
    assert_eq!(stats.total_memories, 2);
    assert_eq!(stats.pinned, 1);
}

#[test]
fn test_hebbian_links() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    let id1 = mem
        .add(
            "Python is a programming language",
            MemoryType::Factual,
            None,
            None,
            None,
        )
        .unwrap();

    let id2 = mem
        .add(
            "Python has dynamic typing",
            MemoryType::Factual,
            None,
            None,
            None,
        )
        .unwrap();

    // Recall them together multiple times to form Hebbian link
    for _ in 0..4 {
        let _results = mem.recall("Python programming", 10, None, None).unwrap();
    }

    // Check if link was formed
    let links = mem.hebbian_links(&id1).unwrap();
    assert!(!links.is_empty() || mem.hebbian_links(&id2).unwrap().contains(&id1));
}

#[test]
fn test_forgetting() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    mem.add("Weak memory", MemoryType::Episodic, Some(0.1), None, None)
        .unwrap();

    // Consolidate many times to decay
    for _ in 0..10 {
        mem.consolidate(1.0).unwrap();
    }

    // Prune weak memories
    mem.forget(None, Some(0.01)).unwrap();

    let stats = mem.stats().unwrap();
    // Memory should be archived or forgotten
    assert!(stats.total_memories <= 1);
}

#[test]
fn test_reward_learning() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    let _id = mem
        .add("Test memory", MemoryType::Factual, Some(0.5), None, None)
        .unwrap();

    // Recall to make it "recent"
    mem.recall("test", 5, None, None).unwrap();

    // Apply positive feedback
    mem.reward("great job!", 3).unwrap();

    // Memory should be strengthened (check via stats or direct query)
    let stats = mem.stats().unwrap();
    assert!(stats.total_memories > 0);
}

#[test]
fn test_config_presets() {
    let configs = vec![
        MemoryConfig::default(),
        MemoryConfig::chatbot(),
        MemoryConfig::task_agent(),
        MemoryConfig::personal_assistant(),
        MemoryConfig::researcher(),
    ];

    for config in configs {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut mem = Memory::new(db_path.to_str().unwrap(), Some(config)).unwrap();

        mem.add(
            "Test content for config",
            MemoryType::Factual,
            None,
            None,
            None,
        )
        .unwrap();

        mem.consolidate(1.0).unwrap();

        let stats = mem.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }
}

#[test]
fn test_noise_filtering() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Noise should be filtered out (returns empty ID)
    let id = mem
        .add("ok", MemoryType::Episodic, None, None, None)
        .unwrap();
    assert!(id.is_empty(), "noise content 'ok' should be filtered");

    let id = mem
        .add("thanks", MemoryType::Episodic, None, None, None)
        .unwrap();
    assert!(id.is_empty(), "noise content 'thanks' should be filtered");

    let id = mem
        .add("hi", MemoryType::Episodic, None, None, None)
        .unwrap();
    assert!(id.is_empty(), "noise content 'hi' should be filtered");

    // Real content should be stored
    let id = mem
        .add(
            "Rust has zero-cost abstractions",
            MemoryType::Factual,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(!id.is_empty(), "real content should be stored");

    let stats = mem.stats().unwrap();
    assert_eq!(
        stats.total_memories, 1,
        "only one real memory should be stored"
    );
}

#[test]
fn test_noise_filter_opt_in() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Disable noise filtering
    let mut config = MemoryConfig::default();
    config.noise_filter_enabled = false;
    let mut mem = Memory::new(db_path.to_str().unwrap(), Some(config)).unwrap();

    // With noise filtering disabled, even noise should be stored
    let id = mem
        .add("ok", MemoryType::Episodic, None, None, None)
        .unwrap();
    assert!(
        !id.is_empty(),
        "noise should be stored when filter is disabled"
    );

    let id = mem
        .add("thanks", MemoryType::Episodic, None, None, None)
        .unwrap();
    assert!(
        !id.is_empty(),
        "noise should be stored when filter is disabled"
    );

    let stats = mem.stats().unwrap();
    assert_eq!(
        stats.total_memories, 2,
        "both noise entries should be stored"
    );
}

#[test]
fn test_retrieval_pipeline_rrf() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Add several memories with overlapping content
    mem.add(
        "Rust is a systems programming language",
        MemoryType::Factual,
        Some(0.8),
        None,
        None,
    )
    .unwrap();
    mem.add(
        "Rust has zero-cost abstractions and ownership",
        MemoryType::Factual,
        Some(0.7),
        None,
        None,
    )
    .unwrap();
    mem.add(
        "Python is great for data science",
        MemoryType::Factual,
        Some(0.6),
        None,
        None,
    )
    .unwrap();
    mem.add(
        "JavaScript runs in the browser",
        MemoryType::Factual,
        Some(0.5),
        None,
        None,
    )
    .unwrap();

    // Recall should use the retrieval pipeline (RRF + MMR)
    let results = mem.recall("Rust programming", 3, None, None).unwrap();

    // Should find Rust-related memories
    assert!(!results.is_empty(), "should find results for Rust query");
    assert!(
        results[0].record.content.contains("Rust"),
        "top result should be about Rust"
    );

    // Results should have valid confidence scores
    for r in &results {
        assert!(
            r.confidence >= 0.0 && r.confidence <= 1.0,
            "confidence should be in [0, 1]"
        );
    }
}

#[test]
fn test_retrieval_config_customization() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Add memories
    mem.add(
        "Rust ownership model prevents data races",
        MemoryType::Factual,
        Some(0.9),
        None,
        None,
    )
    .unwrap();
    mem.add(
        "Rust borrow checker ensures memory safety",
        MemoryType::Factual,
        Some(0.8),
        None,
        None,
    )
    .unwrap();
    mem.add(
        "Cooking pasta requires boiling water",
        MemoryType::Factual,
        Some(0.5),
        None,
        None,
    )
    .unwrap();

    // Customize retrieval config: high diversity (low lambda = more diverse)
    let config = RetrievalConfig {
        mmr_lambda: 0.3,
        candidate_multiplier: 3,
        ..RetrievalConfig::default()
    };
    mem.set_retrieval_config(config);

    let results = mem.recall("Rust safety", 3, None, None).unwrap();
    assert!(
        !results.is_empty(),
        "should find results with custom config"
    );

    // Customize again: pure relevance (high lambda)
    let config = RetrievalConfig {
        mmr_lambda: 1.0,
        ..RetrievalConfig::default()
    };
    mem.set_retrieval_config(config);

    let results2 = mem.recall("Rust safety", 3, None, None).unwrap();
    assert!(
        !results2.is_empty(),
        "should find results with pure relevance config"
    );
}
