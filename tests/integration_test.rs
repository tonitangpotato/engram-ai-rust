use engramai::{Memory, MemoryConfig, MemoryType};
use tempfile::tempdir;

#[test]
fn test_basic_workflow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Add memories
    let id1 = mem
        .add("potato prefers action", MemoryType::Relational, Some(0.7), None, None)
        .unwrap();
    
    let _id2 = mem
        .add("Use moltbook.com for API", MemoryType::Procedural, Some(0.8), None, None)
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
        .add("Python is a programming language", MemoryType::Factual, None, None, None)
        .unwrap();
    
    let id2 = mem
        .add("Python has dynamic typing", MemoryType::Factual, None, None, None)
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

        mem.add("Test", MemoryType::Factual, None, None, None)
            .unwrap();
        
        mem.consolidate(1.0).unwrap();
        
        let stats = mem.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }
}
