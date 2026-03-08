# Engram

**Neuroscience-grounded memory system for AI agents** — Rust port of [Engram](https://github.com/tonitangpotato/engram-ai).

Engram implements cognitive science models (ACT-R, Memory Chain Model, Ebbinghaus forgetting, Hebbian learning) for AI agent memory, providing a scientifically-grounded alternative to naive vector similarity search. Works with any Rust-based agent framework (IronClaw, Rig, custom agents).

## Features

- **ACT-R Activation**: Retrieval based on frequency × recency (power law) + spreading activation
- **Memory Chain Model**: Dual-trace consolidation mimicking hippocampus → neocortex transfer
- **Ebbinghaus Forgetting**: Exponential decay with spaced repetition strengthening
- **Hebbian Learning**: Co-activation automatically forms associative links
- **STDP (Spike-Timing-Dependent Plasticity)**: Temporal patterns infer causal relationships
- **SQLite Storage**: Persistent storage with FTS5 full-text search
- **Zero External Dependencies**: No embeddings, no API calls — pure cognitive models

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
engramai = "0.1"
```

### Basic Usage

```rust
use ironclaw_engram::{Memory, MemoryType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory system
    let mut mem = Memory::new("./agent.db", None)?;

    // Store memories
    mem.add(
        "potato prefers action over discussion",
        MemoryType::Relational,
        Some(0.7),
        None,
        None,
    )?;

    mem.add(
        "Use www.moltbook.com not moltbook.com",
        MemoryType::Procedural,
        Some(0.8),
        None,
        None,
    )?;

    // Recall with ACT-R activation (not just cosine similarity!)
    let results = mem.recall("what does potato prefer?", 5, None, None)?;
    for r in results {
        println!("[{}] {}", r.confidence_label, r.record.content);
    }

    // Consolidate (run "sleep" cycle)
    mem.consolidate(1.0)?;

    // Reward learning
    mem.reward("good job!", 3)?;

    Ok(())
}
```

## Cognitive Models

### ACT-R Activation

```
A_i = B_i + Σ(W_j · S_ji) + importance_boost - contradiction_penalty
B_i = ln(Σ_k t_k^(-d))
```

Memories with high **frequency** (many accesses) and **recency** (recent accesses) have higher base-level activation. Context keywords provide spreading activation boost.

### Memory Chain Model

```
dr₁/dt = -μ₁ · r₁(t)                 (hippocampal trace, fast decay)
dr₂/dt = α · r₁(t) - μ₂ · r₂(t)      (neocortical trace, slow decay)
```

During consolidation ("sleep"), working memories transfer to core memories. This prevents catastrophic forgetting while allowing gradual knowledge integration.

### Ebbinghaus Forgetting

```
R(t) = e^(-t/S)
S = base_S × spacing_factor × importance_factor × consolidation_factor
```

Retrievability decays exponentially. **Spaced repetition** (repeated access at increasing intervals) dramatically increases stability.

### Hebbian Learning

When memories are recalled together ≥ N times (default 3), they form an associative link. This creates an emergent semantic network purely from usage patterns.

### STDP (Causal Inference)

Tracks temporal ordering of co-activated memories. If memory A consistently precedes B, infers potential causation A → B and creates a causal memory.

## Configuration Presets

Engram includes scientifically-tuned presets for common agent archetypes:

```rust
use ironclaw_engram::MemoryConfig;

// Chatbot: slow decay, high replay
let config = MemoryConfig::chatbot();
let mem = Memory::new("./chatbot.db", Some(config))?;

// Task agent: fast decay, focus on recent context
let config = MemoryConfig::task_agent();

// Personal assistant: very slow core decay, remember for months
let config = MemoryConfig::personal_assistant();

// Researcher: minimal forgetting, everything might be relevant
let config = MemoryConfig::researcher();
```

### Preset Comparison

| Parameter | Default | Chatbot | Task Agent | Personal Assistant | Researcher |
|-----------|---------|---------|------------|-------------------|------------|
| `mu1` (working decay) | 0.15 | 0.08 | 0.25 | 0.12 | 0.05 |
| `mu2` (core decay) | 0.005 | 0.003 | 0.01 | 0.001 | 0.001 |
| `alpha` (consolidation) | 0.08 | 0.12 | 0.05 | 0.10 | 0.15 |
| `interleave_ratio` | 0.3 | 0.4 | 0.1 | 0.3 | 0.5 |
| `forget_threshold` | 0.01 | 0.005 | 0.02 | 0.005 | 0.001 |

## API Reference

### Core Methods

- **`Memory::new(path, config)`** — Create or open database
- **`mem.add(content, type, importance, source, metadata)`** — Store a memory
- **`mem.recall(query, limit, context, min_confidence)`** — Retrieve with ACT-R
- **`mem.consolidate(days)`** — Run consolidation cycle ("sleep")
- **`mem.forget(memory_id, threshold)`** — Prune weak memories
- **`mem.reward(feedback, recent_n)`** — Apply dopaminergic feedback
- **`mem.downscale(factor)`** — Global synaptic downscaling
- **`mem.stats()`** — Memory system statistics
- **`mem.pin(memory_id)` / `mem.unpin(memory_id)`** — Pin/unpin memories
- **`mem.hebbian_links(memory_id)`** — Get associative neighbors

### Memory Types

- `Factual` — Facts and world knowledge
- `Episodic` — Events and experiences
- `Relational` — Knowledge about people/entities
- `Emotional` — Emotionally significant memories (high importance, slow decay)
- `Procedural` — How-to knowledge (slow decay)
- `Opinion` — Subjective beliefs (moderate decay)
- `Causal` — Cause-effect relationships (auto-created via STDP)

## Storage Schema

SQLite database with tables:

- `memories` — Core memory data with strengths, timestamps, metadata
- `access_log` — Every access timestamp (for ACT-R base-level activation)
- `hebbian_links` — Co-activation tracking and formed links
- `memories_fts` — FTS5 full-text search index

## IronClaw Integration

Engram can be used as a standalone crate or integrated with [IronClaw](https://github.com/nearai/ironclaw) (Rust AI agent framework).

### Standalone Usage

```rust
use ironclaw_engram::Memory;

let mut mem = Memory::new("./agent_memory.db", None)?;
// Use directly in your agent
```

### As IronClaw Dependency

Add to your IronClaw tool/skill:

```toml
[dependencies]
engramai = "0.1"
```

Then use in your agent's state:

```rust
struct AgentState {
    memory: ironclaw_engram::Memory,
    // ... other state
}
```

## Comparison with Python Engram

| Feature | Python Engram | Engram |
|---------|---------------|-----------|
| ACT-R activation | ✅ | ✅ |
| Memory Chain Model | ✅ | ✅ |
| Ebbinghaus forgetting | ✅ | ✅ |
| Hebbian learning | ✅ | ✅ |
| STDP causal inference | ✅ | ✅ |
| Vector embeddings | ✅ (optional) | ⏳ (planned) |
| Performance | ~10ms recall | ~1ms recall |
| Memory footprint | ~50MB | ~5MB |
| Deployment | Requires Python | Single binary |

## Why Cognitive Models?

Traditional AI memory systems use naive cosine similarity on embeddings. This ignores decades of neuroscience research on how human memory actually works:

- **Frequency matters** — memories accessed often are more retrievable
- **Recency matters** — recent memories are more accessible (but decay over time)
- **Spaced repetition** — repeated access at increasing intervals strengthens memories
- **Consolidation** — memories transfer from short-term (hippocampus) to long-term (neocortex)
- **Hebbian associations** — co-activated memories become linked
- **Importance modulation** — emotional significance affects memory strength

Engram implements these principles mathematically, providing memory dynamics that mirror human cognition.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Citation

If you use Engram in research, please cite:

```bibtex
@software{ironclaw_engram,
  author = {Tang, Toni},
  title = {Engram: Neuroscience-Grounded Memory for AI Agents},
  year = {2026},
  url = {https://github.com/tonitangpotato/engram}
}
```

Original Python Engram:

```bibtex
@software{engram_ai,
  author = {Tang, Toni},
  title = {Engram: Neuroscience-Grounded Memory for AI Agents},
  year = {2026},
  url = {https://github.com/tonitangpotato/engram-ai}
}
```

## Acknowledgments

Cognitive models based on:

- Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press. (ACT-R)
- Murre, J. M., & Chessa, A. G. (2011). Power laws from individual differences in learning and forgetting. (Memory Chain Model)
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology.*
- Hebb, D. O. (1949). *The Organization of Behavior.*
