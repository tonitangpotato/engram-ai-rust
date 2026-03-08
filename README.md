# Engram AI 🧠 (Rust)

**Neuroscience-grounded memory system for AI agents** — pure Rust, zero external dependencies.

[![Crates.io](https://img.shields.io/crates/v/engramai)](https://crates.io/crates/engramai)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

> Give your AI agent a brain that actually remembers, associates, and forgets like a human.

### 🐍 Also available in Python: [`engramai` on PyPI](https://pypi.org/project/engramai/) — [GitHub](https://github.com/tonitangpotato/engram-ai)
> Includes semantic search (50+ languages), MCP server, multiple embedding providers.

---

## Why Engram?

Traditional AI memory = vector database + cosine similarity. That ignores decades of neuroscience.

**Engram implements how human memory actually works:**

| Principle | What it does | Traditional approach |
|-----------|-------------|---------------------|
| **ACT-R Activation** | Frequently-used, recent, important memories rank higher | All memories equal |
| **Hebbian Learning** | Co-accessed memories auto-link | No associations |
| **Ebbinghaus Forgetting** | Unused memories decay naturally | Never forgets |
| **Consolidation** | Working → long-term transfer (like sleep) | No memory tiers |
| **Dopaminergic Reward** | Successful actions strengthen memories | No feedback |

## Performance

| Operation | 500 memories |
|-----------|-------------|
| Store | 69ms (~0.14ms each) |
| Recall | 5ms |
| Consolidate | 60ms |
| Binary size | ~5MB |
| Memory footprint | ~5MB |

## Quick Start

```toml
[dependencies]
engramai = "0.1"
```

```rust
use engramai::{Memory, MemoryType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mem = Memory::new("./agent.db", None)?;

    // Store with cognitive metadata
    mem.add("User prefers Rust over Python", MemoryType::Factual, Some(0.9), None, None)?;
    mem.add("Discussed WASM security model", MemoryType::Episodic, Some(0.6), None, None)?;
    mem.add("cargo build --release for prod", MemoryType::Procedural, Some(0.7), None, None)?;

    // Recall with ACT-R activation (not just keyword match!)
    let results = mem.recall("user language preference", 5, None, None)?;
    for r in results {
        println!("[{:.3}] {}", r.activation, r.record.content);
    }

    // Consolidate (periodic "sleep" cycle)
    mem.consolidate(7.0)?;

    // Reward learning (dopaminergic feedback)
    mem.reward("positive", 3)?;

    Ok(())
}
```

## Memory Types

| Type | Use case | Default importance |
|------|----------|-------------------|
| `Factual` | Facts and knowledge | 0.3 |
| `Episodic` | Events and experiences | 0.4 |
| `Relational` | Knowledge about people/entities | 0.6 |
| `Emotional` | Emotionally significant (slow decay) | 0.9 |
| `Procedural` | How-to knowledge (slow decay) | 0.5 |
| `Opinion` | Subjective beliefs | 0.3 |
| `Causal` | Cause-effect relationships | 0.7 |

## Cognitive Models

### ACT-R Activation
```
A_i = B_i + Σ(W_j · S_ji) + importance_boost
B_i = ln(Σ_k t_k^(-d))    ← frequency × recency (power law)
```

### Memory Chain (Consolidation)
```
dr₁/dt = -μ₁ · r₁(t)                 ← working memory (fast decay)
dr₂/dt = α · r₁(t) - μ₂ · r₂(t)      ← core memory (slow decay)
```

### Ebbinghaus Forgetting
```
R(t) = e^(-t/S)
S = base_S × spacing_factor × importance_factor
```

### Hebbian Learning
Co-accessed memories form associative links → spreading activation boosts related memories on recall.

## Configuration Presets

```rust
use engramai::MemoryConfig;

let config = MemoryConfig::chatbot();            // Slow decay, high replay
let config = MemoryConfig::task_agent();          // Fast decay, recent context
let config = MemoryConfig::personal_assistant();  // Very slow decay, months of memory
let config = MemoryConfig::researcher();          // Minimal forgetting
```

## IronClaw Integration

Works as a cognitive memory layer alongside [IronClaw](https://github.com/nearai/ironclaw)'s FTS+pgvector workspace memory.

See [examples/ironclaw_integration.rs](examples/ironclaw_integration.rs) for a complete example.

```rust
// IronClaw's workspace: FTS + pgvector → document search
// + engramai:           ACT-R + Hebbian → cognitive memory
//
// They complement each other:
// - Workspace memory for searching docs, notes, code
// - Engram for agent personality, preferences, learned patterns
```

**Issue**: [nearai/ironclaw#739](https://github.com/nearai/ironclaw/issues/739)

## API Reference

| Method | Description |
|--------|-------------|
| `Memory::new(path, config)` | Create or open database |
| `mem.add(content, type, importance, source, metadata)` | Store a memory |
| `mem.recall(query, limit, context, min_confidence)` | Retrieve with ACT-R ranking |
| `mem.consolidate(days)` | Run consolidation cycle |
| `mem.forget(memory_id, threshold)` | Prune weak memories |
| `mem.reward(feedback, recent_n)` | Dopaminergic feedback |
| `mem.downscale(factor)` | Global synaptic downscaling |
| `mem.stats()` | Memory system statistics |
| `mem.pin(id)` / `mem.unpin(id)` | Pin/unpin memories |
| `mem.hebbian_links(id)` | Get associative neighbors |

## Python vs Rust

| Feature | Python (`pip install engramai`) | Rust (`cargo add engramai`) |
|---------|------|------|
| ACT-R activation | ✅ | ✅ |
| Hebbian learning | ✅ | ✅ |
| Ebbinghaus forgetting | ✅ | ✅ |
| Consolidation | ✅ | ✅ |
| STDP causal inference | ✅ | ✅ |
| Vector embeddings | ✅ (50+ languages) | ⏳ planned |
| MCP server | ✅ | ⏳ planned |
| Recall latency | ~10ms | **~1-5ms** |
| Memory footprint | ~50MB | **~5MB** |
| Deployment | Requires Python | **Single binary** |

## Roadmap

- [ ] **v0.2**: Namespace isolation for multi-agent shared memory
- [ ] **v0.2**: ACL — CEO agent controls cross-agent memory access
- [ ] **v0.2**: CLI binary (`engram store`, `engram recall`)
- [ ] **v0.3**: Emotional Bus — memory ↔ SOUL ↔ HEARTBEAT closed loop
- [ ] **v0.3**: Vector embeddings (optional, for semantic search)

## License

MIT OR Apache-2.0

## Citation

```bibtex
@software{engramai,
  author = {Tang, Toni},
  title = {Engram AI: Neuroscience-Grounded Memory for AI Agents},
  year = {2026},
  url = {https://github.com/tonitangpotato/engram-ai-rust}
}
```

## Acknowledgments

- **ACT-R** — Anderson, J. R. (2007). Carnegie Mellon University.
- **Memory Chain Model** — Murre, J. M., & Chessa, A. G. (2011).
- **Forgetting Curve** — Ebbinghaus, H. (1885).
- **Hebbian Learning** — Hebb, D. O. (1949).
