# Engram AI 🧠 (Rust)

**Neuroscience-grounded memory system for AI agents** — pure Rust, zero external dependencies.

[![Crates.io](https://img.shields.io/crates/v/engramai)](https://crates.io/crates/engramai)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

> Give your AI agent a brain that actually remembers, associates, forgets, and *evolves* like a human.

### 🐍 Also available in Python: [`engramai` on PyPI](https://pypi.org/project/engramai/) — [GitHub](https://github.com/tonitangpotato/engram-ai)
> Includes semantic search (50+ languages), MCP server, multiple embedding providers.

---

## Why Engram?

Traditional AI memory = vector database + cosine similarity. That ignores decades of neuroscience.

**Engram implements how human memory actually works:**

| Principle | What it does | Traditional approach |
|-----------|-------------|---------------------|
| **ACT-R Activation** | Frequently-used, recent, important memories rank higher | All memories equal |
| **Hebbian Learning** | Co-accessed memories auto-link (even across agents) | No associations |
| **Ebbinghaus Forgetting** | Unused memories decay naturally | Never forgets |
| **Consolidation** | Working → long-term transfer (like sleep) | No memory tiers |
| **Dopaminergic Reward** | Successful actions strengthen memories | No feedback |
| **Emotional Bus** *(v0.2)* | Memory ↔ personality ↔ behavior closed loop | Static config files |
| **Multi-Agent Shared Memory** *(v0.2)* | Namespaced memory with ACL for agent swarms | Context explosion on handoff |

## What's New in v0.2

### 🔄 Emotional Bus — Memory Shapes Personality

Engram v2 isn't just a memory store — it's the **nervous system** connecting all agent modules.

```
Memory shapes personality.    (Engram → SOUL.md)
Personality shapes behavior.  (SOUL.md → HEARTBEAT.md)
Behavior creates new memory.  (HEARTBEAT.md → Engram)
The loop IS the self.
```

- **Emotional accumulation** — track valence trends per domain over time
- **Drive alignment** — memories related to core drives get automatic importance boost
- **Behavior feedback** — actions that yield nothing auto-deprioritize; successful patterns reinforce
- **SOUL/HEARTBEAT/IDENTITY auto-updates** — the agent's personality evolves through experience

### 🤝 Multi-Agent Shared Memory

Replace context explosion with cognitive shared memory for agent swarms:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Agent A  │    │ Agent B  │    │ Agent C  │
│  10K ctx │    │  10K ctx │    │  10K ctx │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
┌─────────────────────────────────────────────┐
│           Shared Engram DB                  │
│  Namespaced: agentA.* │ agentB.* │ global.*│
│  Cross-namespace Hebbian links auto-form    │
│  ACL: CEO controls who reads/writes what    │
└─────────────────────────────────────────────┘
```

- **Namespace isolation** — each agent writes to its own namespace
- **Cross-namespace Hebbian links** — co-occurring concepts auto-connect across agents
- **ACL (Access Control)** — CEO agent controls cross-agent memory access
- **Subscription & notifications** — agents subscribe to namespaces and get notified of high-importance writes

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
engramai = "0.2"
```

### Basic Usage

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

### Emotional Bus (v0.2)

```rust
use engramai::{Memory, MemoryType};
use engramai::bus::EmotionalBus;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with emotional bus connected to workspace files
    let mut mem = Memory::with_emotional_bus(
        "./agent.db",
        None,
        "./workspace",  // directory containing SOUL.md, HEARTBEAT.md, IDENTITY.md
    )?;

    // Store with emotional tagging — bus auto-boosts importance based on SOUL drives
    mem.add_with_emotion(
        "Closed a $10K deal today",
        MemoryType::Episodic,
        Some(0.8),
        "business",    // domain
        0.9,           // positive valence
    )?;

    // Bus tracks emotional trends over time
    let bus = mem.emotional_bus().unwrap();
    let conn = mem.connection();
    let trends = bus.get_trends(conn)?;
    for trend in &trends {
        println!("{}: valence={:.2}, count={}", trend.domain, trend.avg_valence, trend.count);
    }

    // When enough negative experience accumulates → suggest SOUL updates
    let soul_updates = bus.suggest_soul_updates(conn)?;
    for update in &soul_updates {
        println!("Suggest: {} → {}", update.key, update.value);
    }

    // Log behavior outcomes → auto-adjust HEARTBEAT priorities
    bus.log_behavior(conn, "check_email", true)?;   // useful check
    bus.log_behavior(conn, "check_twitter", false)?; // wasted check

    let heartbeat_updates = bus.suggest_heartbeat_updates(conn)?;
    for update in &heartbeat_updates {
        println!("{}: {} (score: {:.2})", update.action,
            if update.boost { "boost" } else { "deprioritize" }, update.score);
    }

    Ok(())
}
```

### Multi-Agent Shared Memory (v0.2)

```rust
use engramai::{Memory, MemoryType, Permission};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mem = Memory::new("./shared.db", None)?;

    // Each agent writes to its own namespace
    mem.set_agent_id("trading-agent");
    mem.add_to_namespace("Oil broke $91 resistance", "trading",
        MemoryType::Factual, Some(0.8), None, None)?;

    mem.set_agent_id("research-agent");
    mem.add_to_namespace("Found XLP→USO lead-lag pair", "research",
        MemoryType::Causal, Some(0.9), None, None)?;

    // CEO queries across all namespaces
    mem.set_agent_id("ceo");
    let results = mem.recall_from_namespace("oil trading signals", "*", 5, None, None)?;
    for r in results {
        println!("[{:.3}] [{}] {}", r.activation, r.namespace, r.record.content);
    }

    // ACL: CEO controls access
    mem.grant("trading-agent", "research", Permission::Read, "ceo")?;
    mem.revoke("research-agent", "trading")?;

    // Check permissions before access
    assert!(mem.check_permission("trading-agent", "research", "read")?);
    assert!(!mem.check_permission("research-agent", "trading", "read")?);

    // Cross-namespace Hebbian discovery
    let cross_links = mem.discover_cross_links(10)?;
    for link in &cross_links {
        println!("Cross-link: {} ↔ {} (strength: {:.2})", link.0, link.1, link.2);
    }

    // Subscriptions: get notified of high-importance writes
    mem.subscribe("ceo", "trading", Some(0.7))?; // notify when importance > 0.7
    let notifications = mem.check_notifications("ceo")?;

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
Co-accessed memories form associative links → spreading activation boosts related memories on recall. In v0.2, Hebbian links form **across namespaces** — enabling cross-agent pattern discovery.

## Emotional Bus Architecture

```
┌─────────────────────────────────────────────────┐
│              Emotional Bus (Engram v0.2)         │
│                                                  │
│  Engram emotions → trigger SOUL updates          │
│  SOUL drives     → influence Engram importance   │
│  Engram feedback → adjust HEARTBEAT priorities   │
│  HEARTBEAT outcomes → flow back to Engram        │
│  Everything      → reshapes IDENTITY over time   │
│                                                  │
│  Memory shapes personality.                      │
│  Personality shapes behavior.                    │
│  Behavior creates new memory.                    │
│  The loop IS the self.                           │
└─────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| **EmotionalAccumulator** | Tracks valence trends per domain, detects when SOUL updates needed |
| **DriveAlignment** | Scores memory importance based on SOUL.md drives |
| **BehaviorFeedback** | Logs action outcomes, suggests HEARTBEAT priority changes |
| **ModIO** | Reads/writes SOUL.md, HEARTBEAT.md, IDENTITY.md programmatically |
| **Subscriptions** | Cross-agent notifications for high-importance memories |

## Configuration Presets

```rust
use engramai::MemoryConfig;

let config = MemoryConfig::chatbot();            // Slow decay, high replay
let config = MemoryConfig::task_agent();          // Fast decay, recent context
let config = MemoryConfig::personal_assistant();  // Very slow decay, months of memory
let config = MemoryConfig::researcher();          // Minimal forgetting
```

## API Reference

### Core Memory

| Method | Description |
|--------|-------------|
| `Memory::new(path, config)` | Create or open database |
| `Memory::with_emotional_bus(path, config, workspace)` | Create with emotional bus |
| `mem.add(content, type, importance, source, metadata)` | Store a memory |
| `mem.add_to_namespace(content, ns, type, importance, source, metadata)` | Store to specific namespace |
| `mem.add_with_emotion(content, type, importance, domain, valence)` | Store with emotional tagging |
| `mem.recall(query, limit, context, min_confidence)` | Retrieve with ACT-R ranking |
| `mem.recall_from_namespace(query, ns, limit, context, min_confidence)` | Retrieve from namespace (`"*"` for all) |
| `mem.recall_with_associations(query, ns, limit, context, min_confidence, depth)` | Retrieve with Hebbian spreading activation |
| `mem.consolidate(days)` | Run consolidation cycle |
| `mem.consolidate_namespace(ns, days)` | Consolidate specific namespace |
| `mem.forget(memory_id, threshold)` | Prune weak memories |
| `mem.reward(feedback, recent_n)` | Dopaminergic feedback |
| `mem.downscale(factor)` | Global synaptic downscaling |
| `mem.stats()` / `mem.stats_ns(ns)` | Memory system statistics |
| `mem.pin(id)` / `mem.unpin(id)` | Pin/unpin memories |
| `mem.hebbian_links(id)` / `mem.hebbian_links_ns(id, cross_ns)` | Get associative neighbors |

### Multi-Agent ACL

| Method | Description |
|--------|-------------|
| `mem.set_agent_id(id)` | Set current agent identity |
| `mem.grant(agent_id, namespace, permission, granted_by)` | Grant access |
| `mem.revoke(agent_id, namespace)` | Revoke access |
| `mem.check_permission(agent_id, namespace, action)` | Check access |
| `mem.list_permissions(agent_id)` | List all permissions |

### Cross-Agent Intelligence

| Method | Description |
|--------|-------------|
| `mem.discover_cross_links(limit)` | Find Hebbian associations across namespaces |
| `mem.get_cross_associations(memory_id)` | Get cross-namespace neighbors |
| `mem.subscribe(agent_id, namespace, min_importance)` | Subscribe to namespace notifications |
| `mem.unsubscribe(agent_id, namespace)` | Unsubscribe |
| `mem.list_subscriptions(agent_id)` | List subscriptions |
| `mem.check_notifications(agent_id)` | Get new notifications (advances cursor) |
| `mem.peek_notifications(agent_id, limit)` | Peek without advancing cursor |

### Emotional Bus

| Method | Description |
|--------|-------------|
| `bus.process_interaction(conn, content, domain, valence)` | Process interaction through full bus pipeline |
| `bus.align_importance(content)` | Get importance boost from SOUL drive alignment |
| `bus.log_behavior(conn, action, positive)` | Log behavior outcome |
| `bus.get_trends(conn)` | Get emotional trends per domain |
| `bus.get_behavior_stats(conn)` | Get action success/failure stats |
| `bus.suggest_soul_updates(conn)` | Get suggested SOUL.md changes |
| `bus.suggest_heartbeat_updates(conn)` | Get suggested HEARTBEAT priority changes |
| `bus.update_soul(key, value)` | Write update to SOUL.md |
| `bus.add_heartbeat_task(description)` | Add task to HEARTBEAT.md |

## IronClaw Integration

Works as a cognitive memory layer alongside [IronClaw](https://github.com/nearai/ironclaw)'s FTS+pgvector workspace memory.

```rust
// IronClaw's workspace: FTS + pgvector → document search
// + engramai:           ACT-R + Hebbian → cognitive memory
//
// They complement each other:
// - Workspace memory for searching docs, notes, code
// - Engram for agent personality, preferences, learned patterns
```

**Issue**: [nearai/ironclaw#739](https://github.com/nearai/ironclaw/issues/739)

## Why Engram Beats Swarm-Style Handoffs

| | Swarm (handoff) | Engram Shared Memory |
|---|---|---|
| Context growth | O(n × conversation_length) | O(query_limit) — constant |
| Agent isolation | ❌ Full history shared | ✅ Namespaced with ACL |
| Cross-domain insights | ❌ Only within handoff chain | ✅ Hebbian cross-links |
| Async | ❌ Synchronous handoff | ✅ Write anytime, read anytime |
| Persistence | ❌ Lost between runs | ✅ SQLite, permanent |
| Personality | ❌ Static prompts | ✅ Evolves through emotional bus |

## Python vs Rust

| Feature | Python (`pip install engramai`) | Rust (`cargo add engramai`) |
|---------|------|------|
| ACT-R activation | ✅ | ✅ |
| Hebbian learning | ✅ | ✅ |
| Ebbinghaus forgetting | ✅ | ✅ |
| Consolidation | ✅ | ✅ |
| STDP causal inference | ✅ | ✅ |
| Emotional Bus | ❌ | ✅ *v0.2* |
| Multi-Agent / Namespace | ❌ | ✅ *v0.2* |
| ACL | ❌ | ✅ *v0.2* |
| Cross-Agent Subscriptions | ❌ | ✅ *v0.2* |
| Vector embeddings | ✅ (50+ languages) | ⏳ planned |
| MCP server | ✅ | ⏳ planned |
| Recall latency | ~10ms | **~1-5ms** |
| Memory footprint | ~50MB | **~5MB** |
| Deployment | Requires Python | **Single binary** |

## Roadmap

- [x] **v0.1**: Core cognitive models (ACT-R, Hebbian, Ebbinghaus, Consolidation)
- [x] **v0.1**: Dopaminergic reward, synaptic downscaling, pinning
- [x] **v0.2**: Namespace isolation for multi-agent shared memory
- [x] **v0.2**: ACL — CEO agent controls cross-agent memory access
- [x] **v0.2**: Emotional Bus — memory ↔ SOUL ↔ HEARTBEAT closed loop
- [x] **v0.2**: Cross-namespace Hebbian link discovery
- [x] **v0.2**: Subscription & notification system
- [ ] **v0.3**: CLI binary (`engram store`, `engram recall`, `engram grant`)
- [ ] **v0.3**: Vector embeddings (optional, for semantic search)
- [ ] **v0.3**: Voice I/O emotion detection (speech rate, energy, pause analysis)
- [ ] **v0.4**: MCP server (Rust-native)

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Commercial licensing available, see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

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
- **Emotional Bus Architecture** — Inspired by conversation between potato and Clawd (2026-03-07).
