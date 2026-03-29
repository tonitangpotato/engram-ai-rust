# engramai

Neuroscience-grounded memory for AI agents. ACT-R activation, Hebbian learning, 
Ebbinghaus forgetting, cognitive consolidation, vector embeddings, and LLM extraction.

[![Crates.io](https://img.shields.io/crates/v/engramai)](https://crates.io/crates/engramai)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

## Features

| Feature | Description |
|---------|-------------|
| ACT-R Activation | Retrieval based on frequency, recency, spreading activation |
| Ollama Embeddings | Semantic similarity via local embeddings (nomic-embed-text) |
| LLM Extraction | Extract key facts using Claude Haiku or local models |
| EmotionBus | Emotional valence tracking, drive alignment (multilingual), behavior feedback |
| Session Working Memory | Miller's Law 7±2, topic continuity detection |
| Confidence Calibration | Two-dimensional metacognitive monitoring |
| Hybrid Search | Adaptive vector + FTS with configurable weights (15% FTS + 60% embedding + 25% ACT-R) |
| CJK Tokenization | jieba-based Chinese/Japanese word segmentation for precise FTS matching |
| Multi-Agent | Namespaces, ACL permissions, subscriptions |
| Hebbian Learning | Associative links from co-activation |
| Ebbinghaus Forgetting | Exponential decay with spaced repetition |

## Quick Start

### Installation

```toml
[dependencies]
engramai = "0.2.1"
```

### Basic Usage

```rust
use engramai::{Memory, MemoryType};

let mut mem = Memory::new("./agent.db", None)?;

// Store a memory
mem.add("potato prefers Rust over Python", MemoryType::Relational, Some(0.7), None, None)?;

// Recall with semantic search + ACT-R
let results = mem.recall("what language does potato prefer?", 5, None, None)?;
```

### With LLM Extraction (Recommended)

Instead of storing raw text, extract key facts using an LLM:

```rust
use engramai::{Memory, AnthropicExtractor};

let mut mem = Memory::new("./agent.db", None)?;

// Option 1: Auto-detect from environment
// Just set ANTHROPIC_API_KEY=sk-ant-... and it works automatically

// Option 2: Explicit setup
mem.set_extractor(Box::new(AnthropicExtractor::new("sk-ant-...", false)));

// Now add() automatically extracts facts
mem.add("I had pizza yesterday and it was great, but my girlfriend didn't like it", 
    MemoryType::Episodic, None, None, None)?;
// Stores: "user likes pizza", "user's girlfriend doesn't like pizza" (separate entries)
```

### With Embeddings

```rust
use engramai::{Memory, MemoryConfig, EmbeddingConfig};

let config = MemoryConfig {
    embedding: EmbeddingConfig {
        provider: "ollama".into(),
        model: "nomic-embed-text".into(),
        ..Default::default()
    },
    ..Default::default()
};

let mut mem = Memory::new("./agent.db", Some(config))?;
// Recall now uses semantic similarity + ACT-R activation
```

## Configuration

Engram supports layered configuration with clear priority:

### Auth Priority (high → low)

1. **Code-level** `set_extractor()` — for agent harnesses (RustClaw, etc.)
2. **Environment variable** `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY`
3. **Config file** provider setting + env var for auth
4. **No extractor** — stores raw text (backward compatible)

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Haiku extraction |
| `ANTHROPIC_AUTH_TOKEN` | OAuth token (Claude Max plan) |
| `ENGRAM_EXTRACTOR_MODEL` | Override extractor model (default: claude-haiku-4-5-20251001) |
| `ENGRAM_DB` | Default database path for CLI |
| `ENGRAM_EMBEDDING_MODEL` | Override embedding model |
| `ENGRAM_EMBEDDING_HOST` | Override Ollama host |

### Config File

Location varies by platform:
- **Linux**: `~/.config/engram/config.json`
- **macOS**: `~/Library/Application Support/engram/config.json`
- **Windows**: `%APPDATA%\engram\config.json`

```json
{
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "host": "http://localhost:11434"
  },
  "extractor": {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001"
  }
}
```

> ⚠️ **Security**: Never store API keys in the config file. Use environment variables or pass tokens in code.

Create config interactively:
```bash
engram init
```

### Search Weights (Hybrid Recall)

Control how `recall()` balances FTS exact matching, semantic similarity, and temporal activation:

```rust
use engramai::MemoryConfig;

let mut config = MemoryConfig::default();
config.fts_weight = 0.15;        // 15% exact term matching (via FTS)
config.embedding_weight = 0.60;   // 60% semantic similarity (via embeddings)
config.actr_weight = 0.25;        // 25% recency/frequency (via ACT-R)

let mut mem = Memory::new("./agent.db", Some(config))?;
```

**Recommended presets:**

| Use case | FTS | Embedding | ACT-R | Description |
|----------|-----|-----------|-------|-------------|
| **Default** (balanced) | 0.15 | 0.60 | 0.25 | Good for most agents |
| **Keyword-focused** | 0.30 | 0.50 | 0.20 | Project names, exact terms matter |
| **Semantic-heavy** | 0.10 | 0.70 | 0.20 | Concept search, less exact matching |
| **Recency-biased** | 0.10 | 0.50 | 0.40 | Recent events most important |

> 💡 **Tip**: Weights should sum to ~1.0 for intuitive percentage interpretation.

### CJK Tokenization

For Chinese/Japanese/Korean text, Engram uses **jieba** for intelligent word segmentation:

```
❌ Without jieba:
   "engram是认知记忆系统" → "engram" "是" "认" "知" "记" "忆" "系" "统"
   Search "记忆系统" matches poorly (3 separate characters)

✅ With jieba:
   "engram是认知记忆系统" → "engram" "是" "认知" "记忆系统"
   Search "记忆系统" matches precisely (as a complete term)
```

**Performance:**
- First jieba load: ~220ms (once per process)
- Per-memory tokenization: <0.02ms (negligible overhead)
- Applies to FTS path only (15% of recall weight)

**No configuration needed** — jieba auto-activates for CJK text.

## Architecture

### Memory Pipeline

```
User message
    ↓
[Extractor] → LLM extracts key facts (Claude Haiku, optional)
    ↓
[Embedding] → Generate 768-dim vector (Ollama nomic-embed-text, local)
    ↓
[Drive Alignment] → Cosine similarity with pre-embedded SOUL drives (if EmotionBus active)
    ↓                 → Importance boost for drive-aligned memories
[Storage] → SQLite (text + FTS5 + vector BLOB + CJK tokenization)
    ↓

Query
    ↓
[Embedding] → Generate query vector (same model)
    ↓
[Hybrid Search] → 15% FTS + 60% embedding cosine + 25% ACT-R activation
    ↓
[Confidence] → Two-dimensional scoring (reliability × salience)
    ↓
Results with confidence labels
```

### Embedding Model

Engram uses **nomic-embed-text** via local Ollama by default:

| Property | Value |
|----------|-------|
| Model | `nomic-embed-text` (Nomic AI, open source) |
| Dimensions | 768 |
| Runtime | Local Ollama (`localhost:11434`) |
| Cost | Free (no API calls) |
| Multilingual | Yes (sufficient for CJK ↔ Latin alignment) |
| Latency | ~5ms per embedding |

Configurable via `EmbeddingConfig` — swap to OpenAI `text-embedding-3-large` or any Ollama model without code changes.

### Cognitive Models

- **ACT-R**: Base-level activation (frequency × recency power law) + spreading activation from context
- **Ebbinghaus**: Exponential forgetting curve, counteracted by consolidation
- **Hebbian**: "Neurons that fire together wire together" — co-activated memories form links
- **STDP**: Temporal ordering creates causal links during consolidation
- **Miller's Law**: Working memory limited to 7±2 chunks (SessionWorkingMemory)

## CLI

```bash
# Install
cargo install engramai

# Initialize config
engram init

# Store memories (with extraction)
engram store "had a great meeting with John about the Q4 roadmap"
engram store --extractor anthropic "..."

# Recall
engram recall "what happened in meetings?"
engram recall-causal "what caused the outage?"

# Manage
engram stats
engram consolidate
engram reindex
engram export ./backup.json
```

## For Agent Developers

### Integration with Agent Harness

```rust
use engramai::{Memory, MemoryConfig, AnthropicExtractor};

// In your agent's init:
let mut mem = Memory::new("./agent-memory.db", Some(config))?;

// Set up extraction using your agent's existing LLM auth
let token = your_agent.get_oauth_token()?;
mem.set_extractor(Box::new(AnthropicExtractor::new(&token, true)));

// Before LLM call: auto-recall relevant context
let memories = mem.recall(&user_message, 5, None, Some(0.3))?;
let context = memories.iter()
    .map(|m| format!("- {}", m.record.content))
    .collect::<Vec<_>>()
    .join("\n");

// After LLM response: auto-store important facts
mem.add(&format!("{} → {}", user_message, response), 
    MemoryType::Episodic, None, None, None)?;
// Extractor automatically extracts facts from the conversation
```

### Session Working Memory

```rust
use engramai::SessionWorkingMemory;

let mut wm = SessionWorkingMemory::new(7, 300.0); // 7 items, 5min decay

// Smart recall: only full search on topic change
let result = mem.session_recall(&message, &mut wm, 5, None, None)?;
if result.full_recall {
    println!("Topic changed, did full recall");
} else {
    println!("Continuous topic, used working memory cache");
}
```

### Multi-Agent Shared Memory

```rust
use engramai::{Memory, MemoryType, Permission};

let mut mem = Memory::new("./shared.db", None)?;

// Each agent writes to its own namespace
mem.set_agent_id("trading-agent");
mem.add_to_namespace("Oil broke $91 resistance", MemoryType::Factual, 
    Some(0.8), None, None, Some("trading"))?;

// CEO queries across all namespaces
mem.set_agent_id("ceo");
let results = mem.recall_from_namespace("oil trading signals", 5, None, None, Some("*"))?;

// ACL: CEO controls access
mem.grant("trading-agent", "research", Permission::Read)?;
```

### Emotional Bus

```rust
use engramai::Memory;

// Initialize with emotional bus connected to workspace files
let mut mem = Memory::with_emotional_bus(
    "./agent.db",
    "./workspace",  // directory with SOUL.md, HEARTBEAT.md
    None,
)?;

// Store with emotional tagging
mem.add_with_emotion(
    "Closed a $10K deal today",
    MemoryType::Episodic,
    Some(0.8),
    Some("business"),
    None,
    None,
    0.9,           // positive valence
    "business",    // domain
)?;

// Get emotional trends
if let Some(bus) = mem.emotional_bus() {
    let trends = bus.get_trends(mem.connection())?;
    for trend in &trends {
        println!("{}: valence={:.2}", trend.domain, trend.valence);
    }
}
```

### Cross-Language Drive Alignment

Engram's EmotionalBus automatically handles multilingual SOUL drives using **embedding-based alignment**:

```
Problem: SOUL.md in Chinese, agent receives English messages
  SOUL drive: "帮potato实现财务自由，找到市场机会"
  Message: "trading profit market opportunity"
  Keyword matching: score = 0.0 ❌ (字面不匹配)

Solution: Hybrid alignment (keyword + embedding)
  Keyword score: 0.0 (cross-language, can't match)
  Embedding score: 0.14 (semantic similarity via nomic-embed-text)
  Final score: max(0.0, 0.14) = 0.14 ✅
```

Drive descriptions are pre-embedded at startup. At store time, content embeddings (already computed for recall) are reused for alignment — **zero additional embedding cost**.

```rust
use engramai::Memory;

let mut mem = Memory::with_emotional_bus("./agent.db", "./workspace", None)?;

// If embedding provider is available, enable cross-language alignment
if let (Some(bus), Some(provider)) = (mem.emotional_bus_mut(), mem.embedding_provider()) {
    bus.init_embeddings(provider);
    // Now Chinese drives align with English content (and vice versa)
}

// Importance boost works regardless of language
let boost = bus.align_importance("trading profit today");  // > 1.0 if aligned
let boost = bus.align_importance("今天交易赚了钱");           // > 1.0 if aligned
let boost = bus.align_importance("nice weather outside");   // = 1.0 (not aligned)
```

**How it works:**
- `nomic-embed-text` (768-dim) maps semantically similar content to nearby vectors regardless of language
- `score_alignment_hybrid()` = `max(keyword_score, embedding_score)` — best of both worlds
- Same-language: keyword matching wins (precise, score=1.0)
- Cross-language: embedding matching wins (semantic, score=0.1-0.3)
- Unrelated content: both return 0.0

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

## Configuration Presets

```rust
use engramai::MemoryConfig;

let config = MemoryConfig::chatbot();            // Slow decay, high replay
let config = MemoryConfig::task_agent();         // Fast decay, recent context
let config = MemoryConfig::personal_assistant(); // Very slow decay, months of memory
let config = MemoryConfig::researcher();         // Minimal forgetting
```

## Performance

| Operation | 500 memories |
|-----------|-------------|
| Store | 69ms (~0.14ms each) |
| Recall | 5ms |
| Consolidate | 60ms |
| Binary size | ~5MB |
| Memory footprint | ~5MB |

## API Reference

### Core Memory

| Method | Description |
|--------|-------------|
| `Memory::new(path, config)` | Create or open database (auto-configures extractor) |
| `Memory::with_emotional_bus(path, workspace, config)` | Create with emotional bus |
| `mem.add(content, type, importance, source, metadata)` | Store a memory |
| `mem.add_to_namespace(...)` | Store to specific namespace |
| `mem.recall(query, limit, context, min_confidence)` | Retrieve with ACT-R ranking |
| `mem.recall_from_namespace(...)` | Retrieve from namespace (`"*"` for all) |
| `mem.set_extractor(extractor)` | Override auto-configured extractor |
| `mem.clear_extractor()` | Disable extraction |
| `mem.consolidate(days)` | Run consolidation cycle |
| `mem.forget(memory_id, threshold)` | Prune weak memories |
| `mem.reward(feedback, recent_n)` | Dopaminergic feedback |
| `mem.stats()` | Memory system statistics |

### Multi-Agent ACL

| Method | Description |
|--------|-------------|
| `mem.set_agent_id(id)` | Set current agent identity |
| `mem.grant(agent_id, namespace, permission)` | Grant access |
| `mem.revoke(agent_id, namespace)` | Revoke access |
| `mem.check_permission(agent_id, namespace, action)` | Check access |

### Cross-Agent Intelligence

| Method | Description |
|--------|-------------|
| `mem.discover_cross_links(ns_a, ns_b)` | Find Hebbian associations across namespaces |
| `mem.subscribe(agent_id, namespace, min_importance)` | Subscribe to namespace notifications |
| `mem.check_notifications(agent_id)` | Get new notifications |

## Python vs Rust

| Feature | Python 2.1 | TypeScript 2.1 | Rust 0.2.1 |
|---------|------------|----------------|------------|
| ACT-R activation | ✅ | ✅ | ✅ |
| Hebbian learning | ✅ | ✅ | ✅ |
| Ebbinghaus forgetting | ✅ | ✅ | ✅ |
| Consolidation | ✅ | ✅ | ✅ |
| Hybrid Search (FTS+Embed+ACT-R) | ✅ | ✅ | ✅ |
| CJK Tokenization | ❌ | ❌ | ✅ (jieba) |
| LLM Extraction | ❌ | ❌ | ✅ (Haiku) |
| Emotional Bus | ❌ | ❌ | ✅ |
| Cross-Language Alignment | ❌ | ❌ | ✅ (embedding) |
| Multi-Agent / Namespace | ❌ | ❌ | ✅ |
| ACL | ❌ | ❌ | ✅ |
| Cross-Agent Subscriptions | ❌ | ❌ | ✅ |
| Vector embeddings | ✅ (Ollama) | ✅ (Ollama) | ✅ (Ollama) |
| MCP server | ✅ | ❌ | ⏳ planned |
| Recall latency | ~10ms | ~8ms | **~1-5ms** |
| Memory footprint | ~50MB | ~30MB | **~5MB** |
| Deployment | Requires Python | Requires Node | **Single binary** |

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{engramai,
  author = {Tang, Toni},
  title = {Engram AI: Neuroscience-Grounded Memory for AI Agents},
  year = {2026},
  url = {https://github.com/tonitangpotato/engram-ai-rust}
}
```
