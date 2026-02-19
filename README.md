# This House Believes In Vibes

**Multi-agent debate simulation using persona-grounded LLMs.**

Simulates a full Cambridge Union Exhibition Debate (3v3) by building speaker personas from real-world corpora — speeches, interviews, academic papers — and running adversarial multi-agent debate through a LangGraph state machine. Supports Monte Carlo ensemble runs with iterative student learning, three-layer judging, and cluster analysis of emergent argument landscapes.

Built as a pre-debate prediction experiment for the Cambridge Union Lent Term Fifth Debate 2026:
*"This House Believes AI Should Be Allowed To Make Decisions About Human Life."*

**[Full methodology write-up](METHODOLOGY.md)**

[https://t1mchee.github.io/this-house-believes-in-vibes/]

---

## How It Works

```
Corpus (PDFs, speeches, interviews)
  → RAG-grounded speaker personas (ChromaDB + OpenAI embeddings)
    → 6-speech sequential debate with POIs (LangGraph)
      → Three-layer judging (rubric + annotation + engagement)
        → Ensemble: N runs × M epochs with coaching feedback
          → Cluster analysis of argument landscape
```

**Models used:** OpenAI `o3` (speakers, verdict), `o4-mini` (POIs), `gpt-4o` (judging, analysis), `text-embedding-3-large` (RAG + clustering).

## Quick Start

```bash
# Clone and setup
git clone https://github.com/t1mchee/this-house-believes-in-vibes.git
cd this-house-believes-in-vibes
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI and (optional) LangSmith keys

# Populate data/speakers/ with speaker corpora (see data/speakers/DATA_GUIDE.md)

# Run a single debate
python -m src.run

# Run an ensemble (20 runs, 4 epochs, with coaching)
python -m src.ensemble

# Run cluster analysis on completed ensemble data
python -m src.cluster
```

## Repository Structure

```
├── src/
│   ├── config.py          # LLM clients, model configuration
│   ├── models.py          # Pydantic data models
│   ├── graph.py           # LangGraph state machine (debate flow)
│   ├── run.py             # Single debate entry point
│   ├── ensemble.py        # Monte Carlo ensemble runner
│   ├── coaching.py        # Epoch-based student learning
│   ├── cluster.py         # Argument cluster analysis
│   ├── corpus/
│   │   ├── ingest.py      # Document loading, chunking, ChromaDB
│   │   └── youtube.py     # YouTube transcript downloader
│   ├── debate/
│   │   ├── speech.py      # Speech generation, POI mechanics
│   │   └── judge.py       # Three-layer judging system
│   └── persona/
│       └── builder.py     # Persona prompt construction from RAG
├── data/
│   └── speakers/          # Speaker corpora (bio, speeches, writings)
├── docs/
│   ├── index.html         # Interactive 3D argument landscape visualisation
│   ├── architecture.md    # Debate format specification
│   └── viz_data.json      # Generated cluster data for visualisation
├── export/                # Saved debate transcripts
├── METHODOLOGY.md         # Full methodology blog post with references
└── pyproject.toml         # Dependencies
```

## Judging System

The simulation uses a three-layer approach to reduce LLM judge bias:

| Layer | Method | Purpose |
|-------|--------|---------|
| **Rubric scoring** | Per-speech dimensional scores (1–10) with anchored descriptors | Granular quality signal |
| **Annotation verdict** | LLM-annotated claims + rebuttals → mechanical tally | Transparent, auditable |
| **Engagement verdict** | Anonymised, side-swapped LLM evaluation | Primary outcome signal |

See [METHODOLOGY.md](METHODOLOGY.md) for the full rationale, including observations on systematic LLM judge bias and the debiasing measures implemented.

## Visualisation

The cluster analysis produces an interactive 3D scatter plot showing how arguments cluster across runs:

```bash
python -m src.cluster          # generates docs/viz_data.json
cd docs && python -m http.server 8080  # open http://localhost:8080
```

## Requirements

- Python ≥ 3.11
- OpenAI API key (models: o3, o4-mini, gpt-4o, text-embedding-3-large)
- Optional: LangSmith API key for tracing

## License

MIT

