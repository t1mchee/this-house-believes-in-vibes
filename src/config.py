"""
Configuration & LLM client initialisation.

Loads API keys from .env and exposes pre-configured model instances.
All models use OpenAI — only one API key required.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Model configuration (all OpenAI)
# ---------------------------------------------------------------------------

# Speaker speech generation — o3 for deep, persona-faithful reasoning
SPEAKER_LLM = ChatOpenAI(
    model="o3",
    temperature=1,  # o3 only supports temperature=1
    max_completion_tokens=16384,  # o3 needs room for thinking + ~1,350-word speech
)

# Points of Information — fast reasoning model
POI_LLM = ChatOpenAI(
    model="o4-mini",
    temperature=1,  # reasoning models use temperature=1
    max_completion_tokens=1024,
)

# Verdict / audience simulation — o3 for sophisticated reasoning
VERDICT_LLM = ChatOpenAI(
    model="o3",
    temperature=1,
    max_completion_tokens=16384,
)

# Style extraction during corpus ingestion — gpt-4o is plenty here
ANALYSIS_LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=2048,
)

# Three-layer judging — gpt-4o with higher token budget for structured verdicts.
# max_retries=6 with exponential backoff handles TPM rate limits (429 errors)
# when multiple judge calls run concurrently.
JUDGE_LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=4096,
    max_retries=6,
)

# Embeddings — OpenAI text-embedding-3-large
EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

# ---------------------------------------------------------------------------
# Debate defaults
# ---------------------------------------------------------------------------
SPEECH_WORD_TARGET = 1_350  # ~7 minutes at speaking pace
MAX_REFINEMENT_ITERATIONS = 3
POI_ACCEPTANCE_PROBABILITY = 0.5  # baseline; adjusted per speaker persona
JUDGE_PANEL_SIZE = 5  # odd number to avoid ties

