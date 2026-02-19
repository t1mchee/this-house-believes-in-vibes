"""
Phase 0: Corpus Ingestion

Loads speaker documents (PDFs, transcripts, text files) into per-speaker
ChromaDB collections with metadata tagging. Also extracts rhetorical style
profiles for each speaker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PERSIST_DIR, EMBEDDINGS, ANALYSIS_LLM
from src.models import SpeakerProfile, StyleProfile


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

def get_chroma_client() -> PersistentClient:
    """Return a persistent ChromaDB client."""
    return PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(client: PersistentClient, name: str):
    """Get or create a ChromaDB collection for a speaker."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Document loading & chunking
# ---------------------------------------------------------------------------

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _extract_pdf_text(filepath: Path) -> str:
    """Extract text from a PDF using pypdf (fast, no external deps)."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_metadata_from_header(text: str) -> dict:
    """
    Parse optional YAML-style metadata from the top of a markdown file.

    Looks for lines like:
        - **Speaker**: Jane Smith
        - **Date**: 2024-03-15
        - **Source**: BBC HARDtalk
        - **Type**: Interview transcript
    """
    import re

    metadata = {}
    for match in re.finditer(r"[-*]\s*\*?\*?(\w+)\*?\*?\s*:\s*(.+)", text[:500]):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key in {"speaker", "date", "source", "type", "topic"}:
            metadata[key] = value
    return metadata


def load_documents_from_directory(speaker_dir: Path) -> list[dict]:
    """
    Load all documents from a speaker's data directory.

    Supports: .md, .txt, .pdf

    Expected structure:
        data/speakers/{speaker_id}/
            bio/         ← bio.md (background + key positions)
            speeches/    ← talk transcripts, lecture notes
            interviews/  ← interview transcripts, podcast appearances
            writings/    ← articles, op-eds, book excerpts, papers

    Returns a list of {"text": ..., "metadata": {...}} dicts.
    """
    documents = []

    for root, _dirs, files in os.walk(speaker_dir):
        for filename in files:
            filepath = Path(root) / filename

            # Skip hidden files and non-document files
            if filename.startswith("."):
                continue

            # Determine source type from directory name
            source_type = Path(root).name  # e.g. "speeches", "interviews"
            if source_type == speaker_dir.name:
                source_type = "general"

            # Load based on file type
            text = None
            if filepath.suffix in {".txt", ".md"}:
                text = filepath.read_text(encoding="utf-8")
            elif filepath.suffix == ".pdf":
                try:
                    text = _extract_pdf_text(filepath)
                except Exception as e:
                    print(f"  ⚠️  Failed to read PDF {filepath}: {e}")
                    continue
            else:
                continue  # Skip unsupported formats

            if not text or not text.strip():
                continue

            # Extract any inline metadata from markdown headers
            inline_meta = _extract_metadata_from_header(text) if filepath.suffix == ".md" else {}

            documents.append({
                "text": text,
                "metadata": {
                    "source_file": str(filepath),
                    "source_type": source_type,
                    "filename": filename,
                    "file_type": filepath.suffix,
                    **inline_meta,
                },
            })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks, preserving metadata."""
    chunks = []
    for doc in documents:
        splits = TEXT_SPLITTER.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                },
            })
    return chunks


# ---------------------------------------------------------------------------
# Embedding & storage
# ---------------------------------------------------------------------------

async def ingest_speaker_corpus(
    speaker: SpeakerProfile,
    data_dir: Path | str,
) -> int:
    """
    Ingest a speaker's full corpus into their ChromaDB collection.

    Args:
        speaker: The speaker profile.
        data_dir: Path to the speaker's data directory.

    Returns:
        Number of chunks ingested.
    """
    data_dir = Path(data_dir)
    client = get_chroma_client()
    collection = get_or_create_collection(client, speaker.corpus_collection)

    # Load and chunk
    documents = load_documents_from_directory(data_dir)
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(f"No documents found for {speaker.name} in {data_dir}")

    # Embed and store
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{speaker.id}_chunk_{i}" for i in range(len(chunks))]

    # Embed in batches
    embeddings = await EMBEDDINGS.aembed_documents(texts)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(chunks)


# ---------------------------------------------------------------------------
# Style extraction
# ---------------------------------------------------------------------------

STYLE_EXTRACTION_PROMPT = """Analyse this corpus by {speaker_name}. Extract the following
attributes of their rhetorical and communicative style. Be specific — use quotes and
examples from the corpus wherever possible.

1. **Speech register**: formal academic / conversational / polemical / humorous / etc.
2. **Opening patterns**: Do they start with anecdotes? Data? Provocations? Personal stories?
3. **Characteristic rhetorical devices**: Tricolons, rhetorical questions, analogies,
   sarcasm, understatement, appeals to authority, personal testimony, etc.
4. **Disagreement style**: Aggressive rebuttal? Diplomatic concession-then-counter?
   Dismissive? Evidence-heavy takedown?
5. **Signature phrases or verbal tics**: Any recurring expressions or patterns.
6. **Closing patterns**: How they typically close a speech or argument.

Corpus:
---
{corpus_sample}
---

Respond in structured detail."""


async def extract_style_profile(
    speaker: SpeakerProfile,
    corpus_sample: str,
) -> StyleProfile:
    """
    Use an LLM to extract a rhetorical style profile from a speaker's corpus.

    Args:
        speaker: The speaker profile.
        corpus_sample: A representative sample of the speaker's writing/speeches.

    Returns:
        A StyleProfile with extracted attributes.
    """
    prompt = STYLE_EXTRACTION_PROMPT.format(
        speaker_name=speaker.name,
        corpus_sample=corpus_sample,
    )

    response = await ANALYSIS_LLM.ainvoke(prompt)
    raw_analysis = response.content

    # Parse into structured profile using a second LLM call with structured output
    parse_prompt = f"""Based on this style analysis, extract the structured fields.

Analysis:
{raw_analysis}

Return a JSON object with these fields:
- speech_register (string)
- opening_patterns (list of strings)
- rhetorical_devices (list of strings)
- disagreement_style (string)
- signature_phrases (list of strings)
- closing_patterns (list of strings)"""

    structured = await ANALYSIS_LLM.with_structured_output(StyleProfile).ainvoke(parse_prompt)

    # Attach the raw analysis
    structured.raw_analysis = raw_analysis

    return structured


# ---------------------------------------------------------------------------
# RAG retrieval helper
# ---------------------------------------------------------------------------

async def retrieve_relevant_passages(
    speaker: SpeakerProfile,
    query: str,
    k: int = 10,
) -> list[str]:
    """
    Retrieve the top-k most relevant passages from a speaker's corpus.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, speaker.corpus_collection)

    query_embedding = await EMBEDDINGS.aembed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    return results["documents"][0] if results["documents"] else []


async def retrieve_for_topics(
    speaker: SpeakerProfile,
    queries: list[str],
    k_per_query: int = 5,
) -> list[str]:
    """
    Retrieve passages across multiple topic queries, deduplicated.
    """
    seen = set()
    passages = []

    for query in queries:
        results = await retrieve_relevant_passages(speaker, query, k=k_per_query)
        for passage in results:
            if passage not in seen:
                seen.add(passage)
                passages.append(passage)

    return passages


# ---------------------------------------------------------------------------
# Passage pool for ensemble variation
# ---------------------------------------------------------------------------

async def generate_retrieval_queries(
    speaker: SpeakerProfile,
    motion: str,
    n_queries: int = 4,
) -> list[str]:
    """
    Generate diverse RAG queries tailored to a speaker's expertise.

    Returns the motion itself plus n_queries additional queries covering
    different facets of the speaker's likely arguments.
    """
    from src.models import Side

    side_label = "Proposition (FOR)" if speaker.side == Side.PROPOSITION else "Opposition (AGAINST)"

    prompt = f"""Generate {n_queries} search queries to find DIFFERENT aspects
of {speaker.name}'s writings and speeches relevant to this debate motion:

"{motion}"

Background: {speaker.bio}
Side: {side_label}

Requirements:
- Each query should target a DIFFERENT facet of what this speaker might
  draw on (e.g. empirical evidence, philosophical arguments, policy
  positions, case studies, personal research, rhetorical examples).
- Queries should be 5-15 words each — concise search terms, not sentences.
- Cover the breadth of material this speaker could use.

Return one query per line. No numbering, no bullets, no extra text."""

    response = await ANALYSIS_LLM.ainvoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    # Always include the motion itself as the baseline query
    return [motion] + queries[:n_queries]


async def build_passage_pool(
    speaker: SpeakerProfile,
    motion: str,
    n_queries: int = 4,
    k_per_query: int = 6,
) -> list[str]:
    """
    Build a large, diverse pool of passages for a speaker.

    Uses multiple retrieval queries to surface different facets of the
    speaker's corpus. The pool is then randomly subsampled per-run in
    the ensemble to introduce natural retrieval variation.

    Typical pool size: 15-25 unique passages (from ~5 queries × 6 each,
    minus duplicates).
    """
    queries = await generate_retrieval_queries(speaker, motion, n_queries)
    pool = await retrieve_for_topics(speaker, queries, k_per_query)
    return pool

