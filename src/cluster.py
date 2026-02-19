"""
Argument cluster analysis pipeline.

Reads all run_*_data.json files from ensemble output directories,
extracts claims, embeds them, clusters with UMAP + HDBSCAN,
labels clusters with gpt-4o, and outputs a precomputed viz_data.json
for the interactive GitHub Pages visualization.

Usage:
    python -m src.cluster
"""

from __future__ import annotations

import asyncio
import json
import glob
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. EXTRACT claims from all ensemble runs
# ---------------------------------------------------------------------------

def extract_claims(output_dir: str = "output") -> list[dict]:
    """Walk all ensemble directories and extract every annotated claim."""
    claims = []
    json_files = sorted(glob.glob(os.path.join(output_dir, "ensemble_*/run_*_data.json")))

    if not json_files:
        print(f"No run_*_data.json files found in {output_dir}/ensemble_*/")
        sys.exit(1)

    for fpath in json_files:
        try:
            data = json.load(open(fpath))
        except (json.JSONDecodeError, IOError):
            continue

        ensemble_dir = os.path.basename(os.path.dirname(fpath))
        run_num = data.get("run_number", 0)
        epoch = data.get("epoch", 1)
        winner = data.get("winner", "unknown")
        margin = data.get("margin", "unknown")
        engagement_winner = data.get("engagement_winner", None)

        annotation = data.get("division", {}).get("annotation", {})
        ann_claims = annotation.get("claims", [])
        rebuttals = annotation.get("rebuttals", [])

        # Build rebuttal lookup: claim_id -> list of rebuttals targeting it
        rebuttal_map: dict[str, list[dict]] = {}
        for r in rebuttals:
            tid = r.get("target_claim_id", "")
            rebuttal_map.setdefault(tid, []).append(r)

        # Build audit lookup: claim_text -> audit info
        audit_claims = data.get("division", {}).get("argument_audit", {}).get("claims", [])
        audit_map: dict[str, dict] = {}
        for ac in audit_claims:
            audit_map[ac.get("claim", "")] = ac

        for c in ann_claims:
            claim_id = c.get("claim_id", "")
            claim_text = c.get("claim_text", "")

            # Find audit match (fuzzy — audit text is often a shortened version)
            audit = None
            for audit_text, audit_data in audit_map.items():
                if audit_text[:40] in claim_text or claim_text[:40] in audit_text:
                    audit = audit_data
                    break

            # Determine rebuttal fate
            targeted_rebuttals = rebuttal_map.get(claim_id, [])
            was_rebutted = len(targeted_rebuttals) > 0
            was_demolished = any(
                r.get("undermines_original", False) and r.get("addresses_specific_logic", False)
                for r in targeted_rebuttals
            )
            survived = audit.get("survives", True) if audit else (not was_demolished)

            side = c.get("side", "unknown")
            side_won = (side == winner.lower()) if winner != "unknown" else None

            claims.append({
                "claim_text": claim_text,
                "claim_type": c.get("claim_type", "assertion"),
                "specificity": c.get("specificity", "generic"),
                "speaker_name": c.get("speaker_name", "Unknown"),
                "side": side,
                "claim_id": claim_id,
                # Run metadata
                "run_number": run_num,
                "epoch": epoch,
                "ensemble": ensemble_dir,
                "winner": winner,
                "margin": margin,
                "engagement_winner": engagement_winner,
                "side_won": side_won,
                # Rebuttal fate
                "was_rebutted": was_rebutted,
                "was_demolished": was_demolished,
                "survived": survived,
                "n_rebuttals": len(targeted_rebuttals),
            })

    return claims


# ---------------------------------------------------------------------------
# 2. EMBED claims
# ---------------------------------------------------------------------------

def embed_claims(claims: list[dict]) -> np.ndarray:
    """Embed all claim texts using text-embedding-3-large."""
    from openai import OpenAI

    client = OpenAI()
    texts = [c["claim_text"] for c in claims]

    print(f"  Embedding {len(texts)} claims...")

    # Batch in groups of 100 (API limit is 2048 but smaller batches are safer)
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"    Batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} done")

    return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# 3. REDUCE with UMAP
# ---------------------------------------------------------------------------

def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project high-dimensional embeddings to 3D with UMAP."""
    import umap

    print(f"  Running UMAP on {embeddings.shape[0]} points...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    return coords


# ---------------------------------------------------------------------------
# 4. CLUSTER with HDBSCAN
# ---------------------------------------------------------------------------

def cluster_hdbscan(coords: np.ndarray) -> np.ndarray:
    """Cluster 2D UMAP coordinates with HDBSCAN."""
    import hdbscan

    print(f"  Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    print(f"    Found {n_clusters} clusters, {n_noise} noise points")
    return labels


# ---------------------------------------------------------------------------
# 5. LABEL clusters with LLM
# ---------------------------------------------------------------------------

async def label_clusters(claims: list[dict], labels: np.ndarray) -> dict[int, dict]:
    """Ask gpt-4o to name and describe each cluster."""
    from langchain_openai import ChatOpenAI
    import asyncio as _asyncio

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=512, max_retries=6)

    cluster_ids = sorted(set(labels))
    cluster_info: dict[int, dict] = {}

    for cid in cluster_ids:
        if cid == -1:
            cluster_info[-1] = {
                "name": "Unclustered",
                "description": "Arguments that don't fit neatly into any thematic group.",
                "color": "#888888",
            }
            continue

        member_texts = [
            claims[i]["claim_text"]
            for i in range(len(claims))
            if labels[i] == cid
        ]

        # Sample up to 20 for the prompt
        sample = member_texts[:20] if len(member_texts) <= 20 else (
            [member_texts[j] for j in np.linspace(0, len(member_texts) - 1, 20, dtype=int)]
        )

        prompt = f"""Here are {len(sample)} argument claims that cluster together based on semantic similarity. They come from simulated Cambridge Union debates on the motion "This House Believes AI Should Be Allowed To Make Decisions About Human Life."

CLAIMS:
{chr(10).join(f'- {t}' for t in sample)}

Provide:
1. A short name for this cluster (3-6 words, e.g. "Precautionary Principle", "Global Health Equity", "Algorithmic Accountability")
2. A one-sentence description of what unifies these arguments.

Respond in JSON: {{"name": "...", "description": "..."}}"""

        # Retry with backoff for rate limits
        for attempt in range(5):
            try:
                response = await llm.ainvoke(prompt)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt + 1
                    print(f"    Rate limited, waiting {wait}s...")
                    await _asyncio.sleep(wait)
                else:
                    raise
        else:
            cluster_info[cid] = {"name": f"Cluster {cid}", "description": "Label failed (rate limit)"}
            print(f"    Cluster {cid} ({len(member_texts)} claims): [label failed]")
            continue

        try:
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            info = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            info = {"name": f"Cluster {cid}", "description": response.content[:200]}

        cluster_info[cid] = info
        print(f"    Cluster {cid} ({len(member_texts)} claims): {info['name']}")

    return cluster_info


# ---------------------------------------------------------------------------
# 6. COMPUTE derived statistics per cluster
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    claims: list[dict], labels: np.ndarray, cluster_info: dict[int, dict]
) -> dict[int, dict]:
    """Compute per-cluster statistics for the visualization."""
    stats: dict[int, dict] = {}

    for cid in sorted(set(labels)):
        members = [claims[i] for i in range(len(claims)) if labels[i] == cid]
        n = len(members)
        if n == 0:
            continue

        # Side distribution
        n_prop = sum(1 for m in members if m["side"] == "proposition")
        n_opp = n - n_prop

        # Win rate (when this argument's side won)
        side_won_vals = [m["side_won"] for m in members if m["side_won"] is not None]
        win_rate = sum(side_won_vals) / len(side_won_vals) if side_won_vals else 0.5

        # Survival rate
        survival_rate = sum(1 for m in members if m["survived"]) / n

        # Rebuttal rate
        rebuttal_rate = sum(1 for m in members if m["was_rebutted"]) / n

        # Demolition rate
        demolition_rate = sum(1 for m in members if m["was_demolished"]) / n

        # Speaker distribution
        speakers: dict[str, int] = {}
        for m in members:
            speakers[m["speaker_name"]] = speakers.get(m["speaker_name"], 0) + 1

        # Epoch distribution
        epochs: dict[int, int] = {}
        for m in members:
            epochs[m["epoch"]] = epochs.get(m["epoch"], 0) + 1

        # Claim type distribution
        types: dict[str, int] = {}
        for m in members:
            types[m["claim_type"]] = types.get(m["claim_type"], 0) + 1

        stats[cid] = {
            "count": n,
            "prop_count": n_prop,
            "opp_count": n_opp,
            "prop_fraction": n_prop / n,
            "win_rate": win_rate,
            "survival_rate": survival_rate,
            "rebuttal_rate": rebuttal_rate,
            "demolition_rate": demolition_rate,
            "speakers": speakers,
            "epochs": epochs,
            "claim_types": types,
        }

    # Merge into cluster_info
    for cid, info in cluster_info.items():
        if cid in stats:
            info.update(stats[cid])

    return cluster_info


# ---------------------------------------------------------------------------
# 7. EXTRACT rebuttal edges
# ---------------------------------------------------------------------------

def extract_edges(claims: list[dict], output_dir: str = "output") -> list[dict]:
    """Extract directed rebuttal edges between claims.

    Each edge goes from the *attacked* claim to the *rebutting speaker's*
    closest claim in the same run — i.e. source is the claim being
    challenged, target is the claim from the speaker who made the rebuttal.
    """
    edges = []
    json_files = sorted(glob.glob(os.path.join(output_dir, "ensemble_*/run_*_data.json")))

    # Build lookup: (ensemble, run, claim_id) -> index in claims list
    claim_index: dict[tuple, int] = {}
    for i, c in enumerate(claims):
        key = (c["ensemble"], c["run_number"], c["claim_id"])
        claim_index[key] = i

    # Build lookup: (ensemble, run, speaker_name) -> list of claim indices
    speaker_claims: dict[tuple, list[int]] = {}
    for i, c in enumerate(claims):
        key = (c["ensemble"], c["run_number"], c["speaker_name"])
        speaker_claims.setdefault(key, []).append(i)

    for fpath in json_files:
        try:
            data = json.load(open(fpath))
        except (json.JSONDecodeError, IOError):
            continue

        ensemble_dir = os.path.basename(os.path.dirname(fpath))
        run_num = data.get("run_number", 0)
        rebuttals = data.get("division", {}).get("annotation", {}).get("rebuttals", [])

        for r in rebuttals:
            target_id = r.get("target_claim_id", "")
            rebutting_speaker = r.get("rebutting_speaker", "")

            # Source = the claim being attacked
            source_key = (ensemble_dir, run_num, target_id)
            source_idx = claim_index.get(source_key)
            if source_idx is None:
                continue

            # Target = pick the first claim by the rebutting speaker in this run
            speaker_key = (ensemble_dir, run_num, rebutting_speaker)
            candidates = speaker_claims.get(speaker_key, [])
            if not candidates:
                continue
            target_idx = candidates[0]

            # Don't create self-loops
            if source_idx == target_idx:
                continue

            undermines = (
                r.get("undermines_original", False) and
                r.get("addresses_specific_logic", False)
            )

            edges.append({
                "source": source_idx,
                "target": target_idx,
                "summary": r.get("rebuttal_summary", "")[:150],
                "undermines": undermines,
                "method": r.get("method", ""),
                "engagement": r.get("engagement_level", ""),
            })

    return edges


# ---------------------------------------------------------------------------
# 8. OUTPUT viz_data.json
# ---------------------------------------------------------------------------

def build_output(
    claims: list[dict],
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_info: dict[int, dict],
    edges: list[dict],
) -> dict:
    """Build the final JSON structure for the visualization."""
    points = []
    for i, claim in enumerate(claims):
        points.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
            "cluster": int(labels[i]),
            "cluster_name": cluster_info.get(int(labels[i]), {}).get("name", "Unknown"),
            **claim,
        })

    return {
        "meta": {
            "n_claims": len(claims),
            "n_clusters": len([k for k in cluster_info if k != -1]),
            "n_runs": len(set(
                (c["ensemble"], c["run_number"]) for c in claims
            )),
            "n_ensembles": len(set(c["ensemble"] for c in claims)),
            "n_edges": len(edges),
            "motion": "This House Believes AI Should Be Allowed To Make Decisions About Human Life",
        },
        "clusters": {str(k): v for k, v in cluster_info.items()},
        "points": points,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("ARGUMENT CLUSTER ANALYSIS")
    print("=" * 60)

    # Step 1: Extract
    print("\n1. Extracting claims from all ensemble runs...")
    claims = extract_claims("output")
    print(f"   Found {len(claims)} claims across {len(set(c['ensemble'] for c in claims))} ensembles")

    if len(claims) < 10:
        print("   Too few claims for meaningful clustering. Need at least 10.")
        sys.exit(1)

    # Step 2: Embed
    print("\n2. Embedding claims...")
    embeddings = embed_claims(claims)

    # Step 3: UMAP
    print("\n3. Reducing dimensions with UMAP...")
    coords = reduce_umap(embeddings)

    # Step 4: Cluster
    print("\n4. Clustering with HDBSCAN...")
    labels = cluster_hdbscan(coords)

    # Step 5: Label
    print("\n5. Labeling clusters with gpt-4o...")
    cluster_info = await label_clusters(claims, labels)

    # Step 6: Statistics
    print("\n6. Computing cluster statistics...")
    cluster_info = compute_cluster_stats(claims, labels, cluster_info)

    # Step 7: Extract edges
    print("\n7. Extracting rebuttal edges...")
    edges = extract_edges(claims, "output")
    n_demolitions = sum(1 for e in edges if e["undermines"])
    print(f"   Found {len(edges)} edges ({n_demolitions} demolitions)")

    # Step 8: Output
    print("\n8. Writing viz_data.json...")
    os.makedirs("docs", exist_ok=True)
    output = build_output(claims, coords, labels, cluster_info, edges)
    output_path = os.path.join("docs", "viz_data.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n   Written to {output_path}")
    print(f"   {output['meta']['n_claims']} claims, {output['meta']['n_clusters']} clusters, "
          f"{output['meta']['n_runs']} runs, {output['meta']['n_edges']} edges")

    # Print cluster summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    for cid_str, info in sorted(output["clusters"].items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        name = info.get("name", "?")
        count = info.get("count", 0)
        win_rate = info.get("win_rate", 0)
        survival = info.get("survival_rate", 0)
        prop_frac = info.get("prop_fraction", 0)
        side_label = "PROP" if prop_frac > 0.6 else ("OPP" if prop_frac < 0.4 else "MIXED")
        print(f"  [{cid:2d}] {name:<35s} n={count:3d}  "
              f"side={side_label:<5s}  win={win_rate:.0%}  surv={survival:.0%}")

    print("\nDone. Open docs/index.html to view the visualization.")


if __name__ == "__main__":
    asyncio.run(main())

