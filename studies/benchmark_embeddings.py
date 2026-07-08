"""Benchmark all-MiniLM-L6-v2 vs BAAI/bge-small-en-v1.5 on the LiteratureCorpus.

Measures: model load time, embedding throughput (chunks/sec), search latency,
peak RSS memory, and retrieval quality on a fixed query set.

Requires sentence-transformers:
    pip install sentence-transformers

Run:
    python studies/benchmark_embeddings.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

# Synthetic corpus: 20 papers × ~15 chunks each = 300 chunks
# Sized to match a realistic LiteratureReviewAgent corpus (15–50 papers)
SYNTHETIC_CHUNKS = [
    f"The critical buckling stress σ_crit depends on the moment of inertia I and the "
    f"effective length L. For slender elastic rods, Euler theory gives σ_crit ∝ d⁴/L². "
    f"Paper {i}, page {(i % 5) + 1}, paragraph {(i % 3) + 1}."
    for i in range(150)
] + [
    f"Gaussian process surrogates are widely used for expensive black-box optimization. "
    f"The acquisition function balances exploration and exploitation. "
    f"Paper {i}, page {(i % 4) + 1}, paragraph {(i % 2) + 1}."
    for i in range(150)
]

QUERIES = [
    "critical buckling stress coilable metamaterial",
    "Gaussian process acquisition function Bayesian optimization",
    "surrogate model expensive black-box function",
    "elastic rod slender buckling instability",
    "moment of inertia cross-section optimization",
]


def benchmark_model(name: str, model_id: str) -> dict:
    import tracemalloc

    import numpy as np
    from sentence_transformers import SentenceTransformer

    results: dict = {"model": name}

    # --- Load time ---
    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, device="cpu")
    results["load_s"] = round(time.perf_counter() - t0, 2)

    # --- Embedding throughput ---
    tracemalloc.start()
    t0 = time.perf_counter()
    embeddings = model.encode(
        SYNTHETIC_CHUNKS,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embed_s = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = len(SYNTHETIC_CHUNKS)
    results["embed_total_s"] = round(embed_s, 2)
    results["chunks_per_sec"] = round(n / embed_s, 1)
    results["ms_per_chunk"] = round(1000 * embed_s / n, 2)
    results["peak_rss_mb"] = round(peak / 1024 / 1024, 1)
    results["embedding_dim"] = embeddings.shape[1]

    # --- Search latency (cosine similarity, top-5) ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-9)

    latencies = []
    top_chunks = {}
    for query in QUERIES:
        q_emb = model.encode([query], convert_to_numpy=True)[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        t0 = time.perf_counter()
        scores = embeddings_normed @ q_norm
        top5 = np.argsort(scores)[-5:][::-1]
        latencies.append(time.perf_counter() - t0)
        top_chunks[query] = [(round(float(scores[i]), 4), SYNTHETIC_CHUNKS[i][:60]) for i in top5]

    results["search_p50_ms"] = round(1000 * sorted(latencies)[len(latencies) // 2], 3)
    results["search_p99_ms"] = round(1000 * sorted(latencies)[-1], 3)

    # --- Quality spot-check (do top results contain expected keywords?) ---
    hits = 0
    for query, tops in top_chunks.items():
        expected = query.split()[0].lower()
        if any(expected in text.lower() for _, text in tops):
            hits += 1
    results["quality_hits"] = f"{hits}/{len(QUERIES)}"

    return results


def main():
    print(f"Corpus size: {len(SYNTHETIC_CHUNKS)} chunks")
    print(f"Queries: {len(QUERIES)}\n")

    all_results = []
    for name, model_id in MODELS.items():
        print(f"Benchmarking {name}...")
        r = benchmark_model(name, model_id)
        all_results.append(r)
        for k, v in r.items():
            if k != "model":
                print(f"  {k}: {v}")
        print()

    # Summary table
    print("=" * 60)
    print(f"{'Metric':<28} {'MiniLM-L6-v2':>15} {'bge-small-v1.5':>15}")
    print("-" * 60)
    keys = [k for k in all_results[0] if k != "model"]
    for k in keys:
        vals = [str(r[k]) for r in all_results]
        print(f"{k:<28} {vals[0]:>15} {vals[1]:>15}")

    out = Path(__file__).parent / "embedding_benchmark_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
