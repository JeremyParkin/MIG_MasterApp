"""Frozen bounded-token baseline from 58bd526e8f8562d7232fecd1abfb4c008d970aff.

Only the two functions whose splitting changes in this slice are frozen here.
Use these original full-split functions for differential tests and benchmarks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

WEAK_SNIPPET_TOKEN_BOUNDARY = 20
WEAK_EVIDENCE_SIMILARITY_THRESHOLD = 0.95


def _leading_token_tuple(text: str, limit: int = 12) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(text.split()[:limit])


def cluster_similar_stories(df: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    df = df.copy()

    texts = (df["Normalized Headline"] + " " + df["Normalized Snippet"]).fillna("")
    if not any(text.strip() for text in texts):
        df["Group ID"] = range(len(df))
        return df

    tfidf_matrix = TfidfVectorizer().fit_transform(texts)

    if tfidf_matrix.shape[0] == 1:
        df["Group ID"] = 0
        return df

    radius = max(1.0 - similarity_threshold, 0.0)
    nn = NearestNeighbors(metric="cosine", radius=radius, algorithm="brute", n_jobs=-1)
    nn.fit(tfidf_matrix)
    graph = nn.radius_neighbors_graph(tfidf_matrix, radius=radius, mode="distance")

    snippet_token_counts = df["Normalized Snippet"].str.split().str.len().to_numpy(dtype=np.int32)
    weak_snippet_rows = snippet_token_counts < WEAK_SNIPPET_TOKEN_BOUNDARY
    source_weakness = np.repeat(weak_snippet_rows, np.diff(graph.indptr))
    edge_has_weak_evidence = source_weakness | weak_snippet_rows[graph.indices]
    weak_distance_threshold = max(1.0 - WEAK_EVIDENCE_SIMILARITY_THRESHOLD, 0.0)

    # Less text evidence requires stronger similarity to reduce false canonical merges.
    keep_edge = ~edge_has_weak_evidence | (graph.data <= min(radius, weak_distance_threshold))
    graph.data = keep_edge.astype(np.int8, copy=False)
    graph.eliminate_zeros()
    graph.data.fill(1)
    graph = graph + sparse.eye(graph.shape[0], format="csr")

    _, labels = csgraph.connected_components(graph, directed=False)
    df["Group ID"] = labels
    return df
