"""Stage 4: retrieval over passages — BM25, dense embeddings, their hybrid, and a
knowledge-graph re-rank that boosts passages connected to the query's entities.

    python retrieve.py "heatwave measures in Gujarat" [method]
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")            # keep transformers off TensorFlow/Keras 3
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import hashlib
import re

import numpy as np
from rank_bm25 import BM25Okapi

from common import PASSAGES, PROCESSED, node_id, node_kind, read_jsonl, tag_entities
from graph import load_graph

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("bm25", "dense", "hybrid", "hybrid_kg")
DENSE_WEIGHT = 0.5     # hybrid = (1 - w) * bm25 + w * dense
GRAPH_WEIGHT = 0.3     # hybrid_kg = (1 - g) * hybrid + g * graph

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the this to
was were will with which what how who where when do does did can into than then there these those
their they our we you i me my about any all""".split())


def tokenize(text):
    tokens = []
    for t in TOKEN.findall(text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


def minmax(x):
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Retriever:
    def __init__(self, passages=None, graph=None):
        self.passages = passages or read_jsonl(PASSAGES)
        self.graph = graph or load_graph()
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in self.passages])
        self.passage_entities = [self._entities_of(p) for p in self.passages]
        self._model = None
        self._embeddings = None
        self.dense_error = None

    # ---- graph side
    def _entities_of(self, p):
        """Entities and measures linked to a passage, plus the region whose plan it belongs to."""
        pid = node_id("Passage", p["id"])
        ents = {s for s in self.graph.predecessors(pid) if self.graph[s][pid]["rel"] == "MENTIONED_IN"}
        if p["region"]:
            ents.add(node_id("Region", p["region"]))
        return ents

    def query_entities(self, query):
        return {e for e in tag_entities(query) if e in self.graph}

    def expand(self, entities):
        """1-hop neighbourhood among Region / Hazard / Sector / Measure nodes."""
        keep = {"Region", "Hazard", "Sector", "Measure"}
        out = set()
        for e in entities:
            for n in list(self.graph.successors(e)) + list(self.graph.predecessors(e)):
                if node_kind(n) in keep:
                    out.add(n)
        return out - set(entities)

    def graph_scores(self, query_ents):
        if not query_ents:
            return np.zeros(len(self.passages))
        neighbours = self.expand(query_ents)
        scores = []
        for ents in self.passage_entities:
            direct = len(ents & query_ents) / len(query_ents)
            near = min(1.0, len(ents & neighbours) / 3)
            scores.append(0.8 * direct + 0.2 * near)
        return np.array(scores)

    # ---- text side
    def bm25_scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)

    def _load_dense(self):
        if self._embeddings is not None or self.dense_error:
            return
        try:
            from sentence_transformers import SentenceTransformer
            try:   # skip the slow hub round-trips once the model is cached locally
                self._model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(MODEL_NAME)
            cache_ids, cache_vec = PROCESSED / "embedding_ids.txt", PROCESSED / "embeddings.npy"
            ids = [p["id"] + "|" + hashlib.md5(p["text"].encode("utf-8")).hexdigest()[:12] for p in self.passages]
            if cache_ids.exists() and cache_vec.exists() and cache_ids.read_text(encoding="utf-8").split("\n") == ids:
                self._embeddings = np.load(cache_vec)
            else:
                self._embeddings = self._model.encode([p["text"] for p in self.passages],
                                                      normalize_embeddings=True, show_progress_bar=False)
                np.save(cache_vec, self._embeddings)
                cache_ids.write_text("\n".join(ids), encoding="utf-8")
        except Exception as exc:  # no model / offline: fall back to lexical-only
            self.dense_error = f"{type(exc).__name__}: {exc}"

    def dense_scores(self, query):
        self._load_dense()
        if self._embeddings is None:
            return None
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self._embeddings @ q

    # ---- fusion
    def score(self, query, method="hybrid_kg"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        components = {"bm25": self.bm25_scores(query)}
        if method != "bm25":
            dense = self.dense_scores(query)
            if dense is not None:
                components["dense"] = dense
        query_ents = self.query_entities(query) if method == "hybrid_kg" else set()
        if method == "bm25":
            total = components["bm25"]
        elif method == "dense":
            total = components.get("dense", components["bm25"])
        else:
            b = minmax(components["bm25"])
            total = b if "dense" not in components else (1 - DENSE_WEIGHT) * b + DENSE_WEIGHT * minmax(components["dense"])
            if method == "hybrid_kg":
                components["graph"] = self.graph_scores(query_ents)
                total = (1 - GRAPH_WEIGHT) * total + GRAPH_WEIGHT * components["graph"]
        return total, components, query_ents

    def search(self, query, method="hybrid_kg", k=10):
        total, components, query_ents = self.score(query, method)
        order = np.argsort(-total, kind="stable")[:k]
        return {
            "query_entities": sorted(query_ents),
            "dense_error": self.dense_error,
            "results": [{
                "rank": r + 1,
                "score": float(total[i]),
                "parts": {name: float(v[i]) for name, v in components.items()},
                "passage": self.passages[i],
                "entities": sorted(self.passage_entities[i]),
            } for r, i in enumerate(order)],
        }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "heatwave measures in Gujarat"
    method = sys.argv[2] if len(sys.argv) > 2 else "hybrid_kg"
    out = Retriever().search(query, method, k=5)
    print(f"Query: {query!r}  method={method}  entities={out['query_entities']}")
    if out["dense_error"]:
        print("  (dense disabled:", out["dense_error"], ")")
    for r in out["results"]:
        p = r["passage"]
        print(f"\n#{r['rank']}  {r['score']:.3f}  [{p['id']}]  {p['title']}")
        print("   ", p["text"][:220] + ("..." if len(p["text"]) > 220 else ""))


if __name__ == "__main__":
    main()
