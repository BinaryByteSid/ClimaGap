"""Stage 6: compare retrieval methods on the labelled queries in eval/queries.json.

Metrics: Precision@5, Recall@10, MRR, nDCG@10 (graded relevance).
"""
import json
import math

from common import EVAL
from retrieve import METHODS, Retriever

K_P, K_R, K_NDCG = 5, 10, 10


def load_judgments(passages):
    spec = json.loads((EVAL / "queries.json").read_text(encoding="utf-8"))
    queries = []
    for q in spec["queries"]:
        qrels = {}
        for rel in q["relevant"]:
            if "id" in rel:   # judged from a pooled run: exact passage id
                hits = [p["id"] for p in passages if p["id"] == rel["id"]]
            else:             # doc + phrase: survives re-chunking
                hits = [p["id"] for p in passages
                        if p["doc_id"] == rel["doc"] and rel["contains"].lower() in p["text"].lower()]
            if not hits:
                print(f"  warning: {q['id']} judgment matches no passage: {rel}")
            for pid in hits:
                qrels[pid] = max(qrels.get(pid, 0), rel["grade"])
        queries.append({"id": q["id"], "query": q["query"], "qrels": qrels})
    return queries


def precision_at(ranked, qrels, k):
    return sum(1 for pid in ranked[:k] if pid in qrels) / k


def recall_at(ranked, qrels, k):
    return sum(1 for pid in ranked[:k] if pid in qrels) / len(qrels) if qrels else 0.0


def reciprocal_rank(ranked, qrels):
    return next((1 / (i + 1) for i, pid in enumerate(ranked) if pid in qrels), 0.0)


def ndcg_at(ranked, qrels, k):
    dcg = sum((2 ** qrels.get(pid, 0) - 1) / math.log2(i + 2) for i, pid in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(retriever=None, methods=METHODS):
    retriever = retriever or Retriever()
    queries = load_judgments(retriever.passages)
    table, per_query = {}, []
    for method in methods:
        sums = {"P@5": 0.0, "R@10": 0.0, "MRR": 0.0, "nDCG@10": 0.0}
        for q in queries:
            ranked = [r["passage"]["id"] for r in retriever.search(q["query"], method, k=K_R)["results"]]
            scores = {"P@5": precision_at(ranked, q["qrels"], K_P),
                      "R@10": recall_at(ranked, q["qrels"], K_R),
                      "MRR": reciprocal_rank(ranked, q["qrels"]),
                      "nDCG@10": ndcg_at(ranked, q["qrels"], K_NDCG)}
            per_query.append({"method": method, "id": q["id"], "query": q["query"], **scores})
            for m, v in scores.items():
                sums[m] += v
        table[method] = {m: round(v / len(queries), 3) for m, v in sums.items()}
    return table, per_query, retriever.dense_error


def main():
    import pandas as pd
    table, per_query, dense_error = evaluate()
    if dense_error:
        print("Dense retrieval unavailable, dense/hybrid fall back to BM25:", dense_error)
    df = pd.DataFrame(table).T
    df.index.name = "method"
    print(f"\nRetrieval evaluation over {len(per_query) // len(table)} labelled queries\n")
    print(df.to_string())
    df.to_csv(EVAL / "results.csv")
    pd.DataFrame(per_query).to_csv(EVAL / "results_per_query.csv", index=False)
    print("\nSaved eval/results.csv and eval/results_per_query.csv")


if __name__ == "__main__":
    main()
