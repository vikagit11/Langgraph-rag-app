"""Честный eval retrieval: recall@k, MRR@k, latency p50/p95.

Конфигурации:
  dense        — поиск только по плотному вектору
  hybrid_rrf   — prefetch dense+sparse, слияние через Reciprocal Rank Fusion

Для каждого запроса известен эталон (набор doc_id), и выдача сравнивается с ним.
Разметку можно задавать двумя способами:
  * doc_id (16 hex-символов), как его посчитал stable_doc_id;
  * уникальная подстрока текста нужного чанка — скрипт сам найдёт его doc_id.

Использование:
  python eval_retrieval.py --corpus README.md --queries eval_queries.csv --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DENSE_MODEL = "thenlper/gte-large"
SPARSE_MODEL = "Qdrant/minicoil-v1"
COLLECTION = "eval"


# --------------------------------------------------------------------------- #
# Корпус и чанки
# --------------------------------------------------------------------------- #

def chunk_text(text: str, target: int = 700, overlap: int = 0) -> list[str]:
    """Режем по абзацам, склеиваем до target символов с перекрытием overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 2 > target:
            chunks.append(buf.strip())
            buf = buf[-overlap:] if overlap else ""
        buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def stable_doc_id(rel_path: str, index: int) -> str:
    """Стабильный id чанка: не зависит от порядка индексации и uuid."""
    return hashlib.sha1(f"{rel_path}#{index}".encode("utf-8")).hexdigest()[:16]


def load_corpus(path: Path, target: int = 700, overlap: int = 0) -> list[tuple[str, str, str]]:
    """Возвращает [(doc_id, source, text)]."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise SystemExit(f"в корпусе нет .md/.txt файлов: {path}")

    out: list[tuple[str, str, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.name if path.is_file() else str(f.relative_to(path)).replace("\\", "/")
        for i, chunk in enumerate(chunk_text(text, target=target, overlap=overlap)):
            out.append((stable_doc_id(rel, i), rel, chunk))
    return out


# --------------------------------------------------------------------------- #
# Индекс
# --------------------------------------------------------------------------- #

def build_index(chunks: list[tuple[str, str, str]], dense_model_name: str, sparse_model_name: str):
    from fastembed import SparseTextEmbedding, TextEmbedding

    dense_model = TextEmbedding(dense_model_name)
    sparse_model = SparseTextEmbedding(sparse_model_name)

    texts = [c[2] for c in chunks]
    dense_vecs = list(dense_model.passage_embed(texts))
    sparse_vecs = list(sparse_model.passage_embed(texts))

    dim = len(dense_vecs[0])

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE_VECTOR: models.VectorParams(
            size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )

    points = [
        models.PointStruct(
            id=idx,
            vector={DENSE_VECTOR: dense_vecs[idx], SPARSE_VECTOR: sparse_vecs[idx].as_object()},
            payload={"doc_id": chunks[idx][0], "source": chunks[idx][1], "text": chunks[idx][2]},
        )
        for idx in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return client, dense_model, sparse_model


# --------------------------------------------------------------------------- #
# Поиск
# --------------------------------------------------------------------------- #

def search_dense(client, dense_model, query: str, k: int) -> list[str]:
    vec = next(dense_model.query_embed(query))
    res = client.query_points(collection_name=COLLECTION, query=vec,
                              using=DENSE_VECTOR, limit=k, with_payload=True)
    return [p.payload["doc_id"] for p in res.points]


def search_hybrid_rrf(client, dense_model, sparse_model, query: str, k: int,
                      prefetch_limit: int = 20) -> list[str]:
    dense_vec = next(dense_model.query_embed(query))
    sparse_vec = next(sparse_model.query_embed(query))
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_VECTOR, limit=max(k, prefetch_limit)),
            models.Prefetch(query=models.SparseVector(**sparse_vec.as_object()),
                            using=SPARSE_VECTOR, limit=max(k, prefetch_limit)),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [p.payload["doc_id"] for p in res.points]


def retrieve_hybrid_pairs(client, dense_model, sparse_model, query: str, limit: int,
                          prefetch_limit: int = 20) -> list[tuple[str, str]]:
    """ Вернуть (doc_id, text) — тексты нужны реранкеру. """
    dense_vec = next(dense_model.query_embed(query))
    sparse_vec = next(sparse_model.query_embed(query))
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_VECTOR, limit=max(limit, prefetch_limit)),
            models.Prefetch(query=models.SparseVector(**sparse_vec.as_object()),
                            using=SPARSE_VECTOR, limit=max(limit, prefetch_limit)),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [(p.payload["doc_id"], p.payload["text"]) for p in res.points]


def search_hybrid_rerank(client, dense_model, sparse_model, reranker, query: str, k: int,
                         prefetch_limit: int = 20, rerank_top: int = 20) -> list[str]:
    """Двухстадийный поиск: RRF набирает кандидатов, cross-encoder их переставляет.

    Смысл: RRF хорошо находит, но плохо ранжирует. Реранкер читает пару
    (запрос, документ) целиком и даёт точную оценку релевантности — но он
    дорогой, поэтому его применяют только к верхушке кандидатов.
    """
    candidates = retrieve_hybrid_pairs(client, dense_model, sparse_model, query,
                                       limit=max(rerank_top, k), prefetch_limit=prefetch_limit)
    if not candidates:
        return []

    texts = [text for _, text in candidates]
    scores = list(reranker.rerank(query, texts))
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i][0] for i in order[:k]]


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[idx]


# --------------------------------------------------------------------------- #
# Разрешение разметки
# --------------------------------------------------------------------------- #

def resolve_relevant(raw: str, chunks: list[tuple[str, str, str]]) -> set[str]:
    """raw: части через | — либо doc_id, либо уникальная подстрока текста чанка."""
    known_ids = {c[0] for c in chunks}
    ids: set[str] = set()
    problems: list[str] = []

    for part in [p.strip() for p in raw.split("|") if p.strip()]:
        if part in known_ids:
            ids.add(part)
            continue
        hits = [c[0] for c in chunks if part in c[2]]
        if len(hits) == 1:
            ids.add(hits[0])
        elif not hits:
            problems.append(f"не найдено в корпусе: {part!r}")
        else:
            problems.append(f"неоднозначно ({len(hits)} чанков): {part!r}")

    if problems:
        raise SystemExit("Ошибка разметки:\n  " + "\n  ".join(problems))
    return ids


def load_queries(path: Path, chunks: list[tuple[str, str, str]]) -> list[tuple[str, set[str]]]:
    rows: list[tuple[str, set[str]]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            query = (row.get("query") or "").strip()
            raw = (row.get("relevant") or "").strip()
            if query and raw:
                rows.append((query, resolve_relevant(raw, chunks)))
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--configs", default="dense,hybrid_rrf")
    ap.add_argument("--target", type=int, default=700, help="целевой размер чанка в символах")
    ap.add_argument("--overlap", type=int, default=0, help="перекрытие чанков в символах")
    ap.add_argument("--dense-model", default=DENSE_MODEL)
    ap.add_argument("--sparse-model", default=SPARSE_MODEL)
    ap.add_argument("--prefetch-limit", type=int, default=20,
                    help="глубина prefetch для гибридной конфигурации")
    ap.add_argument("--rerank-model", default="Xenova/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--rerank-top", type=int, default=20,
                    help="сколько кандидатов отдавать реранкеру")
    args = ap.parse_args()

    corpus_path = Path(args.corpus).resolve()
    chunks = load_corpus(corpus_path, target=args.target, overlap=args.overlap)
    print(f"Корпус: {corpus_path}")
    print(f"Чанков: {len(chunks)}")

    queries = load_queries(Path(args.queries).resolve(), chunks)
    print(f"Запросов: {len(queries)}")
    if not queries:
        raise SystemExit("в CSV нет валидных строк query,relevant")

    client, dense_model, sparse_model = build_index(chunks, args.dense_model, args.sparse_model)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    reranker = None
    if "hybrid_rrf_rerank" in configs:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        reranker = TextCrossEncoder(model_name=args.rerank_model)

    results = []
    for cfg in configs:
        rec5 = rec10 = mrr = 0.0
        lat: list[float] = []
        for query, relevant in queries:
            t0 = time.perf_counter()
            if cfg == "dense":
                got = search_dense(client, dense_model, query, 10)
            elif cfg == "hybrid_rrf":
                got = search_hybrid_rrf(client, dense_model, sparse_model, query, 10,
                                        prefetch_limit=args.prefetch_limit)
            elif cfg == "hybrid_rrf_rerank":
                got = search_hybrid_rerank(client, dense_model, sparse_model, reranker, query, 10,
                                           prefetch_limit=args.prefetch_limit,
                                           rerank_top=args.rerank_top)
            else:
                raise SystemExit(f"неизвестная конфигурация: {cfg}")
            lat.append((time.perf_counter() - t0) * 1000)

            rec5 += recall_at_k(got, relevant, 5)
            rec10 += recall_at_k(got, relevant, 10)
            mrr += mrr_at_k(got, relevant, 10)

        n = len(queries)
        row = {
            "config": cfg,
            "n_queries": n,
            "recall@5": round(rec5 / n, 4),
            "recall@10": round(rec10 / n, 4),
            "mrr@10": round(mrr / n, 4),
            "p50_ms": round(percentile(lat, 50), 2),
            "p95_ms": round(percentile(lat, 95), 2),
        }
        results.append(row)
        print(row)

    out = Path(args.out).resolve()
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nЗаписано: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
