"""Замер влияния Binary Quantization на память Qdrant.

Скрипт наполняет коллекции одинаковым набором векторов, чтобы можно было
сравнить потребление RAM контейнера. Запускается по шагам, чтобы между
шагами снять `docker stats`:

    python bench_quantization.py --mode reset
    python bench_quantization.py --mode plain --copies 500
    python bench_quantization.py --mode bq    --copies 500

Идея: скормить обоим вариантам ровно одни и те же векторы, тогда разница в
RSS контейнера — это и есть разница между float32 и binary представлением.
Векторы берутся реальные (эмбеддинги корпуса), просто повторяются N раз:
для замера памяти это корректно, для качества — нет (качество меряет
eval_retrieval.py на уникальных чанках).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

QDRANT_URL = "http://localhost:6333"
DENSE = "dense"
SPARSE = "sparse"
PLAIN = "bench_plain"
BQ = "bench_bq"


def load_chunks(corpus: Path) -> list[str]:
    texts: list[str] = []
    for f in sorted(corpus.rglob("*.md")):
        for block in f.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if len(block) > 120:
                texts.append(block)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reset", "plain", "bq"], required=True)
    ap.add_argument("--copies", type=int, default=500)
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--on-disk", action="store_true",
                    help="хранить оригинальные векторы на диске, в RAM держать только квантованные")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL)

    if args.mode == "reset":
        for name in (PLAIN, BQ):
            try:
                client.delete_collection(name)
                print("deleted", name)
            except Exception:
                pass
        return 0

    from fastembed import SparseTextEmbedding, TextEmbedding

    texts = load_chunks(Path(args.corpus))
    dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding("Qdrant/minicoil-v1")

    dense_vecs = list(dense_model.passage_embed(texts))
    sparse_vecs = list(sparse_model.passage_embed(texts))
    dim = len(dense_vecs[0])

    name = PLAIN if args.mode == "plain" else BQ
    dense_params = models.VectorParams(size=dim, distance=models.Distance.COSINE,
                                       on_disk=args.on_disk)
    if args.mode == "bq":
        # VectorParams ждёт серверную модель BinaryQuantization, а не *Config
        dense_params.quantization_config = models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        )

    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: dense_params},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    total = len(texts) * args.copies
    batch: list[models.PointStruct] = []
    pid = 0
    for _ in range(args.copies):
        for dv, sv in zip(dense_vecs, sparse_vecs):
            batch.append(models.PointStruct(
                id=pid,
                vector={DENSE: dv, SPARSE: sv.as_object()},
                payload={"doc": "bench"},
            ))
            pid += 1
            if len(batch) >= 2000:
                client.upsert(collection_name=name, points=batch)
                batch = []
    if batch:
        client.upsert(collection_name=name, points=batch)

    info = client.get_collection(name)
    print(f"{name}: points={info.points_count} dim={dim} copies={args.copies}")
    print(f"theoretical float32 MB: {total * dim * 4 / (1024**2):.2f}")
    print(f"theoretical binary  MB: {total * dim / 8 / (1024**2):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
