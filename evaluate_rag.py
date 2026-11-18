"""Диагностика индекса: размер коллекции, конфигурация квантизации, статус.

ЧТО ЭТОТ СКРИПТ НЕ ДЕЛАЕТ
-------------------------
Он не измеряет качество поиска. Recall@k и MRR считаются в
`eval/eval_retrieval.py` по размеченному эталону (CSV: запрос + релевантные doc_id).

ПОЧЕМУ ПРЕДЫДУЩАЯ ВЕРСИЯ БЫЛА НЕВЕРНОЙ
--------------------------------------
1. Она печатала «сжатие векторов в 32 раза (~96.8% экономии) благодаря
   Binary Quantization», хотя квантизация в `create_hybrid_collection` не была
   включена, а число считалось арифметически: float32 (4 байта) → binary
   (1 бит). Это оценка «если включить», а не измерение.
2. Её «recall» искал документ по его собственному вектору (self-retrieval):
   такой тест почти всегда даёт попадание и ничего не говорит о качестве.
3. Она печатала «ГОТОВЫЙ БЛОК ДЛЯ РЕЗЮМЕ» — то есть предлагала вставлять
   неизмеренные числа в документы.

Запуск: python evaluate_rag.py
"""

import sys

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hybrid_documents"
DENSE_DIM = 1024


def describe_quantization(collection_info) -> str:
    """Достаём фактическое состояние квантизации из конфигурации коллекции."""
    try:
        vectors = collection_info.config.params.vectors
        if isinstance(vectors, dict):
            params = next(iter(vectors.values()))
        else:
            params = vectors
        quant = getattr(params, "quantization_config", None)
        if quant is None:
            return "выключена (float32)"
        return f"включена: {type(quant).__name__}"
    except Exception as exc:  # конфигурация может отличаться между версиями
        return f"не удалось определить ({exc})"


def main() -> int:
    client = QdrantClient(url=QDRANT_URL, timeout=5)

    try:
        names = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        print(f"Qdrant недоступен по {QDRANT_URL}: {exc}")
        print("Поднимите сервис: docker compose up -d qdrant")
        return 2

    print(f"Коллекции: {names or '—'}")
    if COLLECTION_NAME not in names:
        print(f"Коллекции {COLLECTION_NAME!r} нет — индексировать пока нечего.")
        return 1

    info = client.get_collection(collection_name=COLLECTION_NAME)
    points = info.points_count or 0

    float32_mb = points * DENSE_DIM * 4 / (1024 * 1024)
    binary_mb = points * (DENSE_DIM / 8) / (1024 * 1024)

    print(f"\nКоллекция: {COLLECTION_NAME}")
    print(f"Точек (чанков): {points}")
    print(f"Квантизация: {describe_quantization(info)}")
    print("\nТеоретический размер плотных векторов:")
    print(f"  float32:            ~{float32_mb:.4f} MB")
    print(f"  binary (если вкл.): ~{binary_mb:.4f} MB")
    print("  Это расчёт по формуле, а НЕ замер RAM. Фактическое потребление")
    print("  смотрите в `docker stats` до и после включения квантизации.")

    print("\nКачество поиска здесь не измеряется.")
    print("Метрики: python eval/eval_retrieval.py --corpus <путь> --queries eval/eval_queries.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
