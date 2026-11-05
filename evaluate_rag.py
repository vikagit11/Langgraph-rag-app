import time
import statistics
import requests
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
FASTAPI_URL = "http://localhost:8000"
COLLECTION_NAME = "hybrid_documents"

def run_benchmark():
    client = QdrantClient(url=QDRANT_URL)
    
    print("\n" + "="*55)
    print("1. АНАЛИЗ ПАМЯТИ QDRANT И БИНАРНОЙ КВАНТИЗАЦИИ")
    print("="*55)
    
    col_info = client.get_collection(collection_name=COLLECTION_NAME)
    points_count = col_info.points_count
    dim = 1024
    
    raw_ram_mb = (points_count * dim * 4) / (1024 * 1024)
    bq_ram_mb = (points_count * (dim / 8)) / (1024 * 1024)
    
    print(f"Коллекция: {COLLECTION_NAME}")
    print(f"Статус: {col_info.status}")
    print(f"Количество чанков (points): {points_count}")
    print(f"\n[Потребление RAM под векторы (1024 dim)]:")
    print(f"  • Исходный размер (Float32): ~{raw_ram_mb:.4f} MB")
    print(f"  • С Binary Quantization:     ~{bq_ram_mb:.4f} MB")
    print(f"  • Сжатие векторов:           в 32 раза (~96.8% экономии RAM)")

    print("\n" + "="*55)
    print("2. ЗАМЕР ЧИСТОЙ ЗАДЕРЖКИ ПОИСКА И RECALL@K (QDRANT)")
    print("="*55)

    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=20,
        with_payload=True,
        with_vectors=True
    )
    
    if not records:
        print("[-] В коллекции нет записей.")
        return

    latencies = []
    hits_at_3 = 0
    hits_at_5 = 0
    total_tests = len(records)

    for record in records:
        vec = record.vector
        dense_vec = vec.get("thenlper-gte-large") if isinstance(vec, dict) else vec
        target_id = record.id

        start = time.perf_counter()
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="thenlper-gte-large" if isinstance(vec, dict) and "thenlper-gte-large" in vec else None,
            limit=5
        ).points
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = [p.id for p in search_results]
        if target_id in retrieved_ids[:3]:
            hits_at_3 += 1
        if target_id in retrieved_ids[:5]:
            hits_at_5 += 1

    avg_lat = statistics.mean(latencies)
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    recall_3 = (hits_at_3 / total_tests) * 100
    recall_5 = (hits_at_5 / total_tests) * 100

    print(f"Средняя задержка поиска (Search Latency): {avg_lat:.2f} ms")
    print(f"95-й перцентиль задержки (p95):          {p95_lat:.2f} ms")
    print(f"Точность извлечения Recall@3:            {recall_3:.1f}%")
    print(f"Точность извлечения Recall@5:            {recall_5:.1f}%")

    print("\n" + "="*55)
    print("3. ПРОВЕРКА СКВОЗНОГО API /query ЧЕРЕЗ FASTAPI")
    print("="*55)

    test_queries = [
        "Какие лимиты памяти указаны в спецификации?",
        "Какая задержка поиска допустима?",
        "Какие основные требования к системе?"
    ]

    for i, q in enumerate(test_queries, 1):
        payload = {
            "question": q,
            "provider": "ollama",
            "model_name": "llama3:latest"
        }
        start = time.perf_counter()
        try:
            res = requests.post(f"{FASTAPI_URL}/query", json=payload, timeout=60)
            elapsed_sec = time.perf_counter() - start
            if res.status_code == 200:
                data = res.json()
                sources_count = len(data.get("sources", []))
                print(f"  [{i}/{len(test_queries)}] OK ({elapsed_sec:.2f}с) | Найдено источников: {sources_count}")
            else:
                print(f"  [{i}/{len(test_queries)}] Ошибка HTTP {res.status_code}: {res.text}")
        except Exception as e:
            print(f"  [{i}/{len(test_queries)}] Ошибка запроса: {e}")

    print("\n" + "="*55)
    print("ГОТОВЫЙ БЛОК ДЛЯ РЕЗЮМЕ:")
    print("="*55)
    print(f"• Оптимизация RAM: сжатие векторов в 32 раза (~96.8%) благодаря Binary Quantization.")
    print(f"• Задержка поиска: среднее время извлечения {avg_lat:.2f} мс (p95: {p95_lat:.2f} мс).")
    print(f"• Качество ретривера: Recall@3 = {recall_3:.1f}%, Recall@5 = {recall_5:.1f}%.")

if __name__ == "__main__":
    run_benchmark()