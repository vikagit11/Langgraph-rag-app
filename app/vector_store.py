from typing import List
import hashlib
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from fastapi import HTTPException

from .config import QDRANT_URL, COLLECTION_NAME

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embedding models
dense_embedding_model = TextEmbedding("thenlper/gte-large")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")

# Global variables
collection_exists = False

DENSE_VECTOR = "thenlper-gte-large"
SPARSE_VECTOR = "miniCOIL"


def stable_doc_id(source: str, chunk_text: str) -> str:
    """Стабильный id чанка.

    Зависит от источника и текста чанка, а не от случайного uuid4: при
    повторной индексации тот же чанк получает тот же id. Это нужно для
    разметки эталона и для сравнения прогонов между собой.
    """
    raw = f"{source}#{chunk_text[:200]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def create_hybrid_collection(quantize: bool = False, collection_name: str = COLLECTION_NAME):
    """Create a collection with hybrid vector configuration.

    Args:
        quantize: включить Binary Quantization для плотного вектора.
            Квантизация задаётся только при создании коллекции — для
            сравнения с ней нужно создавать отдельную коллекцию.
        collection_name: имя коллекции (по умолчанию из конфига).
    """
    global collection_exists

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if collection_name in collection_names:
            print(f"Collection {collection_name} already exists")
            collection_exists = True
            return True

        dense_params = models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
        )
        if quantize:
            dense_params.quantization_config = models.BinaryQuantizationConfig(always_ram=True)

        # Create collection with hybrid vectors
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE_VECTOR: dense_params},
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
            }
        )

        print(f"Created hybrid collection: {collection_name} (quantize={quantize})")
        collection_exists = True
        return True

    except Exception as e:
        print(f"Error creating collection: {e}")
        return False


def index_documents_hybrid(documents: List[Document], collection_name: str = COLLECTION_NAME):
    """Index documents with both dense and sparse embeddings"""
    if not collection_exists:
        if not create_hybrid_collection(collection_name=collection_name):
            raise HTTPException(status_code=500, detail="Failed to create collection")

    try:
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        dense_embeddings = list(dense_embedding_model.embed(texts))
        sparse_embeddings = list(sparse_embedding_model.embed(texts))

        # Create points for Qdrant
        points = []
        for idx, (dense_emb, sparse_emb, doc) in enumerate(zip(dense_embeddings, sparse_embeddings, documents)):
            point_id = str(uuid.uuid4())
            source = str(
                doc.metadata.get("source")
                or doc.metadata.get("file_name")
                or doc.metadata.get("title")
                or "document"
            )
            point = PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR: dense_emb,
                    SPARSE_VECTOR: sparse_emb.as_object(),
                },
                payload={
                    "document": doc.page_content,
                    "metadata": doc.metadata,
                    # Стабильные поля: по ним строится разметка эталона в eval
                    "doc_id": stable_doc_id(source, doc.page_content),
                    "source": source,
                    "chunk_index": idx,
                }
            )
            points.append(point)

        # Upsert to Qdrant
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Indexed {len(points)} documents with hybrid embeddings")
        return len(points)

    except Exception as e:
        print(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")


def hybrid_search(query: str, limit: int = 4, collection_name: str = COLLECTION_NAME) -> List[Document]:
    """Гибридный отбор кандидатов + переранжирование по dense.

    Историческое поведение проекта: prefetch по dense и sparse набирает
    кандидатов, затем финальный скор считается по dense-вектору. Это НЕ RRF —
    для слияния по рангам используйте search_hybrid_rrf.
    """
    if not collection_exists:
        raise HTTPException(status_code=503, detail="Collection not available")

    try:
        # Generate query embeddings
        dense_vector = next(dense_embedding_model.query_embed(query))
        sparse_vector = next(sparse_embedding_model.query_embed(query))

        # Create prefetch queries
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR,
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vector.as_object()),
                using=SPARSE_VECTOR,
                limit=20,
            )
        ]

        # Perform hybrid search with re-ranking
        results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=dense_vector,
            using=DENSE_VECTOR,
            with_payload=True,
            limit=limit,
        )
        # Convert to Document objects
        retrieved_docs = []
        for point in results.points:
            doc = Document(
                page_content=point.payload.get("document", ""),
                metadata={
                    **(point.payload.get("metadata") or {}),
                    "doc_id": point.payload.get("doc_id"),
                    "score": getattr(point, "score", None),
                }
            )
            retrieved_docs.append(doc)

        return retrieved_docs

    except Exception as e:
        print(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def search_hybrid_rrf(query: str, limit: int = 4, collection_name: str = COLLECTION_NAME,
                      prefetch_limit: int = 50) -> List[Document]:
    """Гибридный поиск со слиянием через Reciprocal Rank Fusion.

    Отличие от hybrid_search: в финальный query не передаётся вектор.
    Qdrant сливает два ранжированных списка (dense и sparse) по обратным
    рангам — это и есть RRF. Плюс: не нужно приводить скоры BM25 и косинус
    к одной шкале, они просто не сравниваются.
    """
    if not collection_exists:
        raise HTTPException(status_code=503, detail="Collection not available")

    try:
        dense_vector = next(dense_embedding_model.query_embed(query))
        sparse_vector = next(sparse_embedding_model.query_embed(query))

        results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=dense_vector, using=DENSE_VECTOR, limit=prefetch_limit),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_object()),
                    using=SPARSE_VECTOR,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )

        retrieved_docs = []
        for point in results.points:
            retrieved_docs.append(Document(
                page_content=point.payload.get("document", ""),
                metadata={
                    **(point.payload.get("metadata") or {}),
                    "doc_id": point.payload.get("doc_id"),
                    "score": getattr(point, "score", None),
                }
            ))
        return retrieved_docs

    except Exception as e:
        print(f"Error in hybrid RRF search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def clear_collection():
    """Clear all documents from the collection"""
    global collection_exists

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if COLLECTION_NAME not in collection_names:
            return {
                "message": "Collection does not exist",
                "status": "success"
            }

        # Delete the collection
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        collection_exists = False

        # Recreate the collection
        if create_hybrid_collection():
            return {
                "message": f"Successfully cleared collection '{COLLECTION_NAME}' and recreated it",
                "status": "success"
            }
        else:
            return {
                "message": f"Cleared collection '{COLLECTION_NAME}' but failed to recreate it",
                "status": "warning"
            }

    except Exception as e:
        print(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")


def get_collection_info():
    """Get collection information for health checks"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return {
            "collections": collection_names,
            "hybrid_collection_exists": COLLECTION_NAME in collection_names
        }
    except Exception as e:
        raise Exception(f"Failed to get collection info: {str(e)}")
