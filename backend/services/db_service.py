import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

log = logging.getLogger("db_service")

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "enterprise_docs"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        log.info("Qdrant client initialised")
    return _client


def get_collection_name() -> str:
    return COLLECTION_NAME


def collection_exists(client: QdrantClient) -> bool:
    try:
        client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False


def ensure_collection(embeddings) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    client = get_client()
    if collection_exists(client):
        return
    sample_vec = embeddings.embed_query("test")
    vector_size = len(sample_vec)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info(f"Collection '{COLLECTION_NAME}' created — vector size {vector_size}")
