"""Small Qdrant adapter used by the development vector database."""

from __future__ import annotations

import logging
import uuid
from typing import Any


class VectorStore:
    """Qdrant client wrapper supporting local storage and HTTP server modes."""

    def __init__(self, config, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.collection = config.qdrant_collection
        self.mode = str(config.get("qdrant", "mode", default="local")).lower()
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for vector storage") from exc
        if self.mode in {"http", "server", "remote"}:
            self.client = QdrantClient(url=str(config.get("qdrant", "url", default="http://127.0.0.1:6333")))
        else:
            config.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(config.qdrant_path))

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Return a stable UUID accepted by Qdrant for a chunk identifier."""

        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mdrpi-kb:{chunk_id}"))

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection)
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
            )
            self.logger.info("Created Qdrant collection=%s vector_size=%s", self.collection, vector_size)
        else:
            info = self.client.get_collection(self.collection)
            configured = getattr(info.config.params.vectors, "size", None)
            if configured is not None and int(configured) != int(vector_size):
                raise ValueError(f"Collection {self.collection} has vector size {configured}, expected {vector_size}")

    def upsert(self, vectors: list[list[float]], payloads: list[dict[str, Any]], chunk_ids: list[str]) -> None:
        from qdrant_client.models import PointStruct
        if not (len(vectors) == len(payloads) == len(chunk_ids)):
            raise ValueError("vectors, payloads and chunk_ids must have equal lengths")
        points = [
            PointStruct(id=self.point_id(chunk_id), vector=vector, payload=payload)
            for vector, payload, chunk_id in zip(vectors, payloads, chunk_ids)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def iter_points(self, with_vectors: bool = False):
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, offset=offset, limit=256,
                with_payload=True, with_vectors=with_vectors,
            )
            yield from points
            if offset is None:
                break

    def close(self) -> None:
        """Close the underlying client when the adapter is no longer needed."""

        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def search(self, vector: list[float], limit: int = 10, query_filter=None) -> list[Any]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=query_filter,
                limit=int(limit),
                with_payload=True,
                with_vectors=False,
            )
            return list(response.points)
        return list(self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=int(limit),
            with_payload=True,
        ))
