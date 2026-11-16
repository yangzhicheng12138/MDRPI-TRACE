"""BGE-M3 embedding service with configurable device and OOM fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path


class EmbeddingService:
    """Load and run the configured SentenceTransformer model on demand."""

    def __init__(self, config, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = str(config.get("embedding", "model", default="BAAI/bge-m3"))
        self.batch_size = int(config.get("embedding", "batch_size", default=16))
        self.normalize = bool(config.get("embedding", "normalize", default=True))
        self.max_seq_length = int(config.get("embedding", "max_seq_length", default=8192))
        self.cache_dir = config.path("embedding", "model_cache_dir", default="data/models")
        self.device = self._resolve_device(str(config.get("embedding", "device", default="auto")))
        self._model = None
        self._query_tokenizer = None

    def query_tokenizer(self):
        """Reuse loaded tokenizer, or load only cached CPU tokenizer metadata."""
        if self._model is not None:
            return self._model.tokenizer
        if self._query_tokenizer is None:
            from transformers import AutoTokenizer
            self._query_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir),
                revision=self.config.get("embedding", "revision"),
                local_files_only=bool(self.config.get("embedding", "local_files_only", default=False)))
        return self._query_tokenizer

    def prepare_query(self, query):
        # Document embedding and its persisted index identity are unchanged.
        from .query_policy import prepare_token_bounded_query
        budget = min(self.max_seq_length, int(self.config.get("rag", "retrieval_query_token_budget", default=2048)))
        return prepare_token_bounded_query(query, self.query_tokenizer(), budget)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model
        self._prepare_torchvision_compat()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for embedding") from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Loading embedding model=%s device=%s cache=%s", self.model_name, self.device, self.cache_dir)
        self._model = SentenceTransformer(
            self.model_name, device=self.device, cache_folder=str(self.cache_dir),
            revision=self.config.get("embedding", "revision"),
            local_files_only=bool(self.config.get("embedding", "local_files_only", default=False)),
        )
        self._model.max_seq_length = self.max_seq_length
        self.logger.info("Embedding model loaded; max_seq_length=%s", self.max_seq_length)
        return self._model

    @staticmethod
    def _prepare_torchvision_compat() -> None:
        """Ignore an optional torchvision fake-op mismatch for text models.

        Some shared CUDA installations ship torchvision without its compiled
        NMS operator. Transformers imports torchvision opportunistically even
        for text-only models; a missing optional operator must not prevent BGE
        from loading.
        """
        try:
            import torch

            # The shared CUDA environment keeps dependent libraries next to
            # torch rather than in the system linker path. Make them visible
            # before Transformers probes optional audio/vision backends.
            torch_root = Path(torch.__file__).resolve().parent
            library_paths = [torch_root / "lib"]
            library_paths.extend(torch_root.parent.glob("nvidia/*/lib"))
            current = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
            os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([str(path) for path in library_paths if path.exists()] + current))

            if getattr(torch, "_mdrpi_safe_register_fake", False):
                return
            original = torch.library.register_fake

            def safe_register_fake(*args, **kwargs):
                decorator = original(*args, **kwargs)

                def apply(function):
                    try:
                        return decorator(function)
                    except RuntimeError as exc:
                        if "does not exist" in str(exc):
                            return function
                        raise

                return apply

            torch.library.register_fake = safe_register_fake
            torch._mdrpi_safe_register_fake = True
        except Exception:
            return

    def _encode(self, texts: list[str], batch_size: int):
        return self._load().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed documents in batches, halving the batch on CUDA OOM."""

        if not texts:
            return []
        current_batch = max(1, int(batch_size or self.batch_size))
        while True:
            try:
                output = self._encode(list(texts), current_batch)
                self.logger.info("Embedded %s documents with batch_size=%s device=%s", len(texts), current_batch, self.device)
                return output.tolist()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                self.logger.warning("Embedding OOM; retrying with batch_size=%s", current_batch)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the same normalization as documents."""

        return self.embed_documents([text], batch_size=1)[0]


_SERVICE_CACHE: dict[tuple, EmbeddingService] = {}


def get_embedding_service(config, logger: logging.Logger | None = None) -> EmbeddingService:
    """Return one lazily loaded service per project configuration."""

    from .corpus import embedding_identity, json_digest
    key = (Path(config.project_root).resolve(), json_digest(embedding_identity(config)),
           str(config.get("embedding", "device", default="auto")),
           str(config.path("embedding", "model_cache_dir", default="data/models")),
           bool(config.get("embedding", "local_files_only", default=False)))
    if key not in _SERVICE_CACHE:
        _SERVICE_CACHE[key] = EmbeddingService(config, logger=logger)
    return _SERVICE_CACHE[key]


def embed_documents(texts: list[str], config=None, batch_size: int | None = None) -> list[list[float]]:
    """Convenience wrapper matching the stage-two embedding interface."""

    if config is None:
        raise ValueError("config is required; model name and cache location must come from config.yaml")
    return get_embedding_service(config).embed_documents(texts, batch_size=batch_size)


def embed_query(text: str, config=None) -> list[float]:
    """Convenience query wrapper matching the stage-two embedding interface."""

    if config is None:
        raise ValueError("config is required; model name and cache location must come from config.yaml")
    return get_embedding_service(config).embed_query(text)
