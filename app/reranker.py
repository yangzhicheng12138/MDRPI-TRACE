"""BGE cross-encoder reranking for retrieval candidates."""

from __future__ import annotations

import logging

from .retriever import SearchResult


class Reranker:
    """Lazy CrossEncoder wrapper for BGE reranker v2 M3."""

    def __init__(self, config, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = str(config.get("reranker", "model", default="BAAI/bge-reranker-v2-m3"))
        self.device = self._resolve_device(str(config.get("reranker", "device", default="auto")))
        self.cache_dir = config.path("embedding", "model_cache_dir", default="data/models")
        self.batch_size = int(config.get("reranker", "batch_size", default=8))
        self._model = None
        self._query_tokenizer = None

    def query_tokenizer(self):
        if self._model is not None:
            return self._model.tokenizer
        if self._query_tokenizer is None:
            from transformers import AutoTokenizer
            self._query_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=str(self.cache_dir),
                revision=self.config.get("reranker", "revision"),
                local_files_only=bool(self.config.get("reranker", "local_files_only", default=False)))
        return self._query_tokenizer

    def query_token_budget(self, passage_texts):
        """Reserve the longest admitted passage, plus pair special tokens."""
        from .query_policy import query_token_count
        tokenizer = self.query_tokenizer()
        maximum = int(tokenizer.model_max_length)
        passage_tokens = max((query_token_count(tokenizer, text, add_special_tokens=False)
                              for text in passage_texts), default=0)
        # HF encode('', '') treats an empty second string as no pair in some
        # releases. The tokenizer's explicit pair API gives the actual4tokens.
        pair_special = tokenizer.num_special_tokens_to_add(True)
        single_special = tokenizer.num_special_tokens_to_add(False)
        runtime_cap = int(self.config.get("rag", "retrieval_query_token_budget", default=2048))
        return {"token_budget": min(runtime_cap, maximum - passage_tokens - pair_special + single_special),
                "max_pair_tokens": maximum, "reserved_passage_tokens": passage_tokens,
                "pair_special_tokens": pair_special, "single_special_tokens": single_special,
                "runtime_query_token_cap": runtime_cap}

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
        from .embedding import EmbeddingService
        EmbeddingService._prepare_torchvision_compat()
        from sentence_transformers import CrossEncoder
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Loading reranker model=%s device=%s cache=%s", self.model_name, self.device, self.cache_dir)
        self._model = CrossEncoder(
            self.model_name, device=self.device, cache_folder=str(self.cache_dir),
            revision=self.config.get("reranker", "revision"),
            local_files_only=bool(self.config.get("reranker", "local_files_only", default=False)),
        )
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
        """Score query/candidate pairs and return the highest reranker scores."""

        if not candidates:
            return []
        import torch
        scores = self._load().predict(
            [(query, candidate.text) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
            activation_fn=torch.nn.Sigmoid(),
        )
        ranked = sorted(zip(candidates, [float(score) for score in scores]), key=lambda pair: pair[1], reverse=True)
        limit = int(top_k or self.config.get("reranker", "top_k", default=8))
        return [
            SearchResult(**{**item.to_dict(), "rank": rank, "reranker_score": score})
            for rank, (item, score) in enumerate(ranked[:limit], start=1)
        ]
