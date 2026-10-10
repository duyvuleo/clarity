"""Table-scoped BM25 + cosine search with weighted reciprocal-rank fusion."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading

import numpy as np
import sqlglot
from sqlglot import exp

from clarity.retrieval.config import RetrievalConfig


def tokenize(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


@dataclass(frozen=True)
class ColumnDocument:
    db_id: str
    table: str
    column: str
    data_type: str

    @property
    def text(self):
        return f"column {' '.join(tokenize(self.column))}; table {' '.join(tokenize(self.table))}; type {self.data_type}; identifier {self.column}"


class BM25:
    """Okapi BM25 with positive Lucene-style IDF and identifier-aware tokenization."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.array([sum(row.values()) for row in self.counts], dtype=float)
        self.average_length = float(self.lengths.mean()) if len(texts) else 0
        self.document_frequency = Counter(token for row in self.counts for token in row)
        self.k1, self.b = k1, b

    def scores(self, query):
        scores = np.zeros(len(self.counts), dtype=float)
        if not self.average_length:
            return scores
        denominator_base = self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
        for token in set(tokenize(query)):
            df = self.document_frequency.get(token, 0)
            if not df:
                continue
            idf = np.log(1 + (len(self.counts) - df + 0.5) / (df + 0.5))
            tf = np.array([row.get(token, 0) for row in self.counts], dtype=float)
            scores += idf * tf * (self.k1 + 1) / (tf + denominator_base)
        return scores


def normalize_vectors(values, rows=None):
    matrix = np.asarray(values, dtype=np.float32)
    if (
        matrix.ndim != 2
        or not matrix.shape[1]
        or (rows is not None and len(matrix) != rows)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Embedding response has invalid shape or non-finite values")
    lengths = np.linalg.norm(matrix, axis=1)
    if (lengths <= 0).any():
        raise ValueError("Embedding response contains zero-length vectors")
    return matrix / lengths[:, None]


class SchemaIndex:
    def __init__(self, schemas, columns, config: RetrievalConfig, embedding=None, mode="hybrid"):
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("Conventional retrieval mode must be hybrid, dense or sparse")
        if set(schemas) != set(columns):
            raise ValueError("Schema and column catalogs have different database IDs")
        self.schemas, self.columns, self.config = schemas, columns, config
        self.embedding, self.mode = embedding, mode
        self._vectors = {}
        self._metadata = {}
        self._lock = threading.RLock()
        self.last_index_metadata = {}

    def _documents(self, db_id, table):
        if db_id not in self.columns:
            raise ValueError("Unknown benchmark database ID")
        matches = [name for name in self.columns[db_id] if name.casefold() == table.casefold()]
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous table in the selected database")
        table = matches[0]
        ddl = self.schemas[db_id][table]
        types = {}
        try:
            parsed = sqlglot.parse_one(ddl, read="sqlite")
            types = {
                node.name.casefold(): node.args["kind"].sql(dialect="sqlite")
                for node in parsed.find_all(exp.ColumnDef)
                if node.args.get("kind")
            }
        except sqlglot.errors.ParseError:
            pass
        names = self.columns[db_id][table]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Column names are not unique within the selected table")
        return [
            ColumnDocument(db_id, table, name, types.get(name.casefold(), "unknown"))
            for name in names
        ], ddl

    def _dense_vectors(self, documents, ddl):
        if self.embedding is None:
            raise ValueError("Dense retrieval requires an embedding provider")
        identity = self.embedding.identity()
        metadata = {
            "format_version": 1,
            "embedding": identity,
            "documents": [document.__dict__ for document in documents],
            "schema_sha256": hashlib.sha256(ddl.encode()).hexdigest(),
        }
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        cache = Path(self.config.cache_dir) / (key + ".npz") if self.config.cache_dir else None
        with self._lock:
            if key not in self._vectors:
                if cache and cache.exists():
                    try:
                        with np.load(cache, allow_pickle=False) as stored:
                            actual = json.loads(str(stored["metadata"].item()))
                            vectors = normalize_vectors(stored["vectors"], len(documents))
                        if actual != metadata:
                            raise ValueError("metadata mismatch")
                    except Exception:
                        raise ValueError(
                            "Schema index cache is invalid; remove that cache or choose a new cache directory"
                        ) from None
                    loaded = True
                else:
                    vectors = normalize_vectors(
                        self.embedding.embed_documents([document.text for document in documents]),
                        len(documents),
                    )
                    loaded = False
                    if cache:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            dir=cache.parent, suffix=".npz", delete=False
                        ) as handle:
                            temporary = Path(handle.name)
                            np.savez_compressed(
                                handle,
                                vectors=vectors,
                                metadata=json.dumps(metadata, sort_keys=True),
                            )
                        temporary.replace(cache)
                self._vectors[key] = vectors
                self._metadata[key] = {
                    "fingerprint": key,
                    "embedding": identity,
                    "dimensions": vectors.shape[1],
                    "documents": len(documents),
                    "cache_loaded": loaded,
                }
            self.last_index_metadata = dict(self._metadata[key])
            return self._vectors[key]

    def search(self, query, db_id, table, top_k=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query must be nonempty text")
        documents, ddl = self._documents(db_id, table)
        if not documents:
            return []
        config = self.config
        alpha = 1.0 if self.mode == "dense" else 0.0 if self.mode == "sparse" else config.alpha
        sparse_scores = (
            BM25(
                [document.text for document in documents], config.sparse.k1, config.sparse.b
            ).scores(query)
            if alpha < 1
            else np.zeros(len(documents))
        )
        dense_scores = np.full(len(documents), np.nan)
        dense_order = []
        if alpha > 0:
            vectors = self._dense_vectors(documents, ddl)
            with self._lock:
                query_vector = normalize_vectors([self.embedding.embed_query(query)], 1)[0]
            if query_vector.shape[0] != vectors.shape[1]:
                raise ValueError(
                    "Query embedding dimensions differ from the indexed model; rebuild using consistent embedding settings"
                )
            dense_scores = vectors @ query_vector
            dense_order = sorted(
                (
                    index
                    for index in range(len(documents))
                    if config.min_dense_similarity is None
                    or dense_scores[index] >= config.min_dense_similarity
                ),
                key=lambda index: (-float(dense_scores[index]), documents[index].column.casefold()),
            )[: config.candidate_k]
        sparse_order = (
            sorted(
                (index for index, score in enumerate(sparse_scores) if score > 0),
                key=lambda index: (
                    -float(sparse_scores[index]),
                    documents[index].column.casefold(),
                ),
            )[: config.candidate_k]
            if alpha < 1
            else []
        )
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, 1)}
        sparse_ranks = {index: rank for rank, index in enumerate(sparse_order, 1)}
        hits = []
        for index in set(dense_ranks) | set(sparse_ranks):
            score = (alpha / (config.rrf_k + dense_ranks[index]) if index in dense_ranks else 0) + (
                (1 - alpha) / (config.rrf_k + sparse_ranks[index]) if index in sparse_ranks else 0
            )
            hits.append(
                {
                    **documents[index].__dict__,
                    "score": score,
                    "dense_score": float(dense_scores[index]) if index in dense_ranks else None,
                    "sparse_score": float(sparse_scores[index]) if index in sparse_ranks else None,
                    "dense_rank": dense_ranks.get(index),
                    "sparse_rank": sparse_ranks.get(index),
                }
            )
        return sorted(hits, key=lambda row: (-row["score"], row["column"].casefold()))[
            : top_k or config.top_k
        ]
