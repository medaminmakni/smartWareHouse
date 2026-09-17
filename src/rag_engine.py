import os
import re
import glob
from typing import List, Dict, Optional, Tuple

# chromadb is imported lazily inside WarehouseRAGEngine.__init__ so that the
# pure-python helpers in this module (notably _chunk_markdown) can be imported
# without pulling in the vector store — eval_retrieval.py needs the chunker,
# not the database.

# ---------------------------------------------------------------------------
# Embedding model
#
# all-MiniLM-L6-v2 is trained on English. This knowledge base is mixed:
# chatbot_rules.md is French, gate_pickup_rules.md and warehouse_logic.md are
# English, and the agent queries in French. Cross-language retrieval with an
# English-only model is unreliable, so the default is multilingual.
# Override with RAG_EMBED_MODEL to benchmark alternatives.
# ---------------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking targets, in characters.
TARGET_CHUNK = 550      # aim for chunks around this size
MAX_CHUNK = 900         # hard ceiling before forcing a split
MIN_CHUNK = 120         # below this, merge into the neighbour


def _chunk_markdown(content: str, target: int = TARGET_CHUNK,
                    max_size: int = MAX_CHUNK, min_size: int = MIN_CHUNK) -> List[str]:
    """
    Splits markdown into retrieval chunks.

    Splitting on "\\n\\n" alone produced wildly uneven chunks: warehouse_logic.md
    became 2 very large ones while chatbot_rules.md became 29 tiny ones, which
    skews similarity scoring towards the short fragments. This keeps chunks in a
    predictable size band and prefixes each one with its markdown heading path,
    so a chunk still carries its context ("Gate Pickup Management Rules >
    Vehicle Arrival & Pickup Confirmation Flow > ...") once it is detached from
    the document.
    """
    lines = content.split("\n")
    headings: Dict[int, str] = {}
    blocks: List[Tuple[str, str]] = []   # (heading_path, paragraph)
    buf: List[str] = []

    def heading_path() -> str:
        return " > ".join(headings[k] for k in sorted(headings) if headings.get(k))

    def flush_paragraph():
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append((heading_path(), text))
            buf.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            flush_paragraph()
            level = len(m.group(1))
            headings[level] = m.group(2).strip()
            for deeper in [k for k in headings if k > level]:
                headings.pop(deeper, None)
            continue
        if not line.strip():
            flush_paragraph()
        else:
            buf.append(line)
    flush_paragraph()

    # Group paragraphs into size-banded chunks, never mixing heading paths.
    chunks: List[str] = []
    cur_path: Optional[str] = None
    cur: List[str] = []

    def emit():
        if not cur:
            return
        body = "\n\n".join(cur).strip()
        if not body:
            cur.clear()
            return
        chunks.append(f"[{cur_path}]\n{body}" if cur_path else body)
        cur.clear()

    for path, para in blocks:
        if cur_path is not None and path != cur_path:
            emit()
        cur_path = path
        candidate = sum(len(p) for p in cur) + len(para)
        if cur and candidate > target:
            emit()
        if len(para) > max_size:
            # A single oversized paragraph: split it on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            piece = ""
            for s in sentences:
                if piece and len(piece) + len(s) > target:
                    cur.append(piece.strip())
                    emit()
                    piece = ""
                piece += (" " if piece else "") + s
            if piece.strip():
                cur.append(piece.strip())
        else:
            cur.append(para)
    emit()

    # Merge stragglers so no chunk is too small to carry meaning.
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_size:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


class WarehouseRAGEngine:
    def __init__(self, db_path: str = "./warehouse_db", embed_model: Optional[str] = None):
        import chromadb
        from chromadb.utils import embedding_functions

        self.embed_model_name = embed_model or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)

        self.client = chromadb.PersistentClient(path=db_path)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model_name
        )

        self.collection = self.client.get_or_create_collection(
            name="warehouse_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ load

    def load_documents(self, data_dir: str) -> int:
        """
        Indexes every markdown file under data_dir.

        Idempotent: each file's existing chunks are removed before its new ones
        are written, so re-running this never duplicates or leaves orphans
        behind when a document shrinks. The previous version used .add() with
        fixed ids, which meant a second run either errored or left stale chunks.
        """
        files = sorted(glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True))
        total = 0

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = _chunk_markdown(content)
            if not chunks:
                continue

            category = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)

            try:
                self.collection.delete(where={"source": filename})
            except Exception:
                pass

            self.collection.upsert(
                documents=chunks,
                metadatas=[{"category": category, "source": filename} for _ in chunks],
                ids=[f"{filename}_{i}" for i in range(len(chunks))],
            )
            total += len(chunks)
            print(f"[ok] {filename} ({category}): {len(chunks)} chunks")

        print(f"[ok] {total} chunks indexed with {self.embed_model_name}")
        return total

    # ----------------------------------------------------------------- query

    def query(self, query_text: str, n_results: int = 3,
              category: Optional[str] = None) -> List[str]:
        """Retrieves the most relevant chunks. Returns plain strings."""
        return [d for d, _, _ in self.query_detailed(query_text, n_results, category)]

    def query_detailed(self, query_text: str, n_results: int = 3,
                       category: Optional[str] = None) -> List[Tuple[str, Dict, float]]:
        """Same as query(), but returns (document, metadata, distance) triples."""
        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if category:
            kwargs["where"] = {"category": category}

        results = self.collection.query(**kwargs)
        if not results.get("documents") or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = (results.get("metadatas") or [[{}] * len(docs)])[0]
        dists = (results.get("distances") or [[0.0] * len(docs)])[0]
        return list(zip(docs, metas, dists))

    def query_multi(self, queries: List[str], n_results: int = 3,
                    category: Optional[str] = None) -> List[Tuple[str, Dict, float]]:
        """
        Runs several queries and merges the results, keeping each chunk's best
        (lowest) distance. Used to retrieve client-specific notes and general
        operational policy in one pass — a single query cannot do both, because
        the policy documents never mention a client by name.
        """
        best: Dict[str, Tuple[str, Dict, float]] = {}
        for q in queries:
            for doc, meta, dist in self.query_detailed(q, n_results, category):
                key = f"{meta.get('source','?')}::{doc[:80]}"
                if key not in best or dist < best[key][2]:
                    best[key] = (doc, meta, dist)
        return sorted(best.values(), key=lambda t: t[2])


# Module-level singleton.
# main.py was constructing WarehouseRAGEngine() inside the chat request handler,
# which reloaded the sentence-transformer model on every message.
_ENGINE: Optional[WarehouseRAGEngine] = None


def get_engine(db_path: str = "./warehouse_db") -> WarehouseRAGEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = WarehouseRAGEngine(db_path=db_path)
    return _ENGINE


if __name__ == "__main__":
    engine = WarehouseRAGEngine()
    engine.load_documents(os.path.join(os.getcwd(), "data"))
    print("Warehouse RAG knowledge base ready.")
