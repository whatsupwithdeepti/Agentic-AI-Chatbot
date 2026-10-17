"""Small, local FAISS memory using deterministic feature-hash embeddings.

This keeps ingestion private and avoids requiring a separate embedding API. The
embeddings are lexical (not a substitute for a hosted semantic embedding model).
"""
import hashlib
import re

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class LocalHashEmbeddings(Embeddings):
    dimensions = 512

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[\w+#.-]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_local_memory(files: list[tuple[str, str]]) -> FAISS | None:
    """Create an in-memory FAISS index from (filename, text) pairs."""
    documents = []
    for filename, text in files:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for start in range(0, len(paragraphs), 4):
            chunk = "\n\n".join(paragraphs[start : start + 4])
            if chunk:
                documents.append(Document(page_content=chunk, metadata={"source": filename}))
    if not documents:
        return None
    return FAISS.from_documents(documents, LocalHashEmbeddings())


def retrieve_local_context(store: FAISS | None, query: str, k: int = 4) -> str:
    if store is None:
        return ""
    matches = store.similarity_search(query, k=k)
    return "\n\n".join(
        f"[Local source: {doc.metadata.get('source', 'uploaded file')}]\n{doc.page_content}"
        for doc in matches
    )
