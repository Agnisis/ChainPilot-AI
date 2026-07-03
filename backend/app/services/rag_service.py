import re
import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions

from app.config import settings
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    logger.warning("sentence-transformers not installed, reranking disabled.")


class RagService:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        
        # Use sentence-transformers instead of default Chroma
        try:
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}. Falling back to default.")
            self._ef = embedding_functions.DefaultEmbeddingFunction()
            
        if HAS_RERANKER and settings.reranking_enabled:
            try:
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self._reranker = None
        else:
            self._reranker = None

    def _collection_name(self, session_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return f"session_{safe}"

    def _get_or_create_collection(self, session_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(session_id),
            embedding_function=self._ef
        )

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix in {".txt", ".md", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")

    def _chunk_text(self, text: str, chunk_size: int = settings.rag_chunk_size, overlap: int = settings.rag_chunk_overlap) -> list[str]:
        # Improved chunking with paragraph awareness
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []
            
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            # Rough token estimate (words * 1.3)
            p_len = len(p.split()) * 1.3
            curr_len = len(current_chunk.split()) * 1.3
            
            if curr_len + p_len <= chunk_size:
                current_chunk += "\n\n" + p if current_chunk else p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single paragraph is larger than chunk size, brute force split it
                if p_len > chunk_size:
                    words = p.split()
                    word_chunk_size = int(chunk_size / 1.3)
                    for i in range(0, len(words), word_chunk_size - overlap):
                        chunks.append(" ".join(words[i:i + word_chunk_size]))
                    current_chunk = ""
                else:
                    current_chunk = p
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def ingest_file(self, session_id: str, file_path: Path) -> int:
        text = self._extract_text(file_path)
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
            
        collection = self._get_or_create_collection(session_id)
        ids = [f"{file_path.stem}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
        
        # Add in batches to avoid size limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metas = [{"source": file_path.name, "session_id": session_id}] * len(batch_chunks)
            
            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_metas,
            )
            
        return len(chunks)

    def query(self, session_id: str, query: str, top_k: int = 5) -> tuple[list[str], list[str]]:
        try:
            collection = self._client.get_collection(
                name=self._collection_name(session_id),
                embedding_function=self._ef
            )
        except Exception:
            return [], []
            
        if collection.count() == 0:
            return [], []
            
        # Fetch more candidates if reranking is enabled
        fetch_k = top_k * 3 if self._reranker else top_k
        fetch_k = min(fetch_k, collection.count())
        
        results = collection.query(query_texts=[query], n_results=fetch_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        if not docs:
            return [], []
            
        # Reranking Phase
        if self._reranker and len(docs) > 1:
            try:
                pairs = [[query, doc] for doc in docs]
                scores = self._reranker.predict(pairs)
                
                # Sort docs by reranker score
                doc_score_pairs = list(zip(docs, metas, scores))
                doc_score_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Take top_k
                docs = [p[0] for p in doc_score_pairs[:top_k]]
                metas = [p[1] for p in doc_score_pairs[:top_k]]
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Using original order.")
                docs = docs[:top_k]
                metas = metas[:top_k]
                
        sources = list({m.get("source", "unknown") for m in metas})
        return docs, sources

    def has_documents(self, session_id: str) -> bool:
        try:
            collection = self._client.get_collection(name=self._collection_name(session_id))
            return collection.count() > 0
        except Exception:
            return False


rag_service = RagService()
