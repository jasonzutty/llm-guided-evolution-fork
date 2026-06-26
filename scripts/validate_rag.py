#!/usr/bin/env python3
"""Validate RAG setup and basic functionality."""

from pathlib import Path
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfg.constants import RAG_DATA_DIR, RAG_ENABLED
from src.rag.vector_db import VectorStoreManager
from src.rag.embeddings import EmbeddingService, EmbeddingConfig
from src.rag.retrieval import RetrievalService

def main():
    print("[Validation] Checking RAG configuration...")
    print(f"  RAG_ENABLED: {RAG_ENABLED}")
    print(f"  RAG_DATA_DIR: {RAG_DATA_DIR}")
    
    if not RAG_ENABLED:
        print("[Validation] RAG is disabled. Skipping validation.")
        return
    
    rag_path = Path(RAG_DATA_DIR)
    if not rag_path.exists():
        print(f"[Validation] ERROR: RAG data directory does not exist: {RAG_DATA_DIR}")
        return
    
    print("[Validation] Initializing RAG components...")
    try:
        store = VectorStoreManager(RAG_DATA_DIR)
        embeddings = EmbeddingService(EmbeddingConfig())
        retrieval = RetrievalService(store, embeddings)
        print("[Validation] ✓ RAG components initialized successfully")
    except Exception as e:
        print(f"[Validation] ERROR initializing RAG: {e}")
        return
    
    # Check indexed documents
    code_docs = store.list_documents(VectorStoreManager.CODE_NAMESPACE)
    text_docs = store.list_documents(VectorStoreManager.TEXT_NAMESPACE)
    print(f"[Validation] Indexed documents:")
    print(f"  Code namespace: {len(list(code_docs))} documents")
    print(f"  Text namespace: {len(list(text_docs))} documents")
    
    print("[Validation] ✓ RAG validation complete")

if __name__ == "__main__":
    main()