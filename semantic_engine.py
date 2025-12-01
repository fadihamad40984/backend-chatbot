"""
Semantic Search Engine using Sentence Transformers
Uses all-MiniLM-L6-v2 for embeddings and cosine similarity for retrieval
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class SemanticSearchEngine:
    """
    Semantic search engine using sentence embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search engine
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Storage paths
        self.docs_file = "semantic_documents.json"
        self.embeddings_file = "semantic_embeddings.npy"
        
        # Load existing data if available
        self.load()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_documents(self, documents: List[Dict], chunk: bool = True):
        """
        Add documents to the search index
        
        Args:
            documents: List of dicts with 'title', 'text', 'source', 'url'
            chunk: Whether to chunk long documents
        """
        new_docs = []
        
        for doc in documents:
            text = doc.get('text', '')
            title = doc.get('title', '')
            source = doc.get('source', '')
            url = doc.get('url', '')
            
            if not text:
                continue
            
            # Chunk long documents
            if chunk and len(text) > 500:
                chunks = self.chunk_text(text)
                for i, chunk_text in enumerate(chunks):
                    new_docs.append({
                        'title': f"{title} (Part {i+1})",
                        'text': chunk_text,
                        'source': source,
                        'url': url,
                        'full_text': text  # Keep reference to full text
                    })
            else:
                new_docs.append({
                    'title': title,
                    'text': text,
                    'source': source,
                    'url': url,
                    'full_text': text
                })
        
        if not new_docs:
            return
        
        # Generate embeddings for new documents
        texts = [doc['text'] for doc in new_docs]
        print(f"Generating embeddings for {len(texts)} document chunks...")
        new_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add to existing data
        self.documents.extend(new_docs)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        print(f"Total documents in index: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold
        
        Returns:
            List of dicts with 'title', 'text', 'source', 'url', 'score'
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by minimum similarity and prepare results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_similarity:
                doc = self.documents[idx].copy()
                doc['score'] = score
                results.append(doc)
        
        return results
    
    def get_context_for_qa(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Get combined context from top results for QA model
        
        Args:
            query: User question
            top_k: Number of top documents to combine
        
        Returns:
            (combined_context, source_documents)
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "", []
        
        # Combine texts from top results
        context_parts = []
        for i, doc in enumerate(results):
            context_parts.append(f"[Source {i+1}: {doc['source']}]\n{doc['text']}")
        
        combined_context = "\n\n".join(context_parts)
        
        return combined_context, results
    
    def save(self):
        """Save documents and embeddings to disk"""
        try:
            # Save documents as JSON
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save embeddings as numpy array
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            print(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            print(f"Error saving semantic search data: {e}")
    
    def load(self):
        """Load documents and embeddings from disk"""
        try:
            if os.path.exists(self.docs_file):
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"Loaded {len(self.documents)} documents from disk")
            
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)
                print(f"Loaded embeddings with shape {self.embeddings.shape}")
        except Exception as e:
            print(f"Error loading semantic search data: {e}")
            self.documents = []
            self.embeddings = None
    
    def clear(self):
        """Clear all documents and embeddings"""
        self.documents = []
        self.embeddings = None
        
        # Remove files
        for file in [self.docs_file, self.embeddings_file]:
            if os.path.exists(file):
                os.remove(file)
        
        print("Cleared semantic search index")
    
    def get_stats(self) -> Dict:
        """Get statistics about the search index"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_name": self.model.get_sentence_embedding_dimension()
        }


class KnowledgeBaseBuilder:
    """
    Build a knowledge base from various sources
    """
    
    def __init__(self, semantic_engine: SemanticSearchEngine):
        """
        Initialize knowledge base builder
        
        Args:
            semantic_engine: SemanticSearchEngine instance
        """
        self.engine = semantic_engine
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents into the semantic search engine
        
        Args:
            documents: List of document dicts from source fetchers
        """
        if not documents:
            print("No documents to index")
            return
        
        print(f"Indexing {len(documents)} documents...")
        self.engine.add_documents(documents, chunk=True)
        self.engine.save()
        print("Indexing complete")
    
    def build_from_sources(self, queries: List[str], sources: List[str] = None):
        """
        Build knowledge base by fetching from multiple sources
        
        Args:
            queries: List of queries to search for
            sources: List of source names to use
        """
        from source_fetchers import SourceAggregator
        
        aggregator = SourceAggregator()
        
        all_documents = []
        for query in queries:
            print(f"Fetching data for: {query}")
            docs = aggregator.search_all(query, sources=sources)
            all_documents.extend(docs)
            print(f"  Found {len(docs)} documents")
        
        print(f"\nTotal documents fetched: {len(all_documents)}")
        
        if all_documents:
            self.index_documents(all_documents)
    
    def preload_general_knowledge(self):
        """
        Preload general knowledge topics into the knowledge base
        """
        general_topics = [
            "Python programming language",
            "Machine learning basics",
            "Artificial intelligence",
            "Web development",
            "Database management",
            "Computer science fundamentals",
            "Software engineering best practices"
        ]
        
        print("Preloading general knowledge...")
        self.build_from_sources(general_topics, sources=['wikipedia', 'stackoverflow'])
