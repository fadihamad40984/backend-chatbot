"""
AI Engine V3 - Semantic Search with Free and Open Sources
Uses sentence transformers for embeddings and QA models for answer extraction
Fetches data from Wikipedia, arXiv, PubMed, Stack Exchange, OpenLibrary, and OpenStreetMap
"""

import os
import json
from typing import Dict, Optional, List
import threading

# Import our new modules
from semantic_engine import SemanticSearchEngine, KnowledgeBaseBuilder
from qa_model import QAModel, SmartQAHandler
from source_fetchers import SourceAggregator

# File paths
MEMORY_FILE = "memory.json"
TRAINING_DATA_FILE = "training_data.json"
UNANSWERED_FILE = "unanswered.json"


def _ensure_file(path, default):
    """Ensure a file exists with default content"""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f)


def load_json(path) -> list:
    """Load JSON file"""
    _ensure_file(path, [])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_json(path, data):
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Memory and conversation management
def load_memory() -> List[Dict]:
    """Load conversation memory"""
    return load_json(MEMORY_FILE)


def save_memory(memory: List[Dict]):
    """Save conversation memory"""
    save_json(MEMORY_FILE, memory)


def add_conversation(user_input: str, bot_output: str):
    """Add a conversation to memory"""
    mem = load_memory()
    mem.append({"input": user_input, "output": bot_output})
    save_memory(mem)


def load_training_data() -> List[Dict]:
    """Load training data"""
    return load_json(TRAINING_DATA_FILE)


def add_training_pair(q: str, a: str):
    """Add a training Q&A pair"""
    data = load_training_data()
    data.append({"input": q, "output": a})
    save_json(TRAINING_DATA_FILE, data)


def add_unanswered(question: str):
    """Add an unanswered question to the list"""
    data = load_json(UNANSWERED_FILE)
    data.append({"input": question})
    save_json(UNANSWERED_FILE, data)


def get_unanswered():
    """Get list of unanswered questions"""
    return load_json(UNANSWERED_FILE)


# Global QA handler (initialized lazily)
_qa_handler: Optional[SmartQAHandler] = None
_init_lock = threading.Lock()


def get_qa_handler() -> SmartQAHandler:
    """Get or initialize the QA handler (singleton pattern)"""
    global _qa_handler
    
    if _qa_handler is None:
        with _init_lock:
            if _qa_handler is None:  # Double-check pattern
                print("Initializing AI engine...")
                
                # Initialize semantic search engine
                semantic_engine = SemanticSearchEngine(model_name="all-MiniLM-L6-v2")
                
                # Initialize QA model
                qa_model = QAModel(model_name="deepset/roberta-base-squad2")
                
                # Create handler
                _qa_handler = SmartQAHandler(semantic_engine, qa_model)
                
                print("AI engine initialized successfully")
    
    return _qa_handler


def smart_reply(
    user_input: str,
    fetch_new_data: bool = False,  # Changed default to False for better performance
    sources: List[str] = None,
    threshold: float = 0.1  # Lowered threshold for better recall
) -> tuple:
    """
    Generate a smart reply using semantic search and QA model
    
    Args:
        user_input: User's question
        fetch_new_data: Whether to fetch fresh data from sources
        sources: List of sources to use (default: ['wikipedia', 'stackoverflow'])
        threshold: Minimum confidence threshold
    
    Returns:
        (answer, score, sources_used)
    """
    if not user_input or not user_input.strip():
        return None, 0.0, []
    
    try:
        # Get QA handler
        handler = get_qa_handler()
        
        # Get answer - first try with existing knowledge base
        result = handler.answer(
            user_input,
            fetch_new_data=False,  # Try existing data first
            sources=sources or ['wikipedia', 'stackoverflow']
        )
        
        # If no good answer and fetch_new_data is True, try fetching fresh data
        if (not result['answer'] or result['score'] < threshold) and fetch_new_data:
            print(f"No answer found in knowledge base. Fetching new data...")
            result = handler.answer(
                user_input,
                fetch_new_data=True,
                sources=sources or ['wikipedia', 'stackoverflow']
            )
        
        # Check if we have a good answer
        if result['answer'] and result['score'] >= threshold:
            # Format response with sources
            response = handler.format_response(result)
            return response, result['score'], result['sources']
        
        # If no answer or low confidence, return None
        return None, result['score'], result['sources']
    
    except Exception as e:
        print(f"Error in smart_reply: {e}")
        return None, 0.0, []


def train_and_persist(vectorizer_params=None):
    """
    Train/build the knowledge base from training data and online sources
    
    Args:
        vectorizer_params: Ignored (kept for compatibility)
    
    Returns:
        (number_of_documents, metric)
    """
    try:
        print("Building knowledge base...")
        
        # Get QA handler (initializes if needed)
        handler = get_qa_handler()
        
        # Load custom training data and add to knowledge base
        training_data = load_training_data()
        if training_data:
            print(f"Adding {len(training_data)} custom Q&A pairs...")
            documents = []
            for item in training_data:
                documents.append({
                    'title': item.get('input', ''),
                    'text': item.get('output', ''),
                    'source': 'Custom Training Data',
                    'url': ''
                })
            
            handler.semantic_engine.add_documents(documents, chunk=False)
        
        # Preload general knowledge (only if enabled in config)
        from config import KNOWLEDGE_BASE
        if KNOWLEDGE_BASE.get('auto_preload', False):
            builder = KnowledgeBaseBuilder(handler.semantic_engine)
            builder.preload_general_knowledge()
        else:
            print("Auto-preload disabled. Use /admin/fetch_sources to add topics.")
        
        # Save the knowledge base
        handler.semantic_engine.save()
        
        stats = handler.semantic_engine.get_stats()
        print(f"Knowledge base built: {stats['total_documents']} documents")
        
        return stats['total_documents'], float(stats.get('embedding_dimension', 0))
    
    except Exception as e:
        print(f"Error in train_and_persist: {e}")
        return 0, 0.0


_retrain_lock = threading.Lock()


def retrain_background(vectorizer_params=None):
    """
    Retrain the model in background
    
    Args:
        vectorizer_params: Ignored (kept for compatibility)
    
    Returns:
        True if retraining started, False if already running
    """
    if _retrain_lock.locked():
        return False
    
    def _job():
        with _retrain_lock:
            train_and_persist(vectorizer_params)
    
    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return True


# Backward compatibility function
def smart_reply_tfidf(user_input: str, top_k=1, threshold=0.35):
    """
    Backward compatibility wrapper for smart_reply
    
    Args:
        user_input: User's question
        top_k: Ignored (kept for compatibility)
        threshold: Minimum confidence threshold
    
    Returns:
        (answer, score, None)
    """
    answer, score, sources = smart_reply(user_input, fetch_new_data=True, threshold=threshold)
    return answer, score, None