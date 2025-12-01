"""
Configuration file for the AI Chatbot
Customize these settings based on your needs
"""

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}

# Model Configuration
MODELS = {
    # Semantic search model - change to a smaller/larger model as needed
    # Options: 
    #   - "all-MiniLM-L6-v2" (default, 384 dim, fast)
    #   - "all-mpnet-base-v2" (768 dim, better quality, slower)
    #   - "paraphrase-MiniLM-L3-v2" (smaller, faster, less accurate)
    "semantic_model": "all-MiniLM-L6-v2",
    
    # QA model - change to a smaller/larger model as needed
    # Options:
    #   - "deepset/roberta-base-squad2" (default)
    #   - "distilbert-base-cased-distilled-squad" (smaller, faster)
    #   - "deepset/bert-large-uncased-whole-word-masking-squad2" (larger, better)
    "qa_model": "deepset/roberta-base-squad2"
}

# Source Configuration
SOURCES = {
    # Default sources to use if none specified
    "default": ["wikipedia", "stackoverflow"],
    
    # Available sources
    "available": [
        "wikipedia",      # General knowledge
        "arxiv",         # Scientific papers
        "pubmed",        # Medical information
        "stackoverflow", # Technical Q&A
        "openlibrary",  # Book information
        "osm"           # Geographic data
    ],
    
    # Rate limiting (seconds between requests)
    "rate_limits": {
        "wikipedia": 0.1,
        "arxiv": 1.0,
        "pubmed": 0.5,
        "stackoverflow": 0.2,
        "openlibrary": 0.5,
        "osm": 1.0  # OSM requires 1 second between requests
    }
}

# Search Configuration
SEARCH_CONFIG = {
    # Number of documents to retrieve for QA
    "top_k": 3,
    
    # Minimum similarity threshold (0.0 to 1.0)
    "min_similarity": 0.3,
    
    # Minimum QA confidence score (0.0 to 1.0)
    "min_qa_score": 0.01,
    
    # Maximum answer length (characters)
    "max_answer_length": 200,
    
    # Chunk size for long documents (characters)
    "chunk_size": 500,
    
    # Overlap between chunks (characters)
    "chunk_overlap": 50
}

# Knowledge Base Configuration
KNOWLEDGE_BASE = {
    # Topics to preload on startup
    "preload_topics": [
        "Python programming language",
        "Machine learning basics",
        "Artificial intelligence",
        "Web development",
        "Database management",
        "Computer science fundamentals",
        "Software engineering best practices"
    ],
    
    # Whether to preload on startup (can be slow)
    "auto_preload": False,
    
    # Sources to use for preloading
    "preload_sources": ["wikipedia", "stackoverflow"]
}

# File Paths
FILES = {
    "memory": "memory.json",
    "training_data": "training_data.json",
    "unanswered": "unanswered.json",
    "documents": "semantic_documents.json",
    "embeddings": "semantic_embeddings.npy"
}

# Behavior Configuration
BEHAVIOR = {
    # Whether to fetch new data for each query
    "fetch_new_data": True,
    
    # Whether to add unanswered questions to the list
    "track_unanswered": True,
    
    # Whether to save conversations to memory
    "save_conversations": True,
    
    # Maximum memory entries to keep
    "max_memory_entries": 1000
}

# Response Configuration
RESPONSES = {
    # Message when no answer is found
    "no_answer": "I could not find reliable information on this topic in my sources.",
    
    # Message when answer is found but confidence is low
    "low_confidence": "I found some information, but I'm not very confident about this answer:",
    
    # Greeting responses
    "greetings": [
        "Hey there 👋",
        "Hello!",
        "Hi, how's it going?"
    ],
    
    # Goodbye responses
    "goodbyes": [
        "Goodbye! 👋",
        "See you soon!",
        "Take care!"
    ]
}

# Logging Configuration
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Performance Configuration
PERFORMANCE = {
    # Whether to use GPU if available
    "use_gpu": True,
    
    # Maximum threads for background tasks
    "max_threads": 4,
    
    # Cache size for embeddings (MB)
    "cache_size": 500
}


def get_config(section: str, key: str = None):
    """
    Get configuration value
    
    Args:
        section: Configuration section (e.g., "MODELS", "SEARCH_CONFIG")
        key: Optional key within the section
    
    Returns:
        Configuration value
    """
    config_sections = {
        "server": SERVER_CONFIG,
        "models": MODELS,
        "sources": SOURCES,
        "search": SEARCH_CONFIG,
        "knowledge_base": KNOWLEDGE_BASE,
        "files": FILES,
        "behavior": BEHAVIOR,
        "responses": RESPONSES,
        "logging": LOGGING,
        "performance": PERFORMANCE
    }
    
    section = section.lower()
    if section not in config_sections:
        raise ValueError(f"Unknown configuration section: {section}")
    
    if key is None:
        return config_sections[section]
    
    return config_sections[section].get(key)


if __name__ == "__main__":
    # Print all configurations
    import json
    
    print("AI Chatbot Configuration")
    print("=" * 60)
    
    print("\nServer Configuration:")
    print(json.dumps(SERVER_CONFIG, indent=2))
    
    print("\nModel Configuration:")
    print(json.dumps(MODELS, indent=2))
    
    print("\nSource Configuration:")
    print(json.dumps(SOURCES, indent=2))
    
    print("\nSearch Configuration:")
    print(json.dumps(SEARCH_CONFIG, indent=2))
    
    print("\nKnowledge Base Configuration:")
    print(json.dumps(KNOWLEDGE_BASE, indent=2))
