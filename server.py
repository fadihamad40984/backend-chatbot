# server.py - Flask API for AI Chatbot with Semantic Search and Free Sources
from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_engine_v2 import (
    smart_reply_tfidf, 
    add_conversation, 
    train_and_persist, 
    retrain_background, 
    load_training_data, 
    add_training_pair,
    get_qa_handler,
    load_json,
    save_json,
    UNANSWERED_FILE,
    add_unanswered,
    get_unanswered
)
import random
import datetime
import os

app = Flask(__name__)

# Configure CORS to allow all origins (for development and production)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Ensure required files exist
for f in ("memory.json", "training_data.json", "unanswered.json"):
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("[]")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint - handles user messages
    Fetches data from free sources and uses semantic search + QA model
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()
    
    # Optional: specify which sources to use
    sources = data.get("sources", None)  # Can be ['wikipedia', 'stackoverflow', 'arxiv', etc.]
    fetch_new = data.get("fetch_new", True)  # Whether to fetch fresh data

    if not user_message:
        return jsonify({"reply": "I didn't receive anything 🤖"}), 200

    # Try to get answer using semantic search and QA model
    reply, score, source_list = smart_reply_tfidf(
        user_message, 
        threshold=0.1  # Lowered threshold for better recall
    )
    
    if reply:
        add_conversation(user_message, reply)
        return jsonify({
            "reply": reply, 
            "score": score,
            "sources": source_list
        }), 200

    # If no answer found, add to unanswered and use fallback
    add_unanswered(user_message)

    reply = basic_ai(user_message)
    add_conversation(user_message, reply)
    return jsonify({
        "reply": reply, 
        "score": 0.0,
        "sources": []
    }), 200


@app.route("/train", methods=["POST"])
def train_route():
    """
    Trigger knowledge base building
    Builds semantic embeddings from training data and preloads general knowledge
    """
    params = request.json or {}
    n_docs, metric = train_and_persist(vectorizer_params=params.get("vectorizer"))
    return jsonify({
        "status": "trained", 
        "n_docs": n_docs, 
        "metric": metric,
        "message": "Knowledge base built successfully with semantic search"
    })


@app.route("/admin/add", methods=["POST"])
def admin_add():
    """
    Add a new Q&A pair to training data and retrain
    """
    body = request.json or {}
    q = body.get("input") or body.get("question")
    a = body.get("output") or body.get("answer")

    if not q or not a:
        return jsonify({"error": "Provide 'input' and 'output' in body"}), 400

    add_training_pair(q, a)

    # Remove from unanswered if it exists
    unanswered = load_json(UNANSWERED_FILE)
    new_unanswered = [item for item in unanswered if item.get("input") != q]
    save_json(UNANSWERED_FILE, new_unanswered)

    # Retrain in background
    retrain_background()

    return jsonify({
        "status": "added and training started",
        "question": q,
        "answer": a
    }), 201


@app.route("/admin/fetch_sources", methods=["POST"])
def fetch_sources():
    """
    Manually trigger fetching data from external sources
    """
    body = request.json or {}
    query = body.get("query")
    sources = body.get("sources", ["wikipedia", "stackoverflow"])
    
    if not query:
        return jsonify({"error": "Provide 'query' in body"}), 400
    
    try:
        from source_fetchers import SourceAggregator
        from semantic_engine import KnowledgeBaseBuilder
        
        # Get QA handler
        handler = get_qa_handler()
        
        # Fetch data
        aggregator = SourceAggregator()
        documents = aggregator.search_all(query, sources=sources)
        
        # Add to knowledge base
        if documents:
            handler.semantic_engine.add_documents(documents, chunk=True)
            handler.semantic_engine.save()
            
            return jsonify({
                "status": "success",
                "documents_added": len(documents),
                "query": query,
                "sources": sources
            }), 200
        else:
            return jsonify({
                "status": "no_data",
                "message": "No documents found for the query"
            }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/stats", methods=["GET"])
def get_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        handler = get_qa_handler()
        stats = handler.semantic_engine.get_stats()
        
        return jsonify({
            "knowledge_base": stats,
            "training_data": len(load_training_data())
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/unanswered", methods=["GET"])
def get_unanswered_questions():
    """
    Get list of unanswered questions
    """
    data = get_unanswered()
    return jsonify(data), 200


@app.route("/admin/delete", methods=["POST"])
def admin_delete():
    """
    Delete a training data entry by index
    """
    body = request.json or {}
    idx = body.get("index")
    if idx is None:
        return jsonify({"error": "Provide index"}), 400

    data = load_training_data()
    if idx < 0 or idx >= len(data):
        return jsonify({"error": "Invalid index"}), 400

    data.pop(idx)
    save_json("training_data.json", data)
    retrain_background()
    return jsonify({"status": "deleted"}), 200


@app.route("/training_data", methods=["GET"])
def get_training_data():
    """
    Get all training data
    """
    data = load_training_data()
    return jsonify({"data": data})


def basic_ai(text):
    """
    Basic fallback responses for simple queries
    """
    text = text.lower()
    greetings = ["hello", "hi", "hey"]
    feelings = ["how are you", "how do you feel"]
    bye = ["bye", "goodbye", "see you"]

    if any(word in text for word in greetings):
        return random.choice(["Hey there 👋", "Hello!", "Hi, how's it going?"])
    elif any(word in text for word in feelings):
        return random.choice(["I'm doing great, thanks!", "Feeling awesome 🤖"])
    elif any(word in text for word in bye):
        return random.choice(["Goodbye! 👋", "See you soon!", "Take care!"])
    elif "time" in text:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        return f"The current time is {now}"
    else:
        return "I'm still learning about that topic. Could you rephrase or try asking something else?"


@app.route("/")
def index():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "online",
        "message": "AI Chatbot API with Semantic Search",
        "version": "3.0",
        "sources": [
            "Wikipedia",
            "arXiv",
            "PubMed Central",
            "Stack Overflow",
            "OpenLibrary",
            "OpenStreetMap"
        ]
    })


if __name__ == "__main__":
    # Initialize and train the model on startup
    print("Starting AI Chatbot Server...")
    
    # Check if running on Render (production)
    is_production = os.environ.get("RENDER") == "true"
    
    if not is_production:
        print("Initializing knowledge base...")
        try:
            train_and_persist()
        except Exception as e:
            print(f"Error during initialization: {e}")
            print("Server will start anyway. Knowledge base can be built via /train endpoint.")
    else:
        print("Production mode: Lazy loading enabled (knowledge base will load on first request)")
    
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
