# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_engine_v2 import smart_reply_tfidf, add_conversation, train_and_persist, retrain_background, load_training_data, add_training_pair
import random
import datetime
import os

app = Flask(__name__)
CORS(app)

for f in ("memory.json", "training_data.json"):
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("[]")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower().strip()

    if not user_message:
        return jsonify({"reply": "I didn’t receive anything 🤖"}), 200

    reply, score, idx = smart_reply_tfidf(user_message, threshold=0.35)
    if reply:
        add_conversation(user_message, reply)
        return jsonify({"reply": reply, "score": score}), 200

    from ai_engine_v2 import add_unanswered
    add_unanswered(user_message)

    reply = basic_ai(user_message)
    add_conversation(user_message, reply)
    return jsonify({"reply": reply, "score": 0.0}), 200


@app.route("/train", methods=["POST"])
def train_route():
    params = request.json or {}
    n_docs, metric = train_and_persist(vectorizer_params=params.get("vectorizer"))
    return jsonify({"status": "trained", "n_docs": n_docs, "metric": metric})

@app.route("/admin/add", methods=["POST"])
def admin_add():

    from ai_engine_v2 import add_training_pair, retrain_background, load_json, save_json, UNANSWERED_FILE

    body = request.json or {}
    q = body.get("input") or body.get("question")
    a = body.get("output") or body.get("answer")

    if not q or not a:
        return jsonify({"error": "Provide 'input' and 'output' in body"}), 400

    add_training_pair(q, a)

    unanswered = load_json(UNANSWERED_FILE)
    new_unanswered = [item for item in unanswered if item.get("input") != q]
    save_json(UNANSWERED_FILE, new_unanswered)

    retrain_background()

    return jsonify({
        "status": "added and training started",
        "question": q,
        "answer": a
    }), 201
def basic_ai(text):
    greetings = ["hello", "hi", "hey"]
    feelings = ["how are you", "how do you feel"]
    bye = ["bye", "goodbye", "see you"]

    if any(word in text for word in greetings):
        return random.choice(["Hey there 👋", "Hello!", "Hi, how’s it going?"])
    elif any(word in text for word in feelings):
        return random.choice(["I'm doing great, thanks!", "Feeling awesome 🤖"])
    elif any(word in text for word in bye):
        return random.choice(["Goodbye! 👋", "See you soon!", "Take care!"])
    elif "time" in text:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        return f"The current time is {now}"
    else:
        return "I'm still learning, can you rephrase that?"
    
@app.route("/admin/unanswered", methods=["GET"])
def get_unanswered_questions():
    from ai_engine_v2 import get_unanswered
    data = get_unanswered()
    return jsonify(data), 200

@app.route("/admin/delete", methods=["POST"])
def admin_delete():
    body = request.json or {}
    idx = body.get("index")
    if idx is None:
        return jsonify({"error": "Provide index"}), 400

    data = load_training_data()
    if idx < 0 or idx >= len(data):
        return jsonify({"error": "Invalid index"}), 400

    data.pop(idx)
    save_json(TRAIN_FILE, data)
    retrain_background()
    return jsonify({"status": "deleted"}), 200

@app.route("/training_data", methods=["GET"])
def get_training_data():
    data = load_training_data()
    return jsonify({"data": data})

@app.route("/")
def index():
    return "Hello! API is live."



if __name__ == "__main__":
    train_and_persist()
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
