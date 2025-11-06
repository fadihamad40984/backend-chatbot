import os
import json
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import threading
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words_en = set(stopwords.words("english"))


stop_words_ar = {
    "من", "في", "على", "عن", "الى", "إلى", "كيف", "متى", "هل", "هو", "هي",
    "أنا", "انت", "أنت", "انتي", "هذا", "هذه", "ذلك", "الذي", "التي", "هناك",
    "ما", "لماذا", "مع", "كل", "كان", "لقد", "لقد", "بعد", "قبل", "او", "أو",
    "ثم", "حتى", "أن", "إن", "قد"
}

def preprocess_text(text: str) -> str:

    text = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", "", text)
    text = text.lower()


    tokens = text.split()

    cleaned = []
    for word in tokens:
    
        if word in stop_words_en or word in stop_words_ar:
            continue
    
        if re.match(r"[a-zA-Z]+", word):
            word = stemmer.stem(word)
        cleaned.append(word)

    return " ".join(cleaned)




VEC_FILE = "vectorizer.joblib"
MATRIX_FILE = "corpus_matrix.joblib"
CORPUS_FILE = "corpus_cache.json"   
TRAIN_FILE = "training_data.json"
MEMORY_FILE = "memory.json"
UNANSWERED_FILE = "unanswered.json"


def _ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f)

def load_json(path) -> list:
    _ensure_file(path, [])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_memory() -> List[Dict]:
    return load_json(MEMORY_FILE)

def save_memory(memory: List[Dict]):
    save_json(MEMORY_FILE, memory)

def load_training_data() -> List[Dict]:
    return load_json(TRAIN_FILE)

def add_training_pair(q: str, a: str):
    data = load_training_data()
    data.append({"input": q, "output": a})
    save_json(TRAIN_FILE, data)

def add_conversation(user_input: str, bot_output: str):
    mem = load_memory()
    mem.append({"input": user_input, "output": bot_output})
    save_memory(mem)

def build_corpus() -> List[Dict]:

    training = load_training_data()
    memory = load_memory()
    combined = training + memory
    save_json(CORPUS_FILE, combined)
    return combined

def train_and_persist(vectorizer_params=None) -> Tuple[int, float]:

    corpus = build_corpus()
    texts = [preprocess_text(item.get("input", "")) for item in corpus]
    if not texts:
        for p in (VEC_FILE, MATRIX_FILE, CORPUS_FILE):
            if os.path.exists(p):
                os.remove(p)
        return 0, 0.0

    params = {"ngram_range": (1,2), "max_features": 30000}
    if vectorizer_params:
        params.update(vectorizer_params)

    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(texts)  

    joblib.dump(vec, VEC_FILE)
    joblib.dump(X, MATRIX_FILE)
    save_json(CORPUS_FILE, corpus)
    return len(texts), float(X.sum())

def load_model():

    if not (os.path.exists(VEC_FILE) and os.path.exists(MATRIX_FILE) and os.path.exists(CORPUS_FILE)):
        return None, None, []
    try:
        vec = joblib.load(VEC_FILE)
        X = joblib.load(MATRIX_FILE)
        corpus = load_json(CORPUS_FILE)
        return vec, X, corpus
    except Exception:
        return None, None, []

def smart_reply_tfidf(user_input: str, top_k=1, threshold=0.35):

    user_input = preprocess_text(user_input.strip())
    if not user_input:
        return None, 0.0, None

    vec, X, corpus = load_model()
    if vec is None or X is None or len(corpus) == 0:
        return None, 0.0, None

    q_vec = vec.transform([user_input])  
    sims = cosine_similarity(q_vec, X).flatten() 
    if sims.size == 0:
        return None, 0.0, None

    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        matched = corpus[best_idx]
        return matched.get("output"), best_score, best_idx
    return None, best_score, None

_retrain_lock = threading.Lock()
def retrain_background(vectorizer_params=None):
    if _retrain_lock.locked():
        return False
    def _job():
        with _retrain_lock:
            train_and_persist(vectorizer_params)
    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return True


def add_unanswered(question: str):

    data = load_json(UNANSWERED_FILE)
    data.append({"input": question})
    save_json(UNANSWERED_FILE, data)

def get_unanswered():

    return load_json(UNANSWERED_FILE)