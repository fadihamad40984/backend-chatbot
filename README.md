# AI Chatbot with Semantic Search

An intelligent chatbot powered by semantic search and question answering models, using only free and open data sources.

## Features

- 🔍 **Semantic Search**: Uses `all-MiniLM-L6-v2` sentence transformers for accurate document retrieval
- 🤖 **AI Question Answering**: Powered by `RoBERTa-base-SQuAD2` for extractive QA
- 📚 **Multiple Data Sources**: 
  - Wikipedia
  - arXiv (scientific papers)
  - PubMed (medical research)
  - Stack Exchange
  - OpenLibrary
  - OpenStreetMap
- ⚡ **GPU Acceleration**: CUDA support for faster inference
- 💾 **Persistent Knowledge Base**: Pre-computed embeddings saved to disk
- 🔄 **Dynamic Learning**: Fetch new data on-demand when answers aren't found
- 📊 **Admin API**: Manage training data, view stats, and trigger data fetching

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster performance)
- 4GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fadihamad40984/backend-chatbot.git
cd backend-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python server.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
  "message": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is a field of artificial intelligence...",
  "sources": ["Wikipedia: Machine learning"],
  "confidence": 0.95
}
```

### Admin Endpoints

#### Add Training Data
```bash
POST /admin/add
Content-Type: application/json

{
  "question": "What is your name?",
  "answer": "I am an AI assistant."
}
```

#### Fetch New Sources
```bash
POST /admin/fetch_sources
Content-Type: application/json

{
  "topics": ["Python programming", "Neural networks"]
}
```

#### Get Statistics
```bash
GET /admin/stats
```

**Response:**
```json
{
  "total_documents": 1411,
  "total_training_pairs": 35,
  "model_info": {
    "semantic_model": "all-MiniLM-L6-v2",
    "qa_model": "deepset/roberta-base-squad2",
    "device": "cuda:0"
  }
}
```

#### Get Unanswered Questions
```bash
GET /admin/unanswered
```

#### Delete Training Data
```bash
DELETE /admin/delete/<question>
```

#### Get All Training Data
```bash
GET /training_data
```

#### Train Model
```bash
POST /train
```

## Architecture

### Components

1. **Semantic Search Engine** (`semantic_engine.py`)
   - Generates 384-dimensional embeddings using sentence transformers
   - Performs cosine similarity search
   - Chunks long documents for better retrieval

2. **QA Model** (`qa_model.py`)
   - Uses RoBERTa for extractive question answering
   - Combines semantic search results for context
   - Returns answers with confidence scores

3. **Source Fetchers** (`source_fetchers.py`)
   - Wikipedia: Articles with proper rate limiting and User-Agent
   - arXiv: Scientific papers and preprints
   - PubMed: Medical and biomedical research
   - Stack Exchange: Programming Q&A
   - OpenLibrary: Book information
   - OpenStreetMap: Geographic data

4. **AI Engine** (`ai_engine_v2.py`)
   - Integrates all components
   - Manages knowledge base building
   - Handles conversation memory
   - Singleton pattern for resource efficiency

5. **Flask Server** (`server.py`)
   - RESTful API with CORS support
   - Admin endpoints for management
   - Background knowledge base loading

### Data Flow

```
User Question
    ↓
Semantic Search (find relevant documents)
    ↓
QA Model (extract answer from context)
    ↓
If no answer found → Fetch from external sources
    ↓
Return answer with sources
```

## Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"

# Search settings
SIMILARITY_THRESHOLD = 0.1  # Lower = more permissive
TOP_K_RESULTS = 3

# Document processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CONTEXT_LENGTH = 512

# Data sources
ENABLE_WIKIPEDIA = True
ENABLE_ARXIV = True
ENABLE_PUBMED = True
ENABLE_STACKOVERFLOW = True
```

## Performance

- **Knowledge Base**: 1400+ pre-indexed documents
- **Search Speed**: ~100ms on GPU, ~500ms on CPU
- **QA Inference**: ~200ms on GPU, ~2s on CPU
- **Memory Usage**: ~2GB with models loaded

## Deployment

### Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Environment**: Python 3.11

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "server.py"]
```

Build and run:
```bash
docker build -t ai-chatbot .
docker run -p 5000:5000 ai-chatbot
```

## Development

### Project Structure
```
backend-chatbot/
├── server.py              # Flask API server
├── ai_engine_v2.py       # AI engine integration
├── semantic_engine.py    # Semantic search
├── qa_model.py           # Question answering
├── source_fetchers.py    # Data source integrations
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── training_data.json    # Q&A pairs
├── semantic_documents.json   # Knowledge base
└── semantic_embeddings.npy   # Pre-computed embeddings
```

### Adding New Data Sources

1. Create a new fetcher class in `source_fetchers.py`:
```python
class NewSourceFetcher:
    def __init__(self):
        self.base_url = "https://api.example.com"
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        # Implement search logic
        return [{"title": "...", "text": "...", "source": "...", "url": "..."}]
```

2. Add to `SourceAggregator` in `source_fetchers.py`
3. Enable in `config.py`

## Troubleshooting

### Wikipedia 403 Errors
Fixed with proper User-Agent headers and rate limiting (1 req/sec).

### Low GPU Memory
Reduce `BATCH_SIZE` in config or use CPU:
```python
import torch
device = "cpu"  # Force CPU usage
```

### Slow Response Times
- Increase `SIMILARITY_THRESHOLD` to return fewer results
- Reduce `TOP_K_RESULTS`
- Use GPU acceleration

### Model Download Issues
Models are automatically downloaded from Hugging Face on first run. Ensure stable internet connection.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for semantic search
- [Hugging Face Transformers](https://huggingface.co/transformers/) for QA models
- Wikipedia, arXiv, PubMed, and other open data sources

## Support

For issues and questions:
- GitHub Issues: https://github.com/fadihamad40984/backend-chatbot/issues
- Email: fadih@example.com

---

Built with ❤️ using free and open-source technologies
