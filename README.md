# Agentic RAG Assistant with LangGraph & Ollama

A local, privacy-first Retrieval-Augmented Generation (RAG) system engineered with **LangGraph**, **FastAPI**, **Streamlit**, and **Qdrant**.

The application runs entirely offline without third-party API dependencies, leveraging containerized Ollama models alongside hybrid dense + sparse retrieval in Qdrant.

---

## Key Highlights

- **Graph-Driven Pipeline**: Dynamic routing, state management, and reasoning powered by LangGraph.
- **Hybrid Retrieval**: Dense embeddings (FastEmbed, `thenlper/gte-large`) combined with sparse `miniCOIL`, fused by Reciprocal Rank Fusion in Qdrant.
- **Two-Stage Ranking**: Optional cross-encoder reranking on top of RRF candidates.
- **Measured Quality**: A retrieval evaluation harness with recall@k, MRR@k and latency; see [Evaluation](#evaluation).
- **Adaptive Prompt Engineering**: Context-aware prompts tailored dynamically based on document presence and query intent.
- **Modern Full-Stack Architecture**: Asynchronous FastAPI service paired with an intuitive Streamlit workspace.
- **Self-Contained Deployment**: Docker Compose environment with automated model loading and live system health diagnostics.



###  **Local Ollama Deployment**

- Fully containerized Ollama service
- Automatic model pulling on startup
- No external dependencies

###  **Dynamic Model Discovery**

- Automatically detects all available Ollama models
- No hardcoded model configurations
- Real-time model list updates

###  **Intelligent Context Handling**

- Programmatically detects document availability
- Uses different prompts based on context presence
- Helpful guidance when no documents are found

###  **Enhanced Provider Selection**

- Choose between Ollama and Groq providers
- Model-specific selection within each provider
- Proper backend routing based on UI selection

## Prerequisites

- Docker
- Docker Compose

## Setup and Running

1. **Clone the repository:**

```bash
git clone https://github.com/vikagit11/Langgraph-rag-app.git
cd Langgraph-rag-app
```

2. **Start the services:**

```bash
docker-compose up --build
```

This will automatically:

- Start Qdrant vector database
- Deploy Ollama service locally
- Pull the llama3.2 model
- Start the FastAPI backend
- Launch the Streamlit frontend

3. **Access the applications:**

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Usage

### 1. **System Health Check**

- Use the " Health Check" button in the sidebar
- Verifies connectivity to Ollama and Qdrant
- Shows system status and available models

### 2. **Model Selection**

- **Provider Selection**: Choose between Ollama (local) or Groq (cloud)
- **Model Selection**: Pick from available models in each provider
- **Dynamic Discovery**: Ollama models are automatically detected

### 3. **Document Management**

**PDF Upload:**

- Use the "📄 Upload PDF Files" section
- Select multiple PDF files
- Automatic processing and indexing

**Manual Document Upload:**

- Use the " Add Text Document" section
- Enter document text and optional metadata
- JSON format metadata support

**Document Clearing:**

- " Clear All Documents" button
- Removes all indexed documents
- Confirmation dialog for safety

### 4. **Chat Interface**

- **Smart Responses**: Different prompts based on document availability
- **Context-Aware**: Uses RAG when documents are available
- **Helpful Guidance**: Provides instructions when no documents are found
- **Source Display**: Shows retrieved document sources
- **Retriever Logs**: Optional display of search results

## Architecture

### **Local-First Design**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │     Qdrant      │
│   Frontend      │◄──►│    Backend      │◄──►│  Vector Store   │
│  (Port 8501)    │    │  (Port 8000)    │    │  (Port 6333)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Ollama      │
                       │   LLM Service   │
                       │  (Port 11434)   │
                       └─────────────────┘
```

### **LangGraph Pipeline**

- **Search Node**: Hybrid search (dense + sparse) fused with RRF, optionally reranked by a cross-encoder
- **Generate Node**: Context-aware response generation
- **Smart Routing**: Different prompts based on context availability

### **Components**

- **FastEmbed**: 1024-dimensional dense embeddings (`thenlper/gte-large`) plus sparse `Qdrant/minicoil-v1`
- **Stable chunk ids**: payload carries `doc_id` (content hash), `source` and `chunk_index`, so evaluation labels survive re-indexing
- **Quantization**: Binary Quantization is available as an opt-in (`create_hybrid_collection(quantize=True)`); measured impact is in [Evaluation](#evaluation)
- **Dynamic Models**: Auto-discovery of available models

## Evaluation

Retrieval quality is measured rather than assumed. `eval/eval_retrieval.py` computes recall@5/10, MRR@10 and latency (p50/p95) against a labelled query set. Corpus: public Qdrant documentation pages, 38 chunks, 15 queries.

| Configuration | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| dense only | 0.733 | 1.000 | 0.527 |
| hybrid + RRF | 0.800 | 0.867 | 0.474 |
| hybrid + RRF + cross-encoder rerank | **0.867** | 0.867 | **0.722** |

What the numbers say:

- RRF helps on queries that reuse document wording (+10–12 pp MRR on two corpora) and loses on paraphrased queries, where the sparse channel adds noise without lexical overlap.
- Reranking improves ordering, not recall@10: it can only reorder the candidates it is given.
- Binary Quantization was measured on 37k × 384d points: container RSS growth drops from **192 MiB to 70 MiB (~2.7×)** when original vectors are kept on disk. Without `on_disk`, quantization *increases* memory, because 32× applies to vector bytes only — HNSW and payload do not shrink.

Reproduce:

```bash
python eval/eval_retrieval.py --corpus eval/corpus \
  --queries eval/eval_queries_docs_paraphrased.csv \
  --out eval/results/check.csv --dense-model BAAI/bge-small-en-v1.5 \
  --configs dense,hybrid_rrf,hybrid_rrf_rerank
```

Raw runs are stored in `eval/results/`; the quantization methodology is described in `eval/results/quantization-report.md`.

## API Endpoints

### **Core Endpoints**

- `GET /health` - System health and status
- `GET /models` - Available models from all providers
- `POST /query` - Query documents with provider/model selection
- `POST /upload` - Add documents manually
- `POST /upload-pdfs` - Upload and process PDF files
- `DELETE /clear-collection` - Clear all documents

### **Testing Endpoints**

- `GET /test-hybrid-search` - Test hybrid search functionality
- `GET /test-retriever` - Test basic retrieval

## Configuration

### **Environment Variables**

```bash
# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Ollama Configuration (Local)
OLLAMA_URL=http://ollama:11434

# Groq Configuration (Optional)
GROQ_API_KEY=your_groq_api_key
GROQ_URL=https://api.groq.com/openai/v1
```

### **Docker Services**

```yaml
services:
  app: # FastAPI + Streamlit
  qdrant: # Vector database
  ollama: # Local LLM service
```

## Model Management

### **Ollama Models**

- **Automatic Discovery**: All pulled models are automatically available
- **Dynamic Loading**: No configuration file updates needed
- **Model Pulling**: Add new models with `docker-compose exec ollama ollama pull <model>`

### **Supported Providers**

- **Ollama**: Local inference (recommended)
- **Groq**: Cloud inference (requires API key)

## Smart Context Handling

### **When Documents Are Available**

- Uses comprehensive RAG prompts
- Provides detailed, context-based responses
- Shows source documents and metadata

### **When No Documents Are Found**

- Uses helpful guidance prompts
- Suggests document upload steps
- Asks for question clarification
- Provides clear next steps

## Benefits

### ** Privacy & Security**

- **Local Processing**: Documents never leave your infrastructure
- **No API Keys**: Core functionality works without external services
- **Data Control**: Complete control over your data

### ** Cost Effective**

- **No Per-Token Charges**: Local inference is free
- **Scalable**: Add more models without additional costs
- **Efficient**: Quantization is available as an opt-in; its memory impact is measured in [Evaluation](#evaluation)

### **🛠️ Developer Friendly**

- **Easy Setup**: Single command deployment
- **Extensible**: Add new providers and models easily
- **Well Documented**: Comprehensive API documentation

## Development

### **File Structure**

```
miniCOIL-RAG/
├── app/
│   ├── config.py          # Configuration and dynamic model loading
│   ├── endpoints.py       # FastAPI route handlers
│   ├── graph.py          # LangGraph pipeline with smart context handling
│   ├── llm_providers.py  # Provider abstraction layer
│   ├── models.py         # Pydantic models
│   ├── streamlit_app.py  # Frontend interface
│   └── vector_store.py   # Qdrant integration
├── docker-compose.yml    # Service orchestration
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
├── .dockerignore       # Docker build exclusions
├── .gitignore         # Git exclusions
└── README.md          # This file
```

### **Adding New Models**

1. Pull model in Ollama: `ollama pull <model-name>`
2. Restart the application
3. Model appears automatically in the UI

### **Adding New Providers**

1. Update `llm_providers.py`
2. Add configuration in `config.py`
3. Update UI provider selection

## Troubleshooting

### **Common Issues**

- **Ollama Connection**: Ensure Ollama service is running
- **Model Not Found**: Check if model is pulled in Ollama
- **Memory Issues**: Ensure sufficient RAM for models
- **Port Conflicts**: Check if ports 8000, 8501, 6333, 11434 are available


