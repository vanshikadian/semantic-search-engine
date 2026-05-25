# Semantic Search Engine

Full-stack semantic search over 100K MS MARCO passages with a Next.js frontend,
FastAPI backend, Sentence Transformers embeddings, and FAISS exact vector search.

## Technical Stack

- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS
- Backend: FastAPI, Pydantic, Uvicorn
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: FAISS `IndexFlatIP`
- Data format: JSONL source passages with JSON metadata artifacts
- Serialization: FAISS binary index and `orjson` metadata files
- Containerization: Docker for the backend service

## System Overview

The application indexes MS MARCO passage text into dense vector embeddings and
serves semantic retrieval through a FastAPI API. User queries are embedded with
the same Sentence Transformers model, normalized, and searched against a FAISS
inner-product index. The frontend sends natural-language queries to the API and
renders ranked passages with similarity scores.

## Indexing Pipeline

The indexing script reads passages from `backend/data/passages.jsonl`, embeds
text in batches, checkpoints intermediate NumPy chunks, and assembles the final
FAISS index. The final search artifacts are stored in `backend/faiss_index/`.

Key artifacts:

- `index.faiss.part-*`: split FAISS vector index for 100K passages
- `metadata.json`: passage IDs and text mapped to FAISS row positions
- `eval_results.json`: retrieval evaluation metrics

The backend Docker build reconstructs `index.faiss` from the split index parts.

## Backend API

The backend exposes a small semantic search API:

- `GET /health`: service and artifact readiness
- `GET /stats`: index size, model name, index type, and load state
- `POST /search`: ranked semantic search results for a query

Search requests accept a query string and result count. Responses include the
passage ID, passage text, and similarity score.

## Retrieval Details

Embeddings are generated with MiniLM and stored as `float32` vectors. Before
indexing and querying, vectors are L2-normalized so FAISS inner product behaves
as cosine similarity. The index uses exact search rather than approximate ANN,
which keeps retrieval behavior deterministic for the 100K-passage dataset.

## Frontend

The frontend is a client-rendered Next.js interface with a search input, loading
state, error handling, and ranked result cards. It also includes an evaluation
results page that reads the generated metrics artifact.
