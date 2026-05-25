FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache/home
ENV HF_DATASETS_CACHE=/app/.hf_cache/datasets
ENV HUGGINGFACE_HUB_CACHE=/app/.hf_cache/hub
ENV SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache/sentence_transformers
ENV PRELOAD_MODEL_ON_STARTUP=true
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN if [ ! -f /app/faiss_index/index.faiss ] && ls /app/faiss_index/index.faiss.part-* >/dev/null 2>&1; then \
    cat /app/faiss_index/index.faiss.part-* > /app/faiss_index/index.faiss; \
    fi

RUN mkdir -p /app/.hf_cache/home /app/.hf_cache/datasets /app/.hf_cache/hub /app/.hf_cache/sentence_transformers

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
