# Environment for task_clusters.py (CPU only).
#   docker build -t se-prompts-cluster -f docker/cluster.Dockerfile docker
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      numpy==2.0.2 pandas==2.2.3 scikit-learn==1.5.2 \
      sentence-transformers==5.1.2 transformers==4.57.1 \
      umap-learn==0.5.7 hdbscan==0.8.40 bertopic==0.16.4 matplotlib==3.9.2
ENV PYTHONDONTWRITEBYTECODE=1 HF_HOME=/work/data/hf_cache TOKENIZERS_PARALLELISM=false
