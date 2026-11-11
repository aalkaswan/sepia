# SEPIA Explorer as a Hugging Face Docker Space (CPU Basic is enough: ~3 GB of memory at run time)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

# the same CPU stack the dataset embeddings were computed with
COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r /tmp/requirements.txt

# bake the pinned embedding model into the image, so a cold start does not download 1.2 GB
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/harrier-oss-v1-0.6b', revision='f9b9dc8d367d443f2479d27aa5d8d2850c0774ee', \
allow_patterns=['*.json', '*.txt', '*.jinja', '*.safetensors', '1_Pooling/*'])"
ENV HF_HUB_OFFLINE=1

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user && chown -R user /opt/hf
WORKDIR /app
COPY --chown=user . /app
USER user

EXPOSE 7860
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
