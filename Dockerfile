FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget unzip curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

COPY FRONTEND_DEMO/API_v240326_biolord.py .
COPY FRONTEND_DEMO/pipeline_pretriaje_esi_080426_v5g.py .
COPY FRONTEND_DEMO/INDEX_v240326.html .

RUN gdown 1TXdPFzcLdGOQI51n-pUnTLiy72gAiKjd -O modelo_sapbert_finetuned_v5g_6D_ablation.pt
RUN gdown 17FmGcaxJk1gd-iFRFoN48bJi40MsmPbQ -O umls_embeddings_full_biolord.pt
RUN gdown 1WDczuJenB6IsVXT9rsW3B3QFcE20NDU4 -O biobert_ner.zip && \
    unzip biobert_ner.zip -d . && \
    rm biobert_ner.zip

EXPOSE 7860

CMD ["uvicorn", "API_v240326_biolord:app", "--host", "0.0.0.0", "--port", "7860"]
