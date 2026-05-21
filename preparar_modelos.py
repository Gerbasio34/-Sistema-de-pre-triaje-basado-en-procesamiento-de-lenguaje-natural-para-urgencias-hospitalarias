# setup_models.py
# ejecuta este script una vez antes de arrancar la API
# descarga los modelos necesarios desde Hugging Face Hub

from huggingface_hub import hf_hub_download, snapshot_download
import os

repo_id = "Gerbasio34/Modelos-necesarios-Sistema-pre-triaje-ESI"
dest = "./FRONTEND_DEMO"

print("descargando modelo SapBERT...")
hf_hub_download(repo_id=repo_id,
                filename="modelo_sapbert_finetuned_v5g_6D_ablation.pt",
                local_dir=dest)

print("descargando embeddings UMLS BioLORD...")
hf_hub_download(repo_id=repo_id,
                filename="umls_embeddings_full_biolord.pt",
                local_dir=dest)

print("descargando modelo NER BioBERT+CRF...")
snapshot_download(repo_id=repo_id,
                  allow_patterns="biobert_sintomas_ner_v4_crf/*",
                  local_dir=dest)

print("modelos descargados en", dest)
