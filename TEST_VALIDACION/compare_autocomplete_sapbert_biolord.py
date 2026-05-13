# compara BioLORD vs SapBERT fine-tuned como motor de autocomplete
# para cada consulta coloquial busca el top-5 en el indice umls_symptoms.csv
# y muestra los resultados de ambos modelos lado a lado
#
# requiere umls_embeddings_full_biolord.pt en la misma carpeta
# requiere modelo_sapbert_finetuned_v5g_7D_ablation.pt en ../SAPBERT_RAW

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

biolord_pt_path = "../UMLS_BIOLORD/umls_embeddings_full_biolord.pt"
sapbert_path    = "../SAPBERT_FINETUNED_MODEL/modelo_sapbert_finetuned_v5g_6D_ablation.pt"
sapbert_name    = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

queries = [
    "the runs",
    "trouble peeing",
    "can't breathe",
    "belly hurts",
    "heart racing",
    "throwing up",
    "pins and needles",
    "passed out",
    "chest feels tight",
    "swollen ankles",
    "burning when i pee",
    "seeing double",
    "blood in poo",
    "pus coming out",
    "cannot keep any water down",
    "whole body is covered in hives",
    "lips are getting bigger",
    "cannot move my legs",
    "rash not fading when pressed",
    "accidentally swallowed a fish bone",
]

TOP_K = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# --- carga indice desde .pt ---
print("cargando indice desde .pt...")
biolord_data = torch.load(biolord_pt_path, map_location='cpu', weights_only=False)
name_texts = biolord_data['synonym_texts']
cui_indices = biolord_data['cui_indices']
concept_cuis = biolord_data['concept_cuis']
concept_canonical = biolord_data['concept_canonical']
concept_stys = biolord_data['concept_semantic_types']
biolord_embeddings = biolord_data['embeddings'].float().numpy()
pca_mean = biolord_data['pca_mean'].numpy()
pca_components = biolord_data['pca_components'].numpy()

names = []
for i, text in enumerate(name_texts):
    ci = cui_indices[i]
    names.append((concept_cuis[ci], text, concept_canonical[ci], concept_stys[ci]))

print(f"  {len(names)} nombres en el indice")


# --- carga modelos ---
print("cargando biolord...")
biolord_model = SentenceTransformer("FremyCompany/BioLORD-2023")

print("cargando sapbert fine-tuned...")
tokenizer = AutoTokenizer.from_pretrained(sapbert_name)
checkpoint = torch.load(sapbert_path, map_location=device, weights_only=False)
bert_state = {k.replace("bert.", ""): v
              for k, v in checkpoint["model_state"].items()
              if k.startswith("bert.")}
bert_ft = AutoModel.from_pretrained(sapbert_name)
bert_ft.load_state_dict(bert_state)
bert_ft.to(device).eval()
print("modelos cargados")


# --- encodear indice con sapbert ---
print(f"\nencodeando {len(name_texts)} textos con sapbert fine-tuned...")
BATCH_SIZE = 256
sapbert_embeddings = []

for i in range(0, len(name_texts), BATCH_SIZE):
    batch = name_texts[i:i+BATCH_SIZE]
    encoded = tokenizer(batch, padding=True, truncation=True,
                        max_length=64, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = bert_ft(**encoded)
    embs = out.last_hidden_state[:, 0, :].cpu().numpy()
    sapbert_embeddings.append(embs)

    if i % 50000 == 0:
        print(f"  {i}/{len(name_texts)}")

sapbert_embeddings = np.vstack(sapbert_embeddings)
print(f"  shape: {sapbert_embeddings.shape}")


# --- funciones de busqueda ---
def search(query_emb, index_embs):
    sims = cosine_similarity([query_emb], index_embs)[0]
    top_idx = np.argsort(sims)[::-1][:TOP_K]
    return [(names[i], sims[i]) for i in top_idx]


def encode_biolord(query):
    raw = biolord_model.encode([query])[0]
    centered = raw - pca_mean
    projected = centered @ pca_components.T
    norm = np.linalg.norm(projected)
    return projected / norm if norm > 0 else projected


def encode_sapbert(query):
    encoded = tokenizer([query], padding=True, truncation=True,
                        max_length=64, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = bert_ft(**encoded)
    return out.last_hidden_state[:, 0, :].cpu().numpy()[0]


# --- busqueda ---
print("\nresultados autocomplete: biolord vs sapbert fine-tuned")

for query in queries:
    print(f"\nquery: '{query}'")

    emb_bl = encode_biolord(query)
    emb_ft = encode_sapbert(query)

    results_bl = search(emb_bl, biolord_embeddings)
    results_ft = search(emb_ft, sapbert_embeddings)

    print(f"  biolord")
    for (cui, name, canonical, sty), sim in results_bl:
        print(f"    {sim:.3f}  {name}")

    print(f"  sapbert ft")
    for (cui, name, canonical, sty), sim in results_ft:
        print(f"    {sim:.3f}  {name}")