# umls_embeddings_biolord.py
#
# Pre-calcula los embeddings de todos los conceptos y sinonimos de
# umls_symptoms.csv usando BioLORD-2023, aplica PCA 768D -> 256D
# y guarda el resultado en un .pt listo para la API.
#
# BioLORD se usa para el autocomplete porque preserva discriminacion semantica
# a nivel de frase, a diferencia de SapBERT cuyo espacio colapsa hacia entity
# linking y no permite ranking coherente sobre el indice completo.
#
# Output: umls_embeddings_full_biolord.pt

import csv
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

umls_csv    = "umls_symptoms.csv"
output_path = "umls_embeddings_full_biolord.pt"
biolord_name = "FremyCompany/BioLORD-2023"
batch_size  = 128
pca_dim     = 256


# carga conceptos y expande sinonimos
print("cargando y expandiendo sinonimos")

concept_cuis = []
concept_canonical = []
concept_synonyms_raw = []
concept_semantic_types = []
all_texts = []
all_cui_indices = []

with open(umls_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for concept_idx, row in enumerate(reader):
        cui = row['CUI']
        canonical = row['canonical_name']
        synonyms_str = row['synonyms']
        sty = row['semantic_type']

        concept_cuis.append(cui)
        concept_canonical.append(canonical)
        concept_synonyms_raw.append(synonyms_str)
        concept_semantic_types.append(sty)

        all_texts.append(canonical)
        all_cui_indices.append(concept_idx)

        if synonyms_str:
            for syn in synonyms_str.split('|'):
                syn = syn.strip()
                if syn and len(syn) >= 2 and len(syn) <= 100:
                    if syn.lower() != canonical.lower():
                        all_texts.append(syn)
                        all_cui_indices.append(concept_idx)

n_concepts = len(concept_cuis)
n_texts = len(all_texts)
print(f"  {n_concepts} conceptos")
print(f"  {n_texts} textos totales (canonical + sinonimos)")
print(f"  media: {n_texts/n_concepts:.1f} textos por concepto")


# cargar BioLORD
print(f"\ncargando {biolord_name}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")
tokenizer = AutoTokenizer.from_pretrained(biolord_name)
model = AutoModel.from_pretrained(biolord_name).to(device).eval()
print("  cargado")


# calcular embeddings con mean pooling
print(f"\ncalculando {n_texts} embeddings")
all_embeddings = []

for i in range(0, n_texts, batch_size):
    batch_texts = all_texts[i:i+batch_size]

    encoded = tokenizer(
        batch_texts, padding=True, truncation=True,
        max_length=64, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        token_emb = outputs.last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).expand(token_emb.size()).float()
        mean_emb = (token_emb * mask).sum(1) / mask.sum(1)
        all_embeddings.append(mean_emb.cpu())

embeddings_768 = torch.cat(all_embeddings, dim=0).numpy()
print(f"  {embeddings_768.shape[0]} embeddings calculados ({embeddings_768.shape[1]}D)")

# reduccion PCA 768D -> 256D
# se ajusta sobre subset de 100k por eficiencia de memoria
print(f"\nPCA 768D -> {pca_dim}D...")
pca = PCA(n_components=pca_dim, random_state=42)

if len(embeddings_768) > 100_000:
    rng = np.random.RandomState(42)
    fit_indices = rng.choice(len(embeddings_768), 100_000, replace=False)
    pca.fit(embeddings_768[fit_indices])
else:
    pca.fit(embeddings_768)

explained_var  = pca.explained_variance_ratio_.sum()
print(f"  varianza explicada: {explained_var:.3f} ({explained_var*100:.1f}%)")

embeddings_256 = pca.transform(embeddings_768)
norms = np.maximum(np.linalg.norm(embeddings_256, axis=1, keepdims=True), 1e-8)
embeddings_256 = embeddings_256 / norms
embeddings_final = torch.tensor(embeddings_256, dtype=torch.float16)


# guardar
print(f"\nguardando en {output_path}")

torch.save({
    'embeddings':          embeddings_final,
    'synonym_texts':       all_texts,
    'cui_indices':         all_cui_indices,
    'concept_cuis':        concept_cuis,
    'concept_canonical':   concept_canonical,
    'concept_synonyms':    concept_synonyms_raw,
    'concept_semantic_types': concept_semantic_types,
    'pca_components':      torch.tensor(pca.components_, dtype=torch.float32),
    'pca_mean':            torch.tensor(pca.mean_, dtype=torch.float32),
    'model_name':          biolord_name,
    'pca_dim':             pca_dim,
    'pca_explained_variance': float(explained_var),
    'n_concepts':          n_concepts,
    'n_texts':             n_texts,
}, output_path)

