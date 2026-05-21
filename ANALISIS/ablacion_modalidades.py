# ablacion_modalidades.py
#
# evalua el modelo v5g con tres configuraciones:
#   1. texto + vitales (modelo completo)
#   2. solo texto (vitales puestas a cero)
#   3. solo vitales (texto reemplazado por cadena vacia)
#
# input:  modelo_sapbert_finetuned_v5g_6D_ablacion.pt
#         triage_with_demographics.csv
# output: imprimir AUC macro de cada configuracion

import torch
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_cc(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()


class SapBERTCrossAttentionModel(nn.Module):
    def __init__(self, clin_dim=6, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
        sapbert_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.bert = AutoModel.from_pretrained(sapbert_name)
        bert_dim = 768

        self.clin_project = nn.Sequential(
            nn.Linear(clin_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.cls_norm  = nn.LayerNorm(bert_dim)

        fusion_dim = bert_dim + attn_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_esi    = nn.Linear(128, 5)
        self.head_binary = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, clinical):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb   = outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_project(cls_emb)
        clin_proj = self.clin_project(clinical)
        text_seq  = text_proj.unsqueeze(1)
        clin_seq  = clin_proj.unsqueeze(1)
        attended, _ = self.cross_attn(query=clin_seq, key=text_seq, value=text_seq)
        attended  = self.attn_norm(attended.squeeze(1) + clin_proj)
        cls_normed = self.cls_norm(cls_emb)
        fused = torch.cat([cls_normed, text_proj, attended], dim=1)
        h = self.fusion(fused)
        return self.head_esi(h), self.head_binary(h)


class TriageDataset(Dataset):
    def __init__(self, texts, clinical, labels, tokenizer, max_length=64):
        self.texts    = texts
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'clinical':       self.clinical[idx],
            'labels':         self.labels[idx],
        }


def evaluar(model, dataloader, device):
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            clinical       = batch['clinical'].to(device)
            logits, _      = model(input_ids, attention_mask, clinical)
            probs          = F.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    auc    = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    acc    = (probs.argmax(axis=1) == labels).mean()
    return auc, acc


# carga datos
print("cargando datos...")
df = pd.read_csv("triage_with_demographics.csv", low_memory=False)

for col in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["acuity"])
df["acuity"] = df["acuity"].astype(int)
df = df[df["acuity"].between(1, 5)]
df = df.dropna(subset=["chiefcomplaint"])
df["chiefcomplaint"] = df["chiefcomplaint"].apply(normalize_cc)
df = df[df["chiefcomplaint"].str.len() > 0]

rangos = {
    "pain":        (0.0,  10.0),
    "temperature": (95.0, 107.0),
    "heartrate":   (20.0, 300.0),
    "resprate":    (4.0,  60.0),
}
for col, (lo, hi) in rangos.items():
    df = df[~(df[col].notna() & ~df[col].between(lo, hi))]
df = df[df[["pain", "temperature", "heartrate", "resprate"]].notna().all(axis=1)]
df["age"]      = df["age"].fillna(df["age"].median()).clip(18, 91)
df["gender_M"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print(f"  {len(df)} casos tras limpieza")

texts  = df["chiefcomplaint"].values
labels = df["acuity"].values - 1
clin_cols = ['pain', 'temperature', 'heartrate', 'resprate', 'age', 'gender_M']
clin_raw  = df[clin_cols].values.astype(np.float32)

# mismo split que en entrenamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.1, random_state=42, stratify=labels[idx_train])

# scaler ajustado solo sobre train
scaler = StandardScaler()
clin_scaled = np.zeros_like(clin_raw)
clin_scaled[idx_train] = scaler.fit_transform(clin_raw[idx_train])
clin_scaled[idx_val]   = scaler.transform(clin_raw[idx_val])
clin_scaled[idx_test]  = scaler.transform(clin_raw[idx_test])

# carga modelo
print("cargando modelo...")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("../SAPBERT_FINETUNED_MODEL/modelo_sapbert_finetuned_v5g_6D_ablation.pt", map_location=device, weights_only=False)
tokenizer  = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model      = SapBERTCrossAttentionModel(clin_dim=6).to(device)
model.load_state_dict(checkpoint["model_state"])
print(f"  dispositivo: {device}")

# configuracion 1: modelo completo
print("\nevaluando modelo completo (texto + vitales)...")
ds_completo = TriageDataset(texts[idx_test], clin_scaled[idx_test], labels[idx_test], tokenizer)
dl_completo = DataLoader(ds_completo, batch_size=64, num_workers=0)
auc_completo, acc_completo = evaluar(model, dl_completo, device)
print(f"  AUC: {auc_completo:.4f}  Accuracy: {acc_completo:.4f}")

# configuracion 2: solo texto (vitales a cero)
print("\nevaluando solo texto (vitales = 0)...")
clin_cero = np.zeros_like(clin_scaled[idx_test])
ds_texto = TriageDataset(texts[idx_test], clin_cero, labels[idx_test], tokenizer)
dl_texto = DataLoader(ds_texto, batch_size=64, num_workers=0)
auc_texto, acc_texto = evaluar(model, dl_texto, device)
print(f"  AUC: {auc_texto:.4f}  Accuracy: {acc_texto:.4f}")

# configuracion 3: solo vitales (texto vacio)
print("\nevaluando solo vitales (texto vacio)...")
texts_vacios = [""] * len(idx_test)
ds_vitales = TriageDataset(texts_vacios, clin_scaled[idx_test], labels[idx_test], tokenizer)
dl_vitales = DataLoader(ds_vitales, batch_size=64, num_workers=0)
auc_vitales, acc_vitales = evaluar(model, dl_vitales, device)
print(f"  AUC: {auc_vitales:.4f}  Accuracy: {acc_vitales:.4f}")

print("\nresumen ablacion:")
print(f"  texto + vitales: AUC {auc_completo:.4f}  Acc {acc_completo:.4f}")
print(f"  solo texto:      AUC {auc_texto:.4f}  Acc {acc_texto:.4f}")
print(f"  solo vitales:    AUC {auc_vitales:.4f}  Acc {acc_vitales:.4f}")