# Calibracion de umbrales post-entrenamiento para el modelo SapBERT v5g.
# Busca factores multiplicativos por clase que maximizan el kappa cuadratico
# sobre el validation set, y evalua el resultado en el test set.
#
# Output: redirigir con > calibracion_umbrales.txt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from scipy.optimize import differential_evolution
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

model_path   = "../SAPBERT_RAW/modelo_sapbert_finetuned_v5g_7D_ablation.pt"
data_path    = "triage_with_demographics.csv"
sapbert_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
device       = "cuda" if torch.cuda.is_available() else "cpu"
batch_size   = 64
max_length   = 64

print(f"device: {device}")


# arquitectura del modelo (identica al entrenamiento)
class SapBERTCrossAttentionModel(nn.Module):
    def __init__(self, clin_dim=7, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
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
        attended   = self.attn_norm(attended.squeeze(1) + clin_proj)
        cls_normed = self.cls_norm(cls_emb)
        fused = torch.cat([cls_normed, text_proj, attended], dim=1)
        h = self.fusion(fused)
        return self.head_esi(h), self.head_binary(h)


# dataset
class TriageTextDataset(Dataset):
    def __init__(self, texts, clinical, labels, tokenizer, max_length=64):
        self.texts      = texts
        self.clinical   = torch.tensor(clinical, dtype=torch.float32)
        self.labels     = torch.tensor(labels,   dtype=torch.long)
        self.tokenizer  = tokenizer
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


# carga y limpieza del dataset (identica al entrenamiento)
def normalize_cc(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()

print(f"\ncargando {data_path}...")
df = pd.read_csv(data_path, low_memory=False)
print(f"  {len(df)} filas cargadas")

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
    if col in df.columns:
        df = df[~(df[col].notna() & ~df[col].between(lo, hi))]

df = df[~df[["pain", "temperature", "heartrate", "resprate"]].isna().any(axis=1)]
df["age"]      = df["age"].fillna(df["age"].median()).clip(18, 91)
df["gender_M"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print(f"  {len(df)} casos tras limpieza")
for level in range(1, 6):
    n = (df["acuity"] == level).sum()
    print(f"  ESI {level}: {n} ({n/len(df)*100:.1f}%)")


# split identico al entrenamiento (random_state=42, stratify)
texts  = df["chiefcomplaint"].values
labels = df["acuity"].values - 1

clin_cols = ['pain', 'temperature', 'heartrate', 'resprate', 'age', 'gender_M']
clin_raw  = df[clin_cols].values.astype(np.float32)

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.1, random_state=42, stratify=labels[idx_train])

print(f"\nsplit -- train: {len(idx_train)} | val: {len(idx_val)} | test: {len(idx_test)}")

scaler_clin = StandardScaler()
clin_scaled = np.zeros_like(clin_raw)
clin_scaled[idx_train] = scaler_clin.fit_transform(clin_raw[idx_train])
clin_scaled[idx_val]   = scaler_clin.transform(clin_raw[idx_val])
clin_scaled[idx_test]  = scaler_clin.transform(clin_raw[idx_test])


# carga del modelo
print(f"\ncargando modelo desde {model_path}...")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
tokenizer  = AutoTokenizer.from_pretrained(sapbert_name)
model      = SapBERTCrossAttentionModel(clin_dim=6, dropout=0.3)
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to(device)
print("  modelo cargado")


# inferencia — obtiene probabilidades softmax para un split
def get_probs(idx_split):
    ds = TriageTextDataset(
        texts[idx_split], clin_scaled[idx_split],
        labels[idx_split], tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clinical       = batch["clinical"].to(device)
            logits, _      = model(input_ids, attention_mask, clinical)
            probs          = F.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


print("\ninferencia sobre validation set...")
val_probs, val_labels = get_probs(idx_val)
print(f"  val: {len(val_labels)} casos")

print("inferencia sobre test set...")
test_probs, test_labels = get_probs(idx_test)
print(f"  test: {len(test_labels)} casos")


# calcula metricas con argmax estandar o con factores de escala por clase
def compute_metrics(probs, labels_true, thresholds=None, label=""):
    if thresholds is None:
        preds = np.argmax(probs, axis=1)
    else:
        preds = np.argmax(probs * np.array(thresholds), axis=1)

    target_names = [f"ESI {i+1}" for i in range(5)]
    print(f"\n--- {label} ---")
    print(classification_report(labels_true, preds, target_names=target_names, digits=4))

    acc   = (preds == labels_true).mean()
    auc   = roc_auc_score(labels_true, probs, multi_class="ovr", average="macro")
    kappa = cohen_kappa_score(labels_true, preds, weights="quadratic")
    grave = (np.abs(preds - labels_true) > 1).sum()
    total = len(labels_true)
    undertriage = ((labels_true <= 1) & (preds >= 3)).sum()

    print(f"accuracy: {acc:.4f}")
    print(f"AUC macro: {auc:.4f}")
    print(f"kappa cuadratico: {kappa:.4f}")
    print(f"errores graves: {grave} ({grave/total*100:.1f}%)")
    print(f"undertriage: {undertriage}")

    return preds, kappa


_, baseline_kappa_val  = compute_metrics(val_probs,  val_labels,  label="baseline -- validation set")
_, baseline_kappa_test = compute_metrics(test_probs, test_labels, label="baseline -- test set")


# busqueda de factores optimos con evolucion diferencial sobre validation set
# factor > 1 favorece esa clase, factor < 1 la penaliza
print("\nbuscando umbrales optimos sobre validation set...")
print("(puede tardar 1-2 minutos)")

def neg_kappa(factors):
    preds = np.argmax(val_probs * np.array(factors), axis=1)
    try:
        k = cohen_kappa_score(val_labels, preds, weights="quadratic")
    except Exception:
        k = -1.0
    return -k

bounds = [
    (0.5, 8.0),   # ESI 1
    (0.5, 3.0),   # ESI 2
    (0.5, 2.0),   # ESI 3
    (0.5, 5.0),   # ESI 4
    (0.5, 8.0),   # ESI 5
]

result = differential_evolution(neg_kappa, bounds=bounds, seed=42,
                                maxiter=200, tol=1e-4, popsize=15,
                                workers=1, disp=True)

optimal_factors = result.x
print(f"\nfactores optimos:")
for i, f in enumerate(optimal_factors):
    print(f"  ESI {i+1}: {f:.4f}")
print(f"kappa en val con factores: {-result.fun:.4f}")
print(f"kappa en val baseline: {baseline_kappa_val:.4f}")
print(f"mejora en val: {(-result.fun - baseline_kappa_val):+.4f}")


# evaluacion final en test con umbrales calibrados
_, calib_kappa_test = compute_metrics(
    test_probs, test_labels,
    thresholds=optimal_factors,
    label="calibrado -- test set")

print(f"\nresumen: argmax original vs umbrales calibrados (test set)")
print(f"  kappa original: {baseline_kappa_test:.4f}")
print(f"  kappa calibrado: {calib_kappa_test:.4f}")
print(f"  mejora: {(calib_kappa_test - baseline_kappa_test):+.4f}")
print(f"  factores aplicados: {[round(f,4) for f in optimal_factors]}")
print(f"  (ESI 1..5, factor >1 = mas predicciones de esa clase)")


# AUC por clase (one-vs-rest) en test set, comparado con KUTS
print("\nAUC por clase ESI (one-vs-rest) -- test set")
print("clase   v5g AUC   KUTS AUC   KUTS IC95%              KUTS sin knowledge")

kuts_auc           = [0.928, 0.849, 0.805, 0.921, 0.920]
kuts_ci            = [(0.922, 0.933), (0.847, 0.851), (0.800, 0.808),
                      (0.919, 0.923), (0.897, 0.940)]
kuts_knowledge_auc = [0.914, 0.790, 0.648, 0.909, 0.939]
esi_names          = ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]

v5g_aucs = []
for i in range(5):
    binary_labels = (test_labels == i).astype(int)
    if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
        auc_i = roc_auc_score(binary_labels, test_probs[:, i])
        v5g_aucs.append(auc_i)
        ci_str = f"[{kuts_ci[i][0]:.3f}, {kuts_ci[i][1]:.3f}]"
        print(f"  {esi_names[i]}   {auc_i:.4f}   {kuts_auc[i]:.3f}   {ci_str}   {kuts_knowledge_auc[i]:.3f}")
    else:
        v5g_aucs.append(None)
        print(f"  {esi_names[i]}   N/A")

auc_macro_check = np.mean([a for a in v5g_aucs if a is not None])
print(f"  Macro   {auc_macro_check:.4f}   0.885   [0.879, 0.888]   0.840")