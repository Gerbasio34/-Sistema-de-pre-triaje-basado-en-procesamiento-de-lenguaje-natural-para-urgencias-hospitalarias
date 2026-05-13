# -*- coding: utf-8 -*-

# comparacion_smote_v5g.py
# Compara MLP con y sin SMOTE usando exactamente la misma limpieza
# y features que el modelo SapBERT v5g (6D, sin crosses, sin flags missing).
# El objetivo es validar empiricamente el rechazo de SMOTE.
#
# Input:  triage_with_demographics.csv
# Output: metricas comparativas en consola

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score
import re
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dispositivo: {device}")


# --- limpieza identica al script de entrenamiento v5g ---

def normalize_chiefcomplaint(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()

print("cargando datos...")
df = pd.read_csv("triage_with_demographics.csv", low_memory=False)
print(f"  {len(df):,} filas cargadas")

cols_numericas = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "age"]
for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["acuity"])
df["acuity"] = df["acuity"].astype(int)
df = df[df["acuity"].between(1, 5)]
df = df.dropna(subset=["chiefcomplaint"])
df["chiefcomplaint"] = df["chiefcomplaint"].apply(normalize_chiefcomplaint)
df = df[df["chiefcomplaint"].str.len() > 0]

rangos = {
    "pain":        (0.0,  10.0),
    "temperature": (95.0, 107.0),
    "heartrate":   (20.0, 300.0),
    "resprate":    (4.0,  60.0),
}
for col, (vmin, vmax) in rangos.items():
    if col in df.columns:
        mask = df[col].notna() & ~df[col].between(vmin, vmax)
        if mask.sum() > 0:
            df = df[~mask]

vitals = ["pain", "temperature", "heartrate", "resprate"]
df = df[df[vitals].notna().all(axis=1)]

df["age"] = df["age"].fillna(df["age"].median()).clip(18, 91)
df["gender_M"] = df["gender"].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print(f"  dataset final: {len(df):,} casos")
for level in range(1, 6):
    n = (df["acuity"] == level).sum()
    print(f"  ESI {level}: {n:,} ({n/len(df)*100:.1f}%)")


# --- embeddings sapbert congelado ---

print("\ncargando SapBERT para embeddings...")
sapbert = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

unique_cc = df["chiefcomplaint"].unique()
print(f"  {len(unique_cc):,} chief complaints unicos")

cache = {}
for i in range(0, len(unique_cc), 512):
    batch = unique_cc[i:i+512]
    embs = sapbert.encode(list(batch), convert_to_numpy=True, show_progress_bar=False)
    for cc, emb in zip(batch, embs):
        cache[cc] = emb
    if (i // 512) % 10 == 0:
        print(f"  {min(i+512, len(unique_cc)):,}/{len(unique_cc):,}")

X_emb = np.array([cache[cc] for cc in df["chiefcomplaint"].values], dtype=np.float32)


# --- features clinicas 6D (igual que v5g) ---

labels = df["acuity"].values - 1
clin_cols = ["pain", "temperature", "heartrate", "resprate", "age", "gender_M"]
clin_raw = df[clin_cols].values.astype(np.float32)

idx = np.arange(len(labels))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
idx_train, idx_val = train_test_split(idx_train, test_size=0.1, random_state=42, stratify=labels[idx_train])

print(f"\ntrain: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}")

scaler_emb = StandardScaler()
scaler_clin = StandardScaler()

X_emb_sc = np.zeros_like(X_emb)
X_emb_sc[idx_train] = scaler_emb.fit_transform(X_emb[idx_train])
X_emb_sc[idx_val]   = scaler_emb.transform(X_emb[idx_val])
X_emb_sc[idx_test]  = scaler_emb.transform(X_emb[idx_test])

clin_sc = np.zeros_like(clin_raw)
clin_sc[idx_train] = scaler_clin.fit_transform(clin_raw[idx_train])
clin_sc[idx_val]   = scaler_clin.transform(clin_raw[idx_val])
clin_sc[idx_test]  = scaler_clin.transform(clin_raw[idx_test])


# --- modelo mlp simple ---

class MLP(nn.Module):
    def __init__(self, emb_dim=768, clin_dim=6, dropout=0.3):
        super().__init__()
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.clin_branch = nn.Sequential(
            nn.Linear(clin_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(288, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(128, 5)

    def forward(self, emb, clin):
        h = self.fusion(torch.cat([self.emb_branch(emb), self.clin_branch(clin)], dim=1))
        return self.head(h)


def train_and_eval(emb_tr, clin_tr, y_tr, emb_va, clin_va, y_va,
                   emb_te, clin_te, y_te, name, class_weights):

    model = MLP().to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_dl = DataLoader(TensorDataset(
        torch.tensor(emb_tr), torch.tensor(clin_tr), torch.tensor(y_tr)),
        batch_size=512, shuffle=True)
    va_dl = DataLoader(TensorDataset(
        torch.tensor(emb_va), torch.tensor(clin_va), torch.tensor(y_va)),
        batch_size=512)
    te_dl = DataLoader(TensorDataset(
        torch.tensor(emb_te), torch.tensor(clin_te), torch.tensor(y_te)),
        batch_size=512)

    best_auc, best_state, patience = 0.0, None, 0

    for epoch in range(1, 51):
        model.train()
        for emb, clin, lab in tr_dl:
            emb, clin, lab = emb.to(device), clin.to(device), lab.to(device)
            optimizer.zero_grad()
            criterion(model(emb, clin), lab).backward()
            optimizer.step()

        model.eval()
        all_p, all_l, all_pr = [], [], []
        with torch.no_grad():
            for emb, clin, lab in va_dl:
                logits = model(emb.to(device), clin.to(device))
                probs = F.softmax(logits, dim=1)
                all_p.append(logits.argmax(1).cpu())
                all_l.append(lab)
                all_pr.append(probs.cpu())

        val_preds  = torch.cat(all_p).numpy()
        val_labels = torch.cat(all_l).numpy()
        val_probs  = torch.cat(all_pr).numpy()
        val_auc = roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro")
        val_acc = (val_preds == val_labels).mean()

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"  [{name}] epoca {epoch:2d} | acc: {val_acc:.4f} | auc: {val_auc:.4f} *")
        else:
            patience += 1
            if patience >= 8:
                print(f"  [{name}] early stopping epoca {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval().to(device)

    all_p, all_l, all_pr = [], [], []
    with torch.no_grad():
        for emb, clin, lab in te_dl:
            logits = model(emb.to(device), clin.to(device))
            probs = F.softmax(logits, dim=1)
            all_p.append(logits.argmax(1).cpu())
            all_l.append(lab)
            all_pr.append(probs.cpu())

    preds  = torch.cat(all_p).numpy()
    y_true = torch.cat(all_l).numpy()
    probs  = torch.cat(all_pr).numpy()

    acc   = (preds == y_true).mean()
    auc   = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    kappa = cohen_kappa_score(y_true, preds, weights="quadratic")
    undertriage = ((y_true <= 1) & (preds >= 3)).sum()

    return acc, auc, kappa, undertriage, preds, y_true


# --- entrenamiento sin smote ---

print("\n--- entrenamiento 1: sin smote ---")
class_weights = [5.0, 1.5, 1.0, 3.0, 12.0]

acc_ns, auc_ns, kappa_ns, ut_ns, preds_ns, y_true = train_and_eval(
    X_emb_sc[idx_train], clin_sc[idx_train], labels[idx_train].astype(np.int64),
    X_emb_sc[idx_val],   clin_sc[idx_val],   labels[idx_val].astype(np.int64),
    X_emb_sc[idx_test],  clin_sc[idx_test],  labels[idx_test].astype(np.int64),
    "sin-smote", class_weights)

print(f"\nresultados sin smote:")
print(classification_report(y_true, preds_ns,
    target_names=[f"ESI {i+1}" for i in range(5)], digits=4))


# --- entrenamiento con smote ---

print("\n--- entrenamiento 2: con smote ---")
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "imbalanced-learn", "--break-system-packages", "-q"])
    from imblearn.over_sampling import SMOTE

X_tr_full = np.concatenate([X_emb_sc[idx_train], clin_sc[idx_train]], axis=1)
y_tr = labels[idx_train].astype(np.int64)

print("distribucion antes de smote:")
for i in range(5):
    print(f"  ESI {i+1}: {(y_tr==i).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5,
              sampling_strategy={0: 40000, 4: 10000})
X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_full, y_tr)

print("distribucion despues de smote:")
for i in range(5):
    print(f"  ESI {i+1}: {(y_tr_smote==i).sum():,}")
print(f"total: {len(y_tr):,} -> {len(y_tr_smote):,} (+{len(y_tr_smote)-len(y_tr):,} sinteticos)")

emb_tr_smote  = X_tr_smote[:, :768].astype(np.float32)
clin_tr_smote = X_tr_smote[:, 768:].astype(np.float32)

acc_s, auc_s, kappa_s, ut_s, preds_s, _ = train_and_eval(
    emb_tr_smote, clin_tr_smote, y_tr_smote.astype(np.int64),
    X_emb_sc[idx_val],  clin_sc[idx_val],  labels[idx_val].astype(np.int64),
    X_emb_sc[idx_test], clin_sc[idx_test], labels[idx_test].astype(np.int64),
    "con-smote", [1.0, 1.0, 1.0, 1.0, 1.0])

print(f"\nresultados con smote:")
print(classification_report(y_true, preds_s,
    target_names=[f"ESI {i+1}" for i in range(5)], digits=4))


# --- comparacion final ---

print("\n--- comparacion final ---")
print(f"{'metrica':<25} {'sin smote':>12} {'con smote':>12}")
print("-" * 50)
print(f"{'accuracy':<25} {acc_ns:>12.4f} {acc_s:>12.4f}")
print(f"{'auc macro':<25} {auc_ns:>12.4f} {auc_s:>12.4f}")
print(f"{'kappa cuadratico':<25} {kappa_ns:>12.4f} {kappa_s:>12.4f}")
print(f"{'undertriage':<25} {ut_ns:>12} {ut_s:>12}")