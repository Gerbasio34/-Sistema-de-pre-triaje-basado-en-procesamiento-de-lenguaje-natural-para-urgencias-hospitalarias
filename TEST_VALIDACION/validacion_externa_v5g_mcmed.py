# Validacion externa del modelo v5g sobre MC-MED (Stanford Health Care).
# Pain ausente en MC-MED -> imputado a 0.
# Input:  visits.csv
# Output: redirigir con > validacion_mcmed.txt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')


classifier_path = "../SAPBERT_RAW/modelo_sapbert_finetuned_v5g_7D_ablation.pt"
sapbert_name    = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
mcmed_path      = "visits.csv"
batch_size      = 64
max_length      = 64
pain_imputed    = 0.0  # pain no existe en MC-MED, se imputa a 0


# arquitectura del modelo (igual que en entrenamiento)
class SapBERTCrossAttentionModel(nn.Module):
    def __init__(self, clin_dim=6, dropout=0.3, n_heads=4, attn_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(sapbert_name)
        bert_dim = 768
        self.clin_project = nn.Sequential(
            nn.Linear(clin_dim, attn_dim), nn.LayerNorm(attn_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.text_project = nn.Sequential(
            nn.Linear(bert_dim, attn_dim), nn.LayerNorm(attn_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(attn_dim)
        self.cls_norm   = nn.LayerNorm(bert_dim)
        fusion_dim = bert_dim + attn_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout))
        self.head_esi    = nn.Linear(128, 5)
        self.head_binary = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask, clinical):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb   = outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_project(cls_emb)
        clin_proj = self.clin_project(clinical)
        attended, _ = self.cross_attn(
            query=clin_proj.unsqueeze(1),
            key=text_proj.unsqueeze(1),
            value=text_proj.unsqueeze(1))
        attended   = self.attn_norm(attended.squeeze(1) + clin_proj)
        cls_normed = self.cls_norm(cls_emb)
        fused = torch.cat([cls_normed, text_proj, attended], dim=1)
        h = self.fusion(fused)
        return self.head_esi(h), self.head_binary(h)


# carga del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
model = SapBERTCrossAttentionModel(
    clin_dim=checkpoint['clin_dim'],
    dropout=checkpoint.get('dropout', 0.3),
    n_heads=checkpoint.get('n_heads', 4),
    attn_dim=checkpoint.get('attn_dim', 128))
model.load_state_dict(checkpoint['model_state'])
model.to(device).eval()
scaler    = checkpoint['scaler_clin']
tokenizer = AutoTokenizer.from_pretrained(sapbert_name)
print(f"modelo cargado: clin_dim={checkpoint['clin_dim']}")


# carga y preprocesamiento de MC-MED
print(f"\ncargando {mcmed_path}...")
df = pd.read_csv(mcmed_path, low_memory=False)
print(f"filas brutas: {len(df)}")

# ESI viene como "3-Urgent", extraemos el numero
df['esi_raw'] = df['Triage_acuity'].astype(str).str.extract(r'(\d)').astype(float)
df = df.dropna(subset=['esi_raw', 'CC'])
df['esi'] = df['esi_raw'].astype(int)
df = df[df['esi'].between(1, 5)]

for col in ['Triage_Temp', 'Triage_HR', 'Triage_RR', 'Age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# temperatura en Celsius -> Fahrenheit (MIMIC usa Fahrenheit)
df['temp_F'] = df['Triage_Temp'] * 9/5 + 32

rangos = {
    'temp_F':    (95.0, 107.0),
    'Triage_HR': (20.0, 300.0),
    'Triage_RR': (4.0,  60.0),
}
for col, (lo, hi) in rangos.items():
    mask = df[col].notna() & ~df[col].between(lo, hi)
    if mask.sum() > 0:
        print(f"  {col}: {mask.sum()} outliers eliminados")
        df = df[~mask]

df = df.dropna(subset=['Triage_Temp', 'Triage_HR', 'Triage_RR', 'CC'])

df['age']      = pd.to_numeric(df['Age'], errors='coerce').fillna(50).clip(18, 91)
df['gender_M'] = df['Gender'].map({'M': 1.0, 'Male': 1.0, 'F': 0.0, 'Female': 0.0}).fillna(0.5)
df['cc']       = df['CC'].astype(str).str.lower().str.strip()
df['cc']       = df['cc'].apply(lambda x: re.sub(r'\s+', ' ', x))
df             = df[df['cc'].str.len() > 0].reset_index(drop=True)

print(f"casos validos: {len(df)}")
print(f"pain imputado a {pain_imputed} (ausente en MC-MED)")

print("\ndistribucion ESI en MC-MED:")
for esi in range(1, 6):
    n = (df['esi'] == esi).sum()
    print(f"  ESI {esi}: {n} ({n/len(df)*100:.1f}%)")


# construccion del vector clinico
# orden features: [pain, temperature, heartrate, resprate, age, gender_M]
clin_raw = np.column_stack([
    np.full(len(df), pain_imputed, dtype=np.float32),
    df['temp_F'].values.astype(np.float32),
    df['Triage_HR'].values.astype(np.float32),
    df['Triage_RR'].values.astype(np.float32),
    df['age'].values.astype(np.float32),
    df['gender_M'].values.astype(np.float32),
])
clin_scaled = scaler.transform(clin_raw)


# inferencia por batches
print("\nejecutando inferencia...")
all_probs  = []
all_preds  = []
all_labels = df['esi'].values - 1
texts      = df['cc'].tolist()

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_clin  = clin_scaled[i:i+batch_size]

    encoded = tokenizer(
        batch_texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt')
    input_ids      = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    clinical_t     = torch.tensor(batch_clin, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits_esi, _ = model(input_ids, attention_mask, clinical_t)
        probs = torch.softmax(logits_esi, dim=1).cpu().numpy()

    all_probs.append(probs)
    all_preds.extend(np.argmax(probs, axis=1).tolist())

all_probs = np.vstack(all_probs)
all_preds = np.array(all_preds)


# metricas
print("\nresultados validacion externa MC-MED (Stanford)")

labels_1idx = all_labels + 1
preds_1idx  = all_preds + 1

exact = (all_preds == all_labels).sum()
print(f"\naccuracy exacto: {exact/len(all_labels)*100:.2f}% ({exact}/{len(all_labels)})")

adj = (np.abs(all_preds - all_labels) <= 1).sum()
print(f"accuracy +-1 nivel: {adj/len(all_labels)*100:.2f}% ({adj}/{len(all_labels)})")

grave = (np.abs(all_preds - all_labels) >= 2).sum()
print(f"errores graves (>=2): {grave/len(all_labels)*100:.2f}% ({grave})")

undertriage = ((preds_1idx > labels_1idx) &
               (labels_1idx <= 2) & (preds_1idx >= 3)).sum()
print(f"undertriage critico: {undertriage} (ESI 1-2 clasificado como ESI 3+)")

kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
print(f"kappa cuadratico: {kappa:.4f}")

try:
    auc = roc_auc_score(
        np.eye(5)[all_labels], all_probs,
        multi_class='ovr', average='macro')
    print(f"AUC macro OvR: {auc:.4f}")
except Exception as e:
    print(f"AUC no calculable: {e}")

print(f"\nAUC por nivel ESI:")
for esi in range(5):
    try:
        binary_labels = (all_labels == esi).astype(int)
        if binary_labels.sum() > 0:
            auc_esi = roc_auc_score(binary_labels, all_probs[:, esi])
            print(f"  ESI {esi+1}: {auc_esi:.4f} (n={binary_labels.sum()})")
    except:
        pass

print(f"\ndistribucion de predicciones por ESI real:")
print("ESI real   pred1  pred2  pred3  pred4  pred5")
for true_esi in range(5):
    mask = all_labels == true_esi
    if mask.sum() == 0:
        continue
    counts = [(all_preds[mask] == p).sum() for p in range(5)]
    print(f"  ESI {true_esi+1}    " + "  ".join(f"{c:>5}" for c in counts))