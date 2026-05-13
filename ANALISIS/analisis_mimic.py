# Analisis descriptivo del dataset MIMIC-IV-ED
#
# Calcula estadisticos del dataset usado para entrenar el clasificador ESI,
# incluyendo missing data, demografia, distribucion de features clinicas,
# poder discriminativo de cada variable, solapamiento entre niveles ESI,
# y analisis de chief complaints.
#
# Input:  triage_with_demographics.csv
# Output: redirigir con > analisis_mimic.txt
#
# Secciones:
#   1. Missing data antes de cualquier filtrado
#   2. Demografia del dataset limpio
#   3. Distribucion de features clinicas por nivel ESI
#   4. Poder discriminativo de cada feature (Spearman con ESI)
#   5. Solapamiento entre niveles ESI adyacentes (Cohen d)
#   6. Cohen d ESI 1 vs ESI 2 para todas las features
#   7. Casos con vitales completamente normales
#   8. Analisis de chief complaints
#   9. Techo teorico del clasificador
#
# La limpieza aplicada es identica a la del script de entrenamiento,
# por lo que las estadisticas corresponden exactamente al dataset de entrenamiento.

import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')


def normalize_cc(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()


def cohens_d(a, b):
    pooled = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def efecto(d):
    d = abs(d)
    if d < 0.2: return "trivial"
    if d < 0.5: return "pequeno"
    if d < 0.8: return "medio"
    return "grande"


def interp_r(r):
    r = abs(r)
    if r < 0.05: return "Negligible"
    if r < 0.1:  return "Trivial"
    if r < 0.2:  return "Debil"
    if r < 0.3:  return "Moderada"
    return "Fuerte"


print("cargando datos...")
df_raw = pd.read_csv('triage_with_demographics.csv', low_memory=False)
print("filas totales:", len(df_raw))
print("columnas:", list(df_raw.columns))

for col in ['acuity', 'pain', 'temperature', 'heartrate', 'resprate', 'age', 'o2sat', 'sbp', 'dbp']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')


# --- seccion 1: missing data ---
print("\n--- missing data (antes de filtrar nada) ---")

total = len(df_raw)
for col in ['chiefcomplaint', 'acuity', 'pain', 'temperature', 'heartrate',
            'resprate', 'age', 'gender', 'o2sat', 'sbp', 'dbp']:
    if col in df_raw.columns:
        m = df_raw[col].isna().sum()
        print(f"{col}: {m} missing ({m/total*100:.1f}%)")

# missing por ESI (solo vitales)
df_esi = df_raw.dropna(subset=['acuity']).copy()
df_esi['acuity'] = df_esi['acuity'].astype(int)
df_esi = df_esi[df_esi['acuity'].between(1, 5)]

print("\nmissing por nivel ESI:")
for col in ['pain', 'temperature', 'heartrate', 'resprate', 'age', 'gender']:
    print(f"\n  {col}:")
    for esi in range(1, 6):
        sub = df_esi[df_esi['acuity'] == esi]
        m = sub[col].isna().sum()
        print(f"    ESI {esi}: {m} de {len(sub)} ({m/len(sub)*100:.1f}%)")


# --- limpieza (igual que en el entrenamiento) ---
df = df_raw.copy()
df = df.dropna(subset=['acuity'])
df['acuity'] = df['acuity'].astype(int)
df = df[df['acuity'].between(1, 5)]
df = df.dropna(subset=['chiefcomplaint'])
df['cc_lower'] = df['chiefcomplaint'].apply(normalize_cc)
df = df[df['cc_lower'].str.len() > 0]

rangos = {
    'pain':        (0.0,  10.0),
    'temperature': (95.0, 107.0),
    'heartrate':   (20.0, 300.0),
    'resprate':    (4.0,  60.0),
}
n_antes_rangos = len(df)
for col, (lo, hi) in rangos.items():
    if col in df.columns:
        df = df[~(df[col].notna() & ~df[col].between(lo, hi))]
n_rangos = n_antes_rangos - len(df)

vitals = ['pain', 'temperature', 'heartrate', 'resprate']
n_antes_nulos = len(df)
df = df[df[vitals].notna().all(axis=1)]
n_nulos = n_antes_nulos - len(df)

df['age'] = df['age'].fillna(df['age'].median()).clip(18, 91)
df['gender_M'] = df['gender'].map({"M": 1.0, "F": 0.0}).fillna(0.5)

print("\n--- resumen limpieza ---")
print("dataset crudo:", len(df_raw))
print("tras filtrar acuity y cc:", n_antes_rangos)
print("eliminados por outliers:", n_rangos)
print("eliminados por nulos en vitales:", n_nulos)
print("dataset final:", len(df))

# outliers features adicionales (se ponen a NaN pero no se eliminan filas)
rangos_kuts = {'o2sat': (50.0, 100.0), 'sbp': (40.0, 300.0), 'dbp': (20.0, 200.0)}
print("\noutliers features adicionales (se anulan a NaN, no se eliminan filas):")
for col, (lo, hi) in rangos_kuts.items():
    if col not in df.columns:
        print(f"  {col}: no disponible")
        continue
    n_out = (df[col].notna() & ~df[col].between(lo, hi)).sum()
    n_antes = df[col].isna().sum()
    df.loc[df[col].notna() & ~df[col].between(lo, hi), col] = np.nan
    n_despues = df[col].isna().sum()
    print(f"  {col}: {n_out} outliers -> NaN, nulos: {n_antes} -> {n_despues} ({n_despues/len(df)*100:.1f}%)")


# --- seccion 2: demografia ---
print("\n--- seccion 2: demografia ---")
print("total casos:", len(df))

print("\ndistribucion ESI:")
for esi in range(1, 6):
    n = (df['acuity'] == esi).sum()
    print(f"  ESI {esi}: {n} ({n/len(df)*100:.1f}%)")

esi_counts = [(df['acuity'] == e).sum() for e in range(1, 6)]
print("ratio desbalance (max/min):", round(max(esi_counts)/max(min(esi_counts),1)), ":1")

print("\ngenero:")
for g, n in df['gender'].value_counts().items():
    print(f"  {g}: {n} ({n/len(df)*100:.1f}%)")

ages = df['age'].dropna()
print("\nedad:")
print(f"  media: {ages.mean():.1f} +/- {ages.std():.1f}")
print(f"  mediana: {ages.median():.0f}  IQR: {ages.quantile(0.25):.0f} - {ages.quantile(0.75):.0f}")
print(f"  rango: {ages.min():.0f} - {ages.max():.0f}")

print("\nedad por ESI:")
for esi in range(1, 6):
    a = df[df['acuity'] == esi]['age']
    print(f"  ESI {esi}: media {a.mean():.1f}, mediana {a.median():.0f}")


# --- seccion 3: distribucion features por ESI ---
print("\n--- seccion 3: distribucion features por ESI ---")

for feat, name, unit in [
    ('pain',        'pain score',            '0-10'),
    ('temperature', 'temperatura',           'F'),
    ('heartrate',   'frecuencia cardiaca',   'bpm'),
    ('resprate',    'frecuencia respiratoria','rpm'),
    ('age',         'edad',                  'anos'),
]:
    print(f"\n{name} ({unit}):")
    print("ESI   N        media   std    mediana  Q25    Q75    min    max")
    for esi in range(1, 6):
        v = df[df['acuity'] == esi][feat].dropna()
        print(f"  {esi}   {len(v):<8} {v.mean():.1f}   {v.std():.1f}   {v.median():.1f}    {v.quantile(0.25):.1f}   {v.quantile(0.75):.1f}   {v.min():.1f}   {v.max():.1f}")

for feat in ['o2sat', 'sbp', 'dbp']:
    if feat not in df.columns:
        continue
    vt = df[feat].dropna()
    if len(vt) < 1000:
        continue
    print(f"\n{feat} (no usada en el modelo, n={len(vt)}):")
    print("ESI   N        media   mediana  Q25    Q75")
    for esi in range(1, 6):
        v = df[df['acuity'] == esi][feat].dropna()
        print(f"  {esi}   {len(v):<8} {v.mean():.1f}   {v.median():.1f}     {v.quantile(0.25):.1f}   {v.quantile(0.75):.1f}")


# --- seccion 4: correlacion spearman ---
print("\n--- seccion 4: correlacion spearman con ESI (negativo = mas urgente) ---")

features_info = {
    'age':         ('edad',           True),
    'pain':        ('dolor',          True),
    'resprate':    ('frec resp',      True),
    'heartrate':   ('frec cardiaca',  True),
    'temperature': ('temperatura',    True),
    'o2sat':       ('SpO2',           False),
    'sbp':         ('SBP',            False),
    'dbp':         ('DBP',            False),
}

results = []
for col, (name, in_v5g) in features_info.items():
    if col not in df.columns:
        continue
    valid = df[[col, 'acuity']].dropna()
    if len(valid) < 1000:
        continue
    r, p = stats.spearmanr(valid[col], valid['acuity'])
    en_modelo = "si" if in_v5g else "no (KUTS)"
    print(f"  {name}: r={r:+.4f}, |r|={abs(r):.4f}, p={p:.1e}, {interp_r(r)}, en modelo: {en_modelo}")
    results.append((name, r, abs(r), in_v5g))

v5g  = [a for _, _, a, inv5g in results if inv5g]
kuts = [a for _, _, a, inv5g in results if not inv5g]
if v5g:
    print(f"\nmedia |r| features del modelo: {np.mean(v5g):.4f}")
if kuts:
    print(f"media |r| features KUTS: {np.mean(kuts):.4f}")


# --- seccion 5: solapamiento ESI adyacentes ---
print("\n--- seccion 5: solapamiento entre ESI adyacentes (cohen d) ---")

for feat in ['pain', 'temperature', 'heartrate', 'resprate', 'age']:
    print(f"\n{feat}:")
    for a, b in [(1,2),(2,3),(3,4),(4,5)]:
        va = df[df['acuity']==a][feat].dropna().values
        vb = df[df['acuity']==b][feat].dropna().values
        d = cohens_d(va, vb)
        ra = (np.percentile(va,10), np.percentile(va,90))
        rb = (np.percentile(vb,10), np.percentile(vb,90))
        olo = max(ra[0], rb[0])
        ohi = min(ra[1], rb[1])
        tr  = max(ra[1],rb[1]) - min(ra[0],rb[0])
        overlap = max(0, (ohi-olo)/tr*100) if tr > 0 else 100
        print(f"  ESI {a} vs {b}: d={d:+.3f} ({efecto(d)}), solapamiento={overlap:.0f}%, medianas {np.median(va):.1f} vs {np.median(vb):.1f}")


# --- seccion 6: cohen d ESI 1 vs ESI 2 ---
print("\n--- seccion 6: cohen d ESI 1 vs ESI 2 ---")

esi1 = df[df['acuity'] == 1]
esi2 = df[df['acuity'] == 2]

for col, name in [('pain','dolor'), ('temperature','temperatura'), ('heartrate','frec cardiaca'),
                  ('resprate','frec resp'), ('age','edad'),
                  ('o2sat','SpO2'), ('sbp','SBP'), ('dbp','DBP')]:
    if col not in df.columns:
        continue
    v1 = esi1[col].dropna().values
    v2 = esi2[col].dropna().values
    if len(v1) < 10 or len(v2) < 10:
        continue
    d = cohens_d(v1, v2)
    en_modelo = "si" if col in ['pain','temperature','heartrate','resprate','age'] else "no (KUTS)"
    print(f"  {name}: d={d:+.3f} ({efecto(d)}), medianas {np.median(v1):.1f} vs {np.median(v2):.1f}, en modelo: {en_modelo}")


# --- seccion 7: vitales normales ---
print("\n--- seccion 7: casos con vitales normales ---")
print("definicion: temp 97.7-99.1F, hr 60-100, rr 12-18, dolor 0-4")

mask = (df['temperature'].between(97.7,99.1) &
        df['heartrate'].between(60.0,100.0) &
        df['resprate'].between(12.0,18.0) &
        df['pain'].between(0.0,4.0))

for esi in range(1, 6):
    em = df['acuity'] == esi
    n_norm = (em & mask).sum()
    n_tot = em.sum()
    print(f"  ESI {esi}: {n_norm} de {n_tot} ({n_norm/n_tot*100:.1f}%)")
print(f"  total: {mask.sum()} de {len(df)} ({mask.sum()/len(df)*100:.1f}%)")


# --- seccion 8: chief complaints ---
print("\n--- seccion 8: chief complaints ---")

print(f"total CC unicos: {df['cc_lower'].nunique()}")
for esi in range(1, 6):
    print(f"  ESI {esi}: {df[df['acuity']==esi]['cc_lower'].nunique()} unicos")

cc_per_esi = df.groupby('cc_lower')['acuity'].apply(lambda x: set(x.unique()))
print(f"CC en 4+ niveles ESI distintos: {sum(1 for s in cc_per_esi if len(s)>=4)}")
print(f"CC en 3+ niveles ESI distintos: {sum(1 for s in cc_per_esi if len(s)>=3)}")

print("\ntop 15 CC mas frecuentes:")
print("CC                                   N      ESI1   ESI2   ESI3   ESI4   ESI5   entropia")
for cc, n_total in df['cc_lower'].value_counts().head(15).items():
    dist = df[df['cc_lower']==cc]['acuity'].value_counts().sort_index()
    pcts = [f"{dist.get(e,0)/n_total*100:.1f}%" for e in range(1,6)]
    probs = np.array([dist.get(e,0)/n_total for e in range(1,6)])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    print(f"  {cc:<35} {n_total:<6} {' '.join(pcts)}  {entropy:.2f}")


# --- seccion 9: techo teorico ---
print("\n--- seccion 9: techo teorico ---")

maj = int(df['acuity'].mode()[0])
print(f"baseline siempre ESI {maj}: {(df['acuity']==maj).mean()*100:.1f}%")
probs_class = df['acuity'].value_counts(normalize=True)
print(f"baseline aleatorio estratificado: {(probs_class**2).sum()*100:.1f}%")