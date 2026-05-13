# Analisis comparativo MC-MED vs MIMIC para verificar hipotesis de domain shift.
# Compara distribucion de vitales, chief complaints y poder discriminativo
# entre el dataset de entrenamiento (MIMIC) y el de validacion externa (MC-MED).
#
# Input:  visits.csv (MC-MED, Stanford Health Care 2020-2022)
# Output: redirigir con > analisis_mcmed.txt

import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')


print("ANALISIS MC-MED -- Stanford Health Care 2020-2022")

df_raw = pd.read_csv('visits.csv', low_memory=False)
print(f"\ndataset crudo: {len(df_raw)} filas")

for col in ['Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2', 'Triage_SBP', 'Triage_DBP', 'Age']:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# ESI viene como "3-Urgent", extraemos el numero
df_raw['esi'] = df_raw['Triage_acuity'].astype(str).str.extract(r'(\d)').astype(float)
df_raw = df_raw.dropna(subset=['esi', 'CC'])
df_raw['esi'] = df_raw['esi'].astype(int)
df_raw = df_raw[df_raw['esi'].between(1, 5)]

# temperatura viene en Celsius, convertimos a Fahrenheit para comparar con MIMIC
df_raw['temp_F'] = df_raw['Triage_Temp'] * 9/5 + 32

rangos = {'temp_F': (95.0, 107.0), 'Triage_HR': (20.0, 300.0), 'Triage_RR': (4.0, 60.0)}
n_antes = len(df_raw)
for col, (lo, hi) in rangos.items():
    df_raw = df_raw[~(df_raw[col].notna() & ~df_raw[col].between(lo, hi))]

vitals = ['temp_F', 'Triage_HR', 'Triage_RR']
n_antes_nulos = len(df_raw)
df = df_raw.dropna(subset=vitals + ['CC']).copy()

print(f"outliers eliminados: {n_antes - n_antes_nulos}")
print(f"nulos en vitales eliminados: {n_antes_nulos - len(df)}")
print(f"dataset valido: {len(df)}")

df['cc'] = df['CC'].astype(str).str.lower().str.strip()
df['cc'] = df['cc'].apply(lambda x: re.sub(r'\s+', ' ', x))
df['age'] = df['Age'].clip(18, 91)
df['gender_M'] = df['Gender'].map({'M': 1.0, 'Male': 1.0, 'F': 0.0, 'Female': 0.0}).fillna(0.5)


# --- seccion 1: demografia y distribucion ESI ---
print("\n--- seccion 1: demografia y distribucion ESI ---")

mimic_pct = {1: 3.1, 2: 32.8, 3: 56.4, 4: 7.3, 5: 0.3}
print("\ndistribucion ESI (MC-MED vs MIMIC):")
print("ESI   N MC-MED    % MC-MED   % MIMIC")
for esi in range(1, 6):
    n = (df['esi'] == esi).sum()
    print(f"  {esi}   {n}    {n/len(df)*100:.1f}%   {mimic_pct[esi]:.1f}%")

print(f"\nedad:")
print(f"  MC-MED -- media: {df['age'].mean():.1f}  mediana: {df['age'].median():.0f}  IQR: {df['age'].quantile(0.25):.0f}-{df['age'].quantile(0.75):.0f}")
print(f"  MIMIC  -- media: 49.5  mediana: 50  IQR: 32-65")

mimic_age = {1: 55.2, 2: 54.9, 3: 47.5, 4: 38.4, 5: 37.4}
print(f"\nedad media por ESI (MC-MED vs MIMIC):")
for esi in range(1, 6):
    ages = df[df['esi'] == esi]['age'].dropna()
    print(f"  ESI {esi}: MC-MED {ages.mean():.1f}  MIMIC {mimic_age[esi]:.1f}")

print(f"\ngenero:")
for g, n in df['Gender'].value_counts().items():
    print(f"  {g}: {n} ({n/len(df)*100:.1f}%)")


# --- seccion 2: distribucion de vitales ---
print("\n--- seccion 2: distribucion de vitales (MC-MED vs MIMIC) ---")

# medianas de MIMIC por ESI para comparar
mimic_temp = {1: 98.1, 2: 98.0, 3: 98.0, 4: 98.0, 5: 97.9}  # medianas
mimic_hr   = {1: 91.0, 2: 84.0, 3: 83.0, 4: 81.0, 5: 80.0}  # medianas
mimic_rr   = {1: 18.0, 2: 18.0, 3: 18.0, 4: 16.0, 5: 16.0}  # medianas

for col, name, mimic_meds in [
    ('temp_F',    'temperatura (F)',              mimic_temp),
    ('Triage_HR', 'frecuencia cardiaca (bpm)',    mimic_hr),
    ('Triage_RR', 'frecuencia respiratoria (rpm)', mimic_rr),
]:
    print(f"\n{name}:")
    print("ESI   N        media   mediana   std    MIMIC mediana")
    for esi in range(1, 6):
        v = df[df['esi'] == esi][col].dropna()
        if len(v) > 0:
            print(f"  {esi}   {len(v):<8} {v.mean():.1f}   {v.median():.1f}      {v.std():.1f}   {mimic_meds[esi]:.1f}")


# --- seccion 3: chief complaints ---
print("\n--- seccion 3: chief complaints ---")

print(f"\nvocabulario unico:")
print(f"  MC-MED: {df['cc'].nunique()} CCs unicos sobre {len(df)} casos")
print(f"  MIMIC:  51778 CCs unicos sobre 374306 casos")

print(f"\ntop 20 CCs mas frecuentes en MC-MED:")
print("CC                                       N      ESI1   ESI2   ESI3   ESI4   ESI5   entropia")
for cc, n_total in df['cc'].value_counts().head(20).items():
    dist = df[df['cc'] == cc]['esi'].value_counts().sort_index()
    pcts = [f"{dist.get(e,0)/n_total*100:.1f}%" for e in range(1, 6)]
    probs = np.array([dist.get(e,0)/n_total for e in range(1, 6)])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    print(f"  {cc:<39} {n_total:<6} {' '.join(pcts)}  {entropy:.2f}")


# --- seccion 4: poder discriminativo vitales MC-MED vs MIMIC ---
print("\n--- seccion 4: poder discriminativo vitales MC-MED vs MIMIC ---")

mimic_corrs = {'age': -0.2332, 'Triage_HR': -0.0548, 'Triage_RR': -0.1212, 'temp_F': -0.0212}
mimic_names = {'age': 'age', 'Triage_HR': 'heartrate', 'Triage_RR': 'resprate', 'temp_F': 'temperature'}

print("\ncorrelacion Spearman con ESI:")
for col in ['age', 'Triage_HR', 'Triage_RR', 'temp_F']:
    valid = df[[col, 'esi']].dropna()
    r, _ = stats.spearmanr(valid[col], valid['esi'])
    absr = abs(r)
    if absr < 0.05:   interp = "Negligible"
    elif absr < 0.1:  interp = "Trivial"
    elif absr < 0.2:  interp = "Debil"
    elif absr < 0.3:  interp = "Moderada"
    else:             interp = "Fuerte"
    print(f"  {mimic_names[col]}: r MC-MED={r:+.4f}, r MIMIC={mimic_corrs[col]:+.4f}, {interp}")


# --- seccion 6: casos con vitales normales ---
print("\n--- seccion 6: casos con vitales normales ---")
print("definicion: temp 97-99F, hr 60-100 bpm, rr 14-20 rpm (sin pain en MC-MED)")

mask = (df['temp_F'].between(97.0, 99.0) &
        df['Triage_HR'].between(60.0, 100.0) &
        df['Triage_RR'].between(14.0, 20.0))

mimic_normal_pct = {1: 24.4, 2: 36.2, 3: 28.0, 4: 31.1, 5: 55.7}
print("\nESI   con vitales normales   total    %      MIMIC %")
for esi in range(1, 6):
    em = df['esi'] == esi
    n_norm = (em & mask).sum()
    n_tot = em.sum()
    pct = n_norm / n_tot * 100 if n_tot > 0 else 0
    print(f"  {esi}   {n_norm:<20} {n_tot:<8} {pct:.1f}%   {mimic_normal_pct[esi]:.1f}%")
print(f"  total  {mask.sum():<20} {len(df):<8} {mask.sum()/len(df)*100:.1f}%   30.9%")