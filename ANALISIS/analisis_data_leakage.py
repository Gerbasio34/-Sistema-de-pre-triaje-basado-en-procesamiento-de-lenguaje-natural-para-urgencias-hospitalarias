# analisis de solapamiento train/test por paciente

# mide el riesgo de data leakage por split por visita en lugar de por paciente
# dos cotas: superior (cc+esi identicos) e inferior (cc+vitales similares+esi identicos)

# input:  triage_with_demographics.csv

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def normalize_cc(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.strip()


print("cargando datos...")
df = pd.read_csv("triage_with_demographics.csv", low_memory=False)
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
    df = df[~(df[col].notna() & ~df[col].between(lo, hi))]

df = df[df[["pain", "temperature", "heartrate", "resprate"]].notna().all(axis=1)]
total = len(df)
print(f"  {total} casos tras limpieza")


# pacientes y visitas 
print("\n--- distribucion de visitas por paciente ---")

pacientes_total = df["subject_id"].nunique()
visitas_por_paciente = df.groupby("subject_id").size()
pacientes_multi = (visitas_por_paciente > 1).sum()

print(f"pacientes unicos: {pacientes_total}")
print(f"pacientes con 2+ visitas: {pacientes_multi} ({pacientes_multi/pacientes_total*100:.1f}%)")
print(f"mediana visitas por paciente: {visitas_por_paciente.median():.0f}")
print(f"maximo visitas por paciente: {visitas_por_paciente.max()}")


# cota superior: cc + esi identicos 
print("\n--- cota superior: solapamiento por CC + ESI identicos ---")

dup_superior = df[df.duplicated(subset=["subject_id", "chiefcomplaint", "acuity"], keep=False)]
n_superior = len(dup_superior)

print(f"visitas con CC y ESI identicos del mismo paciente: {n_superior} ({n_superior/total*100:.1f}%)")
print(f"estimacion en test (20%): {int(n_superior*0.20)} visitas sobre {int(total*0.20)}")


# cota inferior: cc + vitales similares + esi identicos 
print("\n--- cota inferior: solapamiento por CC + vitales similares + ESI identicos ---")
print("intervalos: pain exacto, temp 0.5F, hr 5bpm, rr 2rpm")

df["pain_bin"] = df["pain"].round(0)
df["temp_bin"] = (df["temperature"] * 2).round(0)
df["hr_bin"]   = (df["heartrate"] / 5).round(0)
df["rr_bin"]   = (df["resprate"] / 2).round(0)

cols_inferior = ["subject_id", "chiefcomplaint", "pain_bin", "temp_bin", "hr_bin", "rr_bin", "acuity"]
dup_inferior = df[df.duplicated(subset=cols_inferior, keep=False)]
n_inferior = len(dup_inferior)

print(f"visitas con CC + vitales similares + ESI identicos del mismo paciente: {n_inferior} ({n_inferior/total*100:.1f}%)")
print(f"estimacion en test (20%): {int(n_inferior*0.20)} visitas sobre {int(total*0.20)}")


