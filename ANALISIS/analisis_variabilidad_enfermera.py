# Analisis de variabilidad inter-enfermera en MIMIC-IV-ED
# 
# Busca evidencia de que el mismo paciente (mismo chief complaint y vitales similares)
# puede recibir niveles ESI distintos segun quien haga el triaje.
# Esto demuestra que los labels del dataset tienen ruido inherente,
# lo que pone un techo estadistico al rendimiento de cualquier modelo.
#
# Input:  triage_with_demographics.csv
# Output: redirigir con > analisis_variabilidad_enfermera.txt
#
# Secciones:
#   1. CC que aparecen en multiples niveles ESI
#   2. Casos con CC identico y vitales similares pero ESI distinto
#   3. Estimacion del ruido de label
#   4. Frontera ESI 3/4 con vitales normales
#
# La limpieza aplicada es identica a la del script de entrenamiento.

import pandas as pd
import numpy as np
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
    if col in df.columns:
        df = df[~(df[col].notna() & ~df[col].between(lo, hi))]

df = df[df[["pain", "temperature", "heartrate", "resprate"]].notna().all(axis=1)]
print(f"  {len(df)} casos tras limpieza")


# --- seccion 1: CC en multiples niveles ESI ---
print("\n--- seccion 1: chief complaints en multiples niveles ESI ---")

# para cada CC calculamos cuantos niveles distintos tiene y su entropia
cc_stats = {}
for cc, grupo in df.groupby("chiefcomplaint")["acuity"]:
    niveles = sorted(grupo.unique().tolist())
    n = len(grupo)
    probs = grupo.value_counts(normalize=True).values
    entropia = -sum(p * np.log2(p) for p in probs if p > 0)
    cc_stats[cc] = {
        "n_casos": n,
        "n_niveles": len(niveles),
        "niveles": niveles,
        "entropia": entropia,
    }

cc_df = pd.DataFrame(cc_stats).T
total_cc = len(cc_df)

print(f"\ntotal CC unicos: {total_cc}")
print(f"CC en 1 solo nivel ESI: {(cc_df['n_niveles']==1).sum()} ({(cc_df['n_niveles']==1).sum()/total_cc*100:.1f}%)")
print(f"CC en 2+ niveles ESI: {(cc_df['n_niveles']>=2).sum()} ({(cc_df['n_niveles']>=2).sum()/total_cc*100:.1f}%)")
print(f"CC en 3+ niveles ESI: {(cc_df['n_niveles']>=3).sum()} ({(cc_df['n_niveles']>=3).sum()/total_cc*100:.1f}%)")
print(f"CC en 4+ niveles ESI: {(cc_df['n_niveles']>=4).sum()} ({(cc_df['n_niveles']>=4).sum()/total_cc*100:.1f}%)")

cc_multi = cc_df[cc_df["n_niveles"] > 1]
casos_multi = df[df["chiefcomplaint"].isin(cc_multi.index)].shape[0]
print(f"\ncasos con CC multi-nivel: {casos_multi} ({casos_multi/len(df)*100:.1f}% del total)")

print("\ntop 20 CC con mayor variabilidad ESI (mas casos):")
print("Chief Complaint                      N       Niveles  Distribucion ESI                Entropia")
top20 = cc_multi.sort_values("n_casos", ascending=False).head(20)
for cc, row in top20.iterrows():
    dist = df[df["chiefcomplaint"] == cc]["acuity"].value_counts().sort_index()
    dist_str = " ".join([f"ESI{k}:{v}" for k, v in dist.items()])
    cc_short = cc[:34] if len(cc) > 34 else cc
    print(f"  {cc_short:<34} {int(row['n_casos']):>7} {str(row['niveles']):>8}  {dist_str:<30}  {row['entropia']:.3f}")


# --- seccion 2: CC identico + vitales similares + ESI distinto ---
print("\n--- seccion 2: casos con CC identico + vitales similares + ESI distinto ---")
print("tolerancias: pain=exacto, temp bins 0.5F, hr bins 5bpm, rr bins 2rpm")

df_bin = df.copy()
df_bin["pain_bin"] = df_bin["pain"].round(0)
df_bin["temp_bin"] = (df_bin["temperature"] * 2).round(0)   # bins de 0.5°F
df_bin["hr_bin"]   = (df_bin["heartrate"] / 5).round(0)     # bins de 5 lpm
df_bin["rr_bin"]   = (df_bin["resprate"] / 2).round(0)      # bins de 2 rpm

group_cols = ["chiefcomplaint", "pain_bin", "temp_bin", "hr_bin", "rr_bin"]

# calculamos estadisticos por grupo manualmente para evitar lambdas complejas
grupos_data = {}
for key, grupo in df_bin.groupby(group_cols)["acuity"]:
    niveles = sorted(grupo.unique().tolist())
    grupos_data[key] = {
        "n_casos": len(grupo),
        "n_niveles": len(niveles),
        "niveles": niveles,
        "esi_min": grupo.min(),
        "esi_max": grupo.max(),
    }

grupos = pd.DataFrame(grupos_data).T

grupos_multi        = grupos[grupos["n_niveles"] > 1]
grupos_discordantes = grupos[(grupos["n_niveles"] > 1) & (grupos["n_casos"] >= 3)]

print(f"\ntotal grupos (CC + vitales similares): {len(grupos)}")
print(f"grupos con ESI uniforme: {(grupos['n_niveles']==1).sum()} ({(grupos['n_niveles']==1).sum()/len(grupos)*100:.1f}%)")
print(f"grupos con ESI discordante (2+ niv.): {len(grupos_multi)} ({len(grupos_multi)/len(grupos)*100:.1f}%)")
print(f"grupos discordantes con >=3 casos: {len(grupos_discordantes)}")

casos_discordantes = int(grupos_multi["n_casos"].sum())
print(f"\ncasos en grupos discordantes: {casos_discordantes} ({casos_discordantes/len(df)*100:.1f}% del dataset)")

grupos_2_3 = grupos_multi[
    grupos_multi["niveles"].apply(lambda x: 1 in x or 2 in x) &
    grupos_multi["niveles"].apply(lambda x: 3 in x or 4 in x)
]
print(f"grupos con discordancia ESI 1-2 vs 3-4: {len(grupos_2_3)}")

print("\nejemplos de grupos discordantes (>=5 casos, mayor variabilidad):")
print("Chief Complaint                 Pain   Temp    HR    RR     N   Niveles ESI")
top_discord = grupos_discordantes.sort_values("n_casos", ascending=False).head(25)
for (cc, pain, temp, hr, rr), row in top_discord.iterrows():
    cc_short = cc[:29] if len(cc) > 29 else cc
    print(f"  {cc_short:<29} {float(pain):>5.0f} {float(temp)/2:>6.1f} {float(hr)*5:>5.0f} {float(rr)*2:>5.0f} "
          f"{int(row['n_casos']):>5}  {str(row['niveles'])}")


# --- seccion 3: estimacion del ruido de label ---
print("\n--- seccion 3: estimacion del ruido de label ---")

ruido_por_grupo = []
for key, row in grupos_multi.iterrows():
    if row["n_casos"] < 3:
        continue
    cc, pain, temp, hr, rr = key
    subset = df_bin[
        (df_bin["chiefcomplaint"] == cc) &
        (df_bin["pain_bin"] == pain) &
        (df_bin["temp_bin"] == temp) &
        (df_bin["hr_bin"] == hr) &
        (df_bin["rr_bin"] == rr)
    ]["acuity"]
    counts = subset.value_counts()
    mayoritario = counts.iloc[0]
    minoritario = int(row["n_casos"]) - mayoritario
    ruido_por_grupo.append(minoritario / int(row["n_casos"]))

# casos totales en grupos discordantes con >=3 casos
casos_ruido = 0
for key, row in grupos_multi.iterrows():
    if row["n_casos"] < 3:
        continue
    casos_ruido += int(row["n_casos"])
print(f"casos totales en grupos discordantes con >=3 casos: {casos_ruido} ({casos_ruido/len(df)*100:.1f}% del dataset)")

if ruido_por_grupo:
    ruido_medio   = np.mean(ruido_por_grupo)
    ruido_mediana = np.median(ruido_por_grupo)
    print(f"\nen grupos discordantes con >=3 casos y vitales similares:")
    print(f"  fraccion media de casos con ESI no mayoritario: {ruido_medio:.1%}")
    print(f"  fraccion mediana de casos con ESI no mayoritario: {ruido_mediana:.1%}")
    print(f"\n  en un grupo de casos con mismo CC y vitales similares,")
    print(f"  de media el {ruido_medio:.1%} de los casos tiene un ESI distinto al mayoritario.")
    print(f"  numero de grupos analizados: {len(ruido_por_grupo)}")

# --- seccion 4: frontera ESI 3/4 con vitales normales ---
print("\n--- seccion 4: frontera ESI 3/4 con vitales normales ---")

vitales_normales = (
    (df["pain"] <= 3) &
    (df["temperature"].between(97.0, 99.0)) &
    (df["heartrate"].between(60, 100)) &
    (df["resprate"].between(12, 20))
)

esi3_normal  = df[vitales_normales & (df["acuity"] == 3)]
esi4_normal  = df[vitales_normales & (df["acuity"] == 4)]
total_normal = df[vitales_normales]

print(f"\ncasos con vitales completamente normales: {len(total_normal)} ({len(total_normal)/len(df)*100:.1f}%)")
print(f"  de ellos, ESI 3: {len(esi3_normal)} ({len(esi3_normal)/len(total_normal)*100:.1f}%)")
print(f"  de ellos, ESI 4: {len(esi4_normal)} ({len(esi4_normal)/len(total_normal)*100:.1f}%)")

cc_esi3 = set(esi3_normal["chiefcomplaint"].unique())
cc_esi4 = set(esi4_normal["chiefcomplaint"].unique())
cc_ambiguos = cc_esi3 & cc_esi4
print(f"\nCC que aparecen en ESI 3 y ESI 4 con vitales normales: {len(cc_ambiguos)}")

cc_ambiguos_list = []
for cc in cc_ambiguos:
    n3 = esi3_normal[esi3_normal["chiefcomplaint"] == cc].shape[0]
    n4 = esi4_normal[esi4_normal["chiefcomplaint"] == cc].shape[0]
    cc_ambiguos_list.append((cc, n3, n4, n3+n4))

cc_ambiguos_list.sort(key=lambda x: -x[3])
print("\ntop 15 CC mas frecuentes entre ESI 3 y ESI 4 con vitales normales:")
print("Chief Complaint                      ESI3    ESI4   Total")
for cc, n3, n4, total in cc_ambiguos_list[:15]:
    cc_short = cc[:34] if len(cc) > 34 else cc
    print(f"  {cc_short:<34} {n3:>6} {n4:>6} {total:>7}")

# --- seccion 5: frontera ESI 2/3 en grupos discordantes ---
print("\n--- seccion 5: frontera ESI 2/3 en grupos discordantes ---")

grupos_frontera_23 = []
for key, row in grupos_multi.iterrows():
    niveles = row["niveles"]
    if 2 in niveles and 3 in niveles:
        grupos_frontera_23.append((key, row))

print(f"\ngrupos con discordancia ESI 2/3: {len(grupos_frontera_23)}")

resultados_23 = []
total_casos_23 = 0
for (cc, pain, temp, hr, rr), row in grupos_frontera_23:
    subset = df_bin[
        (df_bin["chiefcomplaint"] == cc) &
        (df_bin["pain_bin"] == pain) &
        (df_bin["temp_bin"] == temp) &
        (df_bin["hr_bin"] == hr) &
        (df_bin["rr_bin"] == rr)
    ]["acuity"]
    n2 = int((subset == 2).sum())
    n3 = int((subset == 3).sum())
    n_total = int(row["n_casos"])
    total_casos_23 += n_total
    resultados_23.append((cc, float(pain), float(temp)/2, float(hr)*5, float(rr)*2, n2, n3, n_total))

print(f"casos en esos grupos: {total_casos_23} ({total_casos_23/len(df)*100:.1f}% del dataset)")

resultados_23.sort(key=lambda x: -x[7])
print("\ntop 15 grupos con discordancia ESI 2/3 (mayor numero de casos):")
print("chief complaint, pain, temp, hr, rr, esi2, esi3, n_total")
for cc, pain, temp, hr, rr, n2, n3, total in resultados_23[:15]:
    cc_short = cc[:29] if len(cc) > 29 else cc
    print(f"  {cc_short:<29} {pain:>5.0f} {temp:>6.1f} {hr:>5.0f} {rr:>5.0f} {n2:>6} {n3:>6} {total:>6}")