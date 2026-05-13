# merge_triaje_patients.py
#
# Une triage.csv (MIMIC-IV-ED) con age y gender de patients.csv (MIMIC-IV).
# Output: triage_with_demographics.csv

import pandas as pd

patients = pd.read_csv("patients.csv")
triage   = pd.read_csv("triage.csv", low_memory=False)

merged = triage.merge(patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')
merged = merged.rename(columns={'anchor_age': 'age'})

print(f"Total filas: {len(merged):,}")
print(f"Con edad:    {merged['age'].notna().sum():,} ({merged['age'].notna().mean()*100:.1f}%)")
print(f"Con genero:  {merged['gender'].notna().sum():,} ({merged['gender'].notna().mean()*100:.1f}%)")

merged.to_csv("triage_with_demographics.csv", index=False)
print(f"\nGuardado: triage_with_demographics.csv")
print(f"Columnas: {list(merged.columns)}")
