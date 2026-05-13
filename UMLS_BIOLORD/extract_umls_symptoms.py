# extract_umls_symptoms.py
#
# Filtra MRCONSO.RRF y MRSTY.RRF del metatesauro UMLS para extraer
# conceptos de sintomas, signos y enfermedades relevantes para urgencias,
# junto con todos sus sinonimos en ingles.
#
# El resultado se usa como vocabulario de referencia para el autocomplete
# (indexado por BioLORD) y para generar pares de sinonimos en el
# contrastive loss del fine-tuning de SapBERT.
#
# Inputs:  MRCONSO.RRF, MRSTY.RRF (metatesauro UMLS 2025AB)
# Output:  umls_symptoms.csv  (CUI, canonical_name, synonyms, semantic_type, n_synonyms)
#
# Uso: python extract_umls_symptoms.py

import csv
import os
from collections import defaultdict

mrconso_path = "/home/german/Descargas/umls-2025AB-mrconso/2025AB/META/MRCONSO.RRF"
mrsty_path   = "/home/german/Descargas/umls-2025AB-metathesaurus-level-0/2025AB/META/MRSTY.RRF"
output_path  = "umls_symptoms.csv"

# tipos semanticos relevantes para urgencias
target_types = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T047",  # Disease or Syndrome
    "T037",  # Injury or Poisoning
    "T048",  # Mental or Behavioral Dysfunction
    "T046",  # Pathologic Function
    "T190",  # Anatomical Abnormality
    "T019",  # Congenital Abnormality
}


# paso 1: leer MRSTY para obtener CUIs con tipos relevantes
# formato: CUI|TUI|STN|STY|ATUI|CVF|
print("leyendo MRSTY.RRF...")
target_cuis = {}
with open(mrsty_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        cui = parts[0]
        tui = parts[1]
        sty = parts[3]
        if tui in target_types:
            target_cuis[cui] = sty

print(f"  CUIs con tipos relevantes: {len(target_cuis)}")


# paso 2: leer MRCONSO para extraer nombres en ingles de esos CUIs
# formato: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
print("leyendo MRCONSO.RRF...")
concepts = {}

with open(mrconso_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        cui      = parts[0]
        lat      = parts[1]
        ts       = parts[2]
        ispref   = parts[6]
        name     = parts[14]
        suppress = parts[16]

        if lat != "ENG" or suppress == "Y" or cui not in target_cuis:
            continue

        name_clean = name.strip()
        if not name_clean or len(name_clean) < 2:
            continue

        # quitar qualifiers clinicos que no aportan al autocomplete
        if any(x in name_clean for x in ['(finding)', '(disorder)', '(morphologic abnormality)',
                                          '(situation)', '(event)', '[Ambiguous]', 'NOS']):
            for suffix in ['(finding)', '(disorder)', '(morphologic abnormality)',
                           '(situation)', '(event)']:
                name_clean = name_clean.replace(suffix, '').strip()
            if not name_clean:
                continue

        if cui not in concepts:
            concepts[cui] = {"preferred": None, "synonyms": set()}

        if ts == "P" and ispref == "Y":
            concepts[cui]["preferred"] = name_clean

        concepts[cui]["synonyms"].add(name_clean)

print(f"  conceptos unicos: {len(concepts)}")


# paso 3: limpiar y exportar
print(f"exportando a {output_path}...")
exported = 0

with open(output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['CUI', 'canonical_name', 'synonyms', 'semantic_type', 'n_synonyms'])

    for cui, data in sorted(concepts.items()):
        canonical = data["preferred"]
        if not canonical:
            if data["synonyms"]:
                canonical = min(data["synonyms"], key=len)
            else:
                continue

        # quitar sinonimos identicos al canonical y los demasiado largos
        synonyms = {s for s in data["synonyms"] if s.lower() != canonical.lower()}
        synonyms = {s for s in synonyms if len(s) <= 100}

        if len(canonical) > 100:
            continue

        sty = target_cuis.get(cui, "Unknown")
        synonyms_str = "|".join(sorted(synonyms))

        writer.writerow([cui, canonical, synonyms_str, sty, len(synonyms) + 1])
        exported += 1

print(f"  conceptos exportados: {exported}")

print(f"\ncompletado: {output_path}")