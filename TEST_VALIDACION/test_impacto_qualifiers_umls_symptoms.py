# test_impacto_qualifiers_umls_symptoms.py
#
# analiza el impacto de los qualifiers residuales del vocabulario UMLS
# en el autocomplete de BioLORD. para cada nombre con qualifier de ruido,
# comprueba si su version limpia existe en el indice global.

import csv
import re
import unicodedata
import csv
import re
import unicodedata

csv_path = "../ANALISIS/umls_symptoms.csv"

noise_qualifiers = {
    'diagnosis', 'physical finding', 'observable entity',
    'context-dependent category', 'symptom', 'manifestation',
    'qualifier value', 'etiology', 'procedure', 'nos',
    'or disorder', 'navigational concept', 'lab test'
}

def normalize(text):
    # minusculas
    text = text.lower().strip()
    # normalizar unicode (quitar tildes si las hubiera)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # colapsar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_noise_qualifiers(text):
    def remove_noise(m):
        content = m.group(1).lower().strip()
        return '' if content in noise_qualifiers else m.group(0)
    text = re.sub(r'\s*\(([^)]+)\)', remove_noise, text)
    return re.sub(r'\s+', ' ', text).strip()


# paso 1: construir set SOLO con nombres que no tienen qualifier
print("construyendo set de nombres limpios (sin qualifier)...")
clean_names_set = set()

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        cui, canonical, synonyms_str, sty, n = row

        all_names = [canonical]
        if synonyms_str:
            all_names += synonyms_str.split('|')

        for name in all_names:
            # solo metemos los que no tienen ningun qualifier
            if not re.search(r'\([^)]+\)', name):
                clean_names_set.add(normalize(name))

print(f"  {len(clean_names_set)} nombres limpios unicos en el set")


# paso 2: para cada nombre sucio, quitar qualifiers de ruido,
# normalizar y ver si existe en el set de limpios
print("analizando nombres sucios...")

total_sucios = 0
con_version_limpia = 0
sin_version_limpia = 0
ejemplos_sin_limpia = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        cui, canonical, synonyms_str, sty, n = row

        all_names = [canonical]
        if synonyms_str:
            all_names += synonyms_str.split('|')

        for name in all_names:
            if not re.search(r'\([^)]+\)', name):
                continue
            
            qualifiers_found = re.findall(r'\(([^)]+)\)', name)
            has_noise = any(q.lower().strip() in noise_qualifiers for q in qualifiers_found)
            if not has_noise:
                continue

            total_sucios += 1

            # quitar qualifiers de ruido y normalizar
            cleaned = remove_noise_qualifiers(name)
            normalized = normalize(cleaned)

            if not normalized:
                sin_version_limpia += 1
                continue

            if normalized in clean_names_set:
                con_version_limpia += 1
            else:
                sin_version_limpia += 1
                if len(ejemplos_sin_limpia) < 20:
                    ejemplos_sin_limpia.append((cui, name, normalized))

print(f"\ntotal nombres con qualifier: {total_sucios}")
print(f"con version limpia en el set: {con_version_limpia} ({con_version_limpia/total_sucios*100:.1f}%)")
print(f"sin version limpia en el set: {sin_version_limpia} ({sin_version_limpia/total_sucios*100:.1f}%)")

print(f"\nejemplos sin version limpia (primeros 20):")
for cui, sucio, norm in ejemplos_sin_limpia:
    print(f"  '{sucio}' -> normalizado: '{norm}'")