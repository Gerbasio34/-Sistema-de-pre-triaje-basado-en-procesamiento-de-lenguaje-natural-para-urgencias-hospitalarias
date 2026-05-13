# analisis del vocabulario umls_symptoms.csv
# generado por extract_umls_symptoms.py a partir de MRCONSO.RRF y MRSTY.RRF
#
# examina la calidad del vocabulario desde dos perspectivas:
#   1. autocomplete: que ve el paciente en la lista de sugerencias
#   2. contrastive loss: calidad de los pares de sinonimos usados en el fine-tuning
#
# output: redirigir con > analisis_umls_symptoms.txt

import csv
import re
from collections import Counter, defaultdict

csv_path = "umls_symptoms.csv"


# carga completa del csv
print("cargando umls_symptoms.csv...")
concepts = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        syns = row['synonyms'].split('|') if row['synonyms'] else []
        concepts.append({
            'cui':      row['CUI'],
            'canonical': row['canonical_name'],
            'synonyms': syns,
            'sty':      row['semantic_type'],
            'n':        int(row['n_synonyms']),
        })

print(f"  {len(concepts)} conceptos cargados")


# --- seccion 1: estadisticas globales ---
print("\n--- seccion 1: estadisticas globales ---")

total_concepts = len(concepts)
total_synonyms = sum(len(c['synonyms']) for c in concepts)
total_names = sum(c['n'] for c in concepts)  # canonico + sinonimos

print(f"\ntotal conceptos (CUIs): {total_concepts}")
print(f"total sinonimos (sin canonico): {total_synonyms}")
print(f"total nombres incluyendo canonico: {total_names}")
print(f"media sinonimos por concepto: {total_synonyms/total_concepts:.1f}")

n_counts = Counter(c['n'] for c in concepts)
print(f"\ndistribucion por numero de nombres (canonico + sinonimos):")
for n in sorted(n_counts):
    if n <= 10 or n in [20, 30, 50]:
        print(f"  {n} nombres: {n_counts[n]} conceptos ({n_counts[n]/total_concepts*100:.1f}%)")
print(f"  >10 nombres: {sum(v for k,v in n_counts.items() if k > 10)} conceptos ({sum(v for k,v in n_counts.items() if k > 10)/total_concepts*100:.1f}%)")

print(f"\ntop 15 conceptos con mas sinonimos:")
sorted_concepts = sorted(concepts, key=lambda x: x['n'], reverse=True)
for c in sorted_concepts[:15]:
    print(f"  {c['cui']}  {c['n']} nombres  {c['canonical']}")


# --- seccion 2: distribucion por tipo semantico ---
print("\n--- seccion 2: distribucion por tipo semantico ---")

sty_counts = Counter(c['sty'] for c in concepts)
print(f"\ndistribucion por tipo semantico:")
for sty, count in sty_counts.most_common():
    print(f"  {sty}: {count} ({count/total_concepts*100:.1f}%)")



# --- seccion 3: longitud de nombres ---
print("\n--- seccion 3: longitud de nombres ---")

canonical_lengths = [len(c['canonical'].split()) for c in concepts]
synonym_lengths = [len(s.split()) for c in concepts for s in c['synonyms']]

print(f"\nlongitud canonicos (palabras):")
print(f"  media: {sum(canonical_lengths)/len(canonical_lengths):.1f}")
print(f"  mediana: {sorted(canonical_lengths)[len(canonical_lengths)//2]}")
print(f"  max: {max(canonical_lengths)}")
print(f"  con 1 palabra: {sum(1 for l in canonical_lengths if l == 1)} ({sum(1 for l in canonical_lengths if l == 1)/len(canonical_lengths)*100:.1f}%)")
print(f"  con 2-3 palabras: {sum(1 for l in canonical_lengths if 2 <= l <= 3)} ({sum(1 for l in canonical_lengths if 2 <= l <= 3)/len(canonical_lengths)*100:.1f}%)")
print(f"  con 4+ palabras: {sum(1 for l in canonical_lengths if l >= 4)} ({sum(1 for l in canonical_lengths if l >= 4)/len(canonical_lengths)*100:.1f}%)")

print(f"\nlongitud sinonimos (palabras):")
print(f"  media: {sum(synonym_lengths)/len(synonym_lengths):.1f}")
print(f"  mediana: {sorted(synonym_lengths)[len(synonym_lengths)//2]}")
print(f"  max: {max(synonym_lengths)}")


# --- seccion 4: qualifiers residuales ---
print("\n--- seccion 4: qualifiers residuales ---")

canonical_with_q = 0
synonyms_with_q = 0
qualifier_in_canonical = Counter()
qualifier_in_synonyms = Counter()

for c in concepts:
    matches = re.findall(r'\([^)]+\)', c['canonical'])
    if matches:
        canonical_with_q += 1
        for m in matches:
            qualifier_in_canonical[m.lower()] += 1
    for s in c['synonyms']:
        matches = re.findall(r'\([^)]+\)', s)
        if matches:
            synonyms_with_q += 1
            for m in matches:
                qualifier_in_synonyms[m.lower()] += 1

print(f"\nnombres canonicos con qualifier: {canonical_with_q} ({canonical_with_q/total_concepts*100:.1f}%)")
print(f"sinonimos con qualifier: {synonyms_with_q} ({synonyms_with_q/total_synonyms*100:.1f}%)")

print(f"\ntop 20 qualifiers en canonicos:")
for pat, count in qualifier_in_canonical.most_common(20):
    print(f"  {count}  {pat}")

print(f"\ntop 20 qualifiers en sinonimos:")
for pat, count in qualifier_in_synonyms.most_common(20):
    print(f"  {count}  {pat}")


# --- seccion 5: impacto en autocomplete ---
print("\n--- seccion 5: impacto en autocomplete ---")

# para cada canonico sucio, ver si existe version limpia como canonico de otro CUI
canonical_index = defaultdict(list)
for c in concepts:
    canonical_index[c['canonical'].lower().strip()].append(c['cui'])

duplicados_reales = 0
for c in concepts:
    if not re.search(r'\([^)]+\)', c['canonical']):
        continue
    clean = re.sub(r'\s*\([^)]+\)', '', c['canonical']).strip().lower()
    if not clean:
        continue
    if clean in canonical_index:
        clean_cuis = canonical_index[clean]
        if set([c['cui']]) != set(clean_cuis):
            duplicados_reales += 1

print(f"\ncanonicos con qualifier: {canonical_with_q} ({canonical_with_q/total_concepts*100:.1f}%)")
print(f"de esos, con version limpia como canonico de otro CUI (duplicado real): {duplicados_reales} ({duplicados_reales/canonical_with_q*100:.1f}%)")
print(f"de esos, sin duplicado real en el indice de canonicos: {canonical_with_q - duplicados_reales} ({(canonical_with_q-duplicados_reales)/canonical_with_q*100:.1f}%)")

# --- seccion 6: impacto en contrastive loss ---
print("\n--- seccion 6: impacto en contrastive loss ---")

cuis_en_contrastive = 0
cuis_con_alguno_sucio = 0
total_syns_contrastive = 0
total_sucios_contrastive = 0
sucios_con_limpia = 0
sucios_sin_limpia = 0

for c in concepts:
    all_names = [c['canonical']] + c['synonyms']
    if len(all_names) < 2:
        continue

    cuis_en_contrastive += 1
    total_syns_contrastive += len(all_names)

    sucios = [s for s in all_names if re.search(r'\([^)]+\)', s)]
    if sucios:
        cuis_con_alguno_sucio += 1
        total_sucios_contrastive += len(sucios)

        all_lower = [n.lower().strip() for n in all_names]
        for s in sucios:
            clean = re.sub(r'\s*\([^)]+\)', '', s).strip().lower()
            if clean and clean in all_lower:
                sucios_con_limpia += 1
            else:
                sucios_sin_limpia += 1

print(f"\nCUIs que participan en contrastive (>=2 nombres): {cuis_en_contrastive}")
print(f"de esos, con al menos 1 nombre sucio: {cuis_con_alguno_sucio} ({cuis_con_alguno_sucio/cuis_en_contrastive*100:.1f}%)")
print(f"total nombres en CUIs que participan: {total_syns_contrastive}")
print(f"de esos, nombres sucios: {total_sucios_contrastive} ({total_sucios_contrastive/total_syns_contrastive*100:.1f}%)")
print(f"\nde los nombres sucios:")
print(f"  con version limpia en el mismo CUI: {sucios_con_limpia} ({sucios_con_limpia/total_sucios_contrastive*100:.1f}%)")
print(f"  sin version limpia en el mismo CUI: {sucios_sin_limpia} ({sucios_sin_limpia/total_sucios_contrastive*100:.1f}%)")


# --- seccion 7: formato indice invertido ---
print("\n--- seccion 7: formato indice invertido ---")

# detectar nombres con punto y coma (patron de indice invertido)
canonicos_invertidos = sum(1 for c in concepts if ';' in c['canonical'])
sinonimos_invertidos = sum(1 for c in concepts for s in c['synonyms'] if ';' in s)

print(f"\ncanonicos con formato indice invertido (;): {canonicos_invertidos} ({canonicos_invertidos/total_concepts*100:.1f}%)")
print(f"sinonimos con formato indice invertido (;): {sinonimos_invertidos} ({sinonimos_invertidos/total_synonyms*100:.1f}%)")

# ejemplos
print(f"\nejemplos de canonicos con formato invertido:")
count = 0
for c in concepts:
    if ';' in c['canonical']:
        print(f"  {c['cui']}  {c['canonical']}")
        count += 1
        if count >= 10:
            break


# --- seccion 8: conclusion ---
print("\n--- seccion 8: resumen ---")
print(f"""
vocabulario generado:
  {total_concepts} conceptos UMLS (CUIs)
  {total_synonyms} sinonimos
  {total_names} nombres totales (canonicos + sinonimos)

calidad del autocomplete:
  {canonical_with_q} canonicos con qualifier residual ({canonical_with_q/total_concepts*100:.1f}%)
  {duplicados_reales} de esos son duplicados reales de otro CUI ({duplicados_reales/canonical_with_q*100:.1f}%)
  {canonicos_invertidos} canonicos con formato indice invertido ({canonicos_invertidos/total_concepts*100:.1f}%)

calidad del contrastive loss:
  {cuis_en_contrastive} CUIs generan pares de sinonimos
  {total_sucios_contrastive} nombres sucios ({total_sucios_contrastive/total_syns_contrastive*100:.1f}% del total)
  {sucios_con_limpia} de esos tienen version limpia en el mismo CUI ({sucios_con_limpia/total_sucios_contrastive*100:.1f}%)
  {sucios_sin_limpia} no tienen version limpia en el mismo CUI ({sucios_sin_limpia/total_sucios_contrastive*100:.1f}%)
""")