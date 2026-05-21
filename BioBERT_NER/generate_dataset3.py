# generate_dataset_psytar.py
#
# convierte PsyTAR (ADR_Identified, WD_Identified, SSI_Identified)
# a formato BIO para entrenamiento NER.
# output: psytar_bio.txt
# formato: una linea por token con "token\tetiqueta", oraciones separadas por linea vacia.
# compatible con train_biobert_crf.py
# el output de este script se concatena de la siguiente manera cat train_dataset_dedup.txt psytar_bio.txt > train_dataset_combined.txt

import pandas as pd
import re

input_file  = "PsyTAR_dataset.xlsx"
output_file = "psytar_bio.txt"

# hojas con sintomas y nombres de columnas de spans
sheets = {
    "ADR_Identified": [f"ADR{i}"  for i in range(1, 31)],
    "WD_Identified":  [f"WD{i}"   for i in range(1, 11)],
    "SSI_Identified": [f"SSI{i}"  for i in range(1, 11)],
}


def tokenize(text):
    # separa palabras y signos de puntuacion preservando offsets
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    result = []
    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        end   = start + len(token)
        result.append((token, start, end))
        offset = end
    return result


def find_span_in_tokens(tokens, span_text):
    # busca span_text como subsecuencia de tokens (case-insensitive)
    span_tokens = re.findall(r"\w+|[^\w\s]", span_text, re.UNICODE)
    if not span_tokens:
        return None
    span_lower = [t.lower() for t in span_tokens]
    for i in range(len(tokens) - len(span_lower) + 1):
        window = [t[0].lower() for t in tokens[i:i + len(span_lower)]]
        if window == span_lower:
            return (i, i + len(span_lower))
    return None


def sentence_to_bio(sentence, spans):
    tokens = tokenize(sentence)
    if not tokens:
        return []

    labels = ["O"] * len(tokens)

    for span_text in spans:
        if not span_text or not span_text.strip():
            continue
        span_text = span_text.strip()
        match = find_span_in_tokens(tokens, span_text)
        if match:
            start, end = match
            labels[start] = "B-SINTOMA"
            for i in range(start + 1, end):
                labels[i] = "I-SINTOMA"

    return list(zip([t[0] for t in tokens], labels))


total_sentences = 0
total_skipped   = 0

with open(output_file, "w", encoding="utf-8") as out:
    for sheet_name, span_cols in sheets.items():
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        print(f"procesando {sheet_name}: {len(df)} filas")

        for _, row in df.iterrows():
            sentence = row.get("sentences", "")
            if not isinstance(sentence, str) or not sentence.strip():
                total_skipped += 1
                continue

            spans = []
            for col in span_cols:
                if col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        spans.append(str(val).strip())

            bio = sentence_to_bio(sentence, spans)
            if not bio:
                total_skipped += 1
                continue

            for token, label in bio:
                out.write(f"{token}\t{label}\n")
            out.write("\n")
            total_sentences += 1

print(f"\ngenerado: {output_file}")
print(f"  oraciones escritas: {total_sentences}")
print(f"  oraciones saltadas: {total_skipped}")
