# generate_dataset.py

# Convierte el corpus CADEC local a formato BIO para entrenamiento NER.

# CADEC (Concept Annotation and Drug Coding) es un corpus de posts de foros
# médicos con anotaciones de efectos adversos (ADR) y síntomas. Los archivos
# de texto están en cadec/text/ y las anotaciones en cadec/original/ con
# extensión .ann en formato Brat.

# Este script lee cada par (.txt, .ann), tokeniza el texto preservando los
# offsets de caracteres, y asigna etiquetas BIO (B-SINTOMA, I-SINTOMA, O)
# a cada token según si cae dentro de un span anotado como ADR o Symptom.

# Salida: sintomas_ner_bio.txt
#   Formato: una línea por token con "token\tetiqueta", oraciones separadas
#   por línea vacía. Compatible con el formato que espera train_biobert_crf.py.

# Relación con otros scripts:
#   Este script genera sintomas_ner_bio.txt
#   generate_dataset_2.py genera cadec_combined_bio_v3.txt
#   Ambos archivos se concatenan por terminal para obtener train_dataset.txt:
#     cat sintomas_ner_bio.txt cadec_combined_bio_v3.txt > train_dataset.txt
#   train_dataset.txt es el input de train_biobert_crf.py

import os
import re

text_dir = "cadec/text"
ann_dir = "cadec/original"
output_file = "sintomas_ner_bio.txt"

# Solo anotamos ADR y Symptom como síntomas, el resto (Drug, Disease...) se ignora
symptom_types = {"ADR", "Symptom"}


def read_annotations(ann_path):
    """Lee un archivo .ann en formato Brat y devuelve lista de (start, end)
    para las entidades de tipo ADR o Symptom. Ignora spans discontinuos."""
    spans = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("T"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            info = parts[1].split()
            entity_type = info[0]
            if entity_type not in symptom_types:
                continue
            # Los spans discontinuos tienen ; en las posiciones, se ignoran
            if ';' in info[1] or ';' in info[2]:
                continue
            start = int(info[1])
            end = int(info[2])
            spans.append((start, end))
    return spans


def get_span_label(start, end, spans):
    """Devuelve el span (s, e) al que pertenece el token [start, end],
    o (None, None) si no está dentro de ningún span anotado."""
    for s, e in spans:
        if start >= s and end <= e:
            return s, e
    return None, None


def tokenize(text):
    """Tokeniza el texto separando palabras y signos de puntuación,
    preservando los offsets de caracteres de cada token."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    result = []
    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        end = start + len(token)
        result.append((token, start, end))
        offset = end
    return result


with open(output_file, "w", encoding="utf-8") as out:
    for txt_file in os.listdir(text_dir):
        if not txt_file.endswith(".txt"):
            continue

        txt_path = os.path.join(text_dir, txt_file)
        ann_path = os.path.join(ann_dir, txt_file.replace(".txt", ".ann"))

        if not os.path.exists(ann_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        spans_ner = read_annotations(ann_path)
        tokens = tokenize(text)

        current_span_end = -1
        inside_span = False

        for token, start, end in tokens:
            span_start, span_end = get_span_label(start, end, spans_ner)
            if span_start is not None:
                if start == span_start or not inside_span:
                    label = "B-SINTOMA"
                    inside_span = True
                    current_span_end = span_end
                else:
                    label = "I-SINTOMA"
            else:
                label = "O"
                inside_span = False
                current_span_end = -1

            out.write(f"{token}\t{label}\n")

        out.write("\n")
