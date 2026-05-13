# generate_dataset_2.py

# Descarga 4 variantes del corpus CADEC desde HuggingFace y las convierte
# a formato BIO para entrenamiento NER.

# Los 4 datasets son versiones distintas del mismo corpus con estructuras
# diferentes, por lo que cada uno necesita su propio procesador:

#   1. KevinSpaghetti/cadec
#      Campos: text, ade, term_PT
#      El campo 'ade' es el span de texto que hay que buscar dentro de 'text'.
#      Se tokeniza ambos y se localiza el span por búsqueda de subsecuencia.
#
#   2. mireiaplalis/processed_cadec_no_prefix
#      Campos: tokens, ner_tags, info
#      El campo 'info' contiene anotaciones en formato dict o string serializado.
#      Solo se procesan anotaciones con ner_tag == 'ADR'.

#   3. akramRedjdal/cadec-ner-dataset-clean
#      Campos: tokens, labels
#      Las etiquetas son strings (B-ADR, I-ADR, B-Drug...).
#      Solo se mapean B-ADR e I-ADR a B-SINTOMA e I-SINTOMA.
#
#   4. mireiaplalis/processed_cadec
#      Misma estructura que el dataset 2, mismo procesador.

# Salida: cadec_combined_bio_v3.txt
#   Formato: una línea por token con "token\tetiqueta", oraciones separadas
#   por línea vacía. Compatible con el formato que espera train_biobert_crf.py.

# Relación con otros scripts:
#   generate_dataset.py genera sintomas_ner_bio.txt desde CADEC local
#   Este script genera cadec_combined_bio_v3.txt desde HuggingFace
#   Ambos archivos se concatenan por terminal:
#     cat sintomas_ner_bio.txt cadec_combined_bio_v3.txt > train_dataset.txt
#   train_dataset.txt es el input de train_biobert_crf.py

from datasets import load_dataset
import re
import ast

output_file = "cadec_combined_bio_v3.txt"


def tokenize_simple(text):
    """Tokeniza separando puntuación básica como token propio."""
    text = re.sub(r'([.,;:!?\)])', r' \1', text)
    text = re.sub(r'(\()', r'\1 ', text)
    return text.split()


def find_span_in_tokens(tokens, ade_tokens):
    """Busca la secuencia ade_tokens dentro de tokens (case-insensitive).
    Devuelve (start, end) del span o None si no se encuentra."""
    ade_lower = [t.lower() for t in ade_tokens]
    for i in range(len(tokens) - len(ade_lower) + 1):
        window = [t.lower() for t in tokens[i:i + len(ade_lower)]]
        if window == ade_lower:
            return (i, i + len(ade_lower))
    return None


def process_kevin_spaghetti(dataset):
    """Convierte KevinSpaghetti/cadec a formato BIO.
    Busca el campo 'ade' dentro del campo 'text' por coincidencia de tokens."""
    sentences = []
    skipped = 0

    for row in dataset:
        text = row["text"]
        ade = row["ade"]

        tokens = tokenize_simple(text)
        ade_tokens = tokenize_simple(ade)

        span = find_span_in_tokens(tokens, ade_tokens)
        if span is None:
            skipped += 1
            continue

        start, end = span
        labels = []
        for i, token in enumerate(tokens):
            if i == start:
                labels.append("B-SINTOMA")
            elif start < i < end:
                labels.append("I-SINTOMA")
            else:
                labels.append("O")

        sentences.append(list(zip(tokens, labels)))

    print(f"  [KevinSpaghetti] Procesados: {len(sentences)} | Ignorados (no match): {skipped}")
    return sentences


def normalize_original(original_text):
    """Normaliza el texto del campo 'original' para facilitar el matching.
    Quita guiones, convierte a minúsculas y elimina intensificadores."""
    normalized = original_text.replace('-', ' ').lower()
    normalized = re.sub(r'\b(terrible|extreme|severe|mild)\b\s*', '', normalized, flags=re.IGNORECASE)
    return normalized.strip()


def process_mireiaplalis_fixed(dataset):
    """Convierte datasets mireiaplalis a formato BIO usando el campo 'original'
    de cada anotación ADR para localizar el span en los tokens."""
    sentences = []
    errors = 0

    for idx, row in enumerate(dataset):
        tokens = row["tokens"]
        info_raw = row["info"]

        if isinstance(info_raw, list):
            info_list = info_raw
        elif isinstance(info_raw, str):
            try:
                info_list = ast.literal_eval(info_raw)
            except Exception as e:
                if idx < 3:
                    print(f"  Error parsing info: {e}")
                errors += 1
                continue
        else:
            errors += 1
            continue

        labels = ["O"] * len(tokens)

        for annotation in info_list:
            if isinstance(annotation, dict):
                ner_tag = annotation.get('ner_tag')
            elif isinstance(annotation, str):
                try:
                    annotation = ast.literal_eval(annotation)
                    ner_tag = annotation.get('ner_tag')
                except:
                    continue
            else:
                continue

            if ner_tag != 'ADR':
                continue

            original = annotation.get('original', '')
            if not original:
                continue

            original_norm = normalize_original(original)
            original_tokens = tokenize_simple(original_norm)

            span = find_span_in_tokens(tokens, original_tokens)
            if span:
                start, end = span
                labels[start] = "B-SINTOMA"
                for i in range(start + 1, end):
                    labels[i] = "I-SINTOMA"

        sentences.append(list(zip(tokens, labels)))

    print(f"  [mireiaplalis FIXED] Procesados: {len(sentences)} | Errors: {errors}")
    return sentences


def process_akram(dataset):
    """Convierte akramRedjdal/cadec-ner-dataset-clean a formato BIO.
    Mapea B-ADR/I-ADR a B-SINTOMA/I-SINTOMA, el resto a O."""
    sintoma_tags = {"B-ADR": "B-SINTOMA", "I-ADR": "I-SINTOMA"}
    sentences = []

    for row in dataset:
        tokens = row["tokens"]
        labels_raw = row["labels"]

        if len(tokens) != len(labels_raw):
            continue

        labels = [sintoma_tags.get(l, "O") for l in labels_raw]
        sentences.append(list(zip(tokens, labels)))

    print(f"  [akramRedjdal] Procesados: {len(sentences)}")
    return sentences


def write_bio_file(all_sentences, filepath):
    """Escribe todas las oraciones en formato BIO al archivo de salida."""
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")

    print(f"\nArchivo generado: {filepath}")
    print(f"  Total oraciones: {len(all_sentences)}")
    total_tokens = sum(len(s) for s in all_sentences)
    total_sintomas = sum(1 for s in all_sentences for _, l in s if l != "O")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tokens etiquetados como SINTOMA: {total_sintomas}")


def main():
    all_sentences = []

    print("Cargando datasets...\n")

    print("[1/4] KevinSpaghetti/cadec")
    ds1 = load_dataset("KevinSpaghetti/cadec", split="train")
    all_sentences.extend(process_kevin_spaghetti(ds1))

    print("[2/4] mireiaplalis/processed_cadec_no_prefix")
    ds2 = load_dataset("mireiaplalis/processed_cadec_no_prefix", split="train")
    all_sentences.extend(process_mireiaplalis_fixed(ds2))

    print("[3/4] akramRedjdal/cadec-ner-dataset-clean")
    ds3 = load_dataset("akramRedjdal/cadec-ner-dataset-clean", split="train")
    all_sentences.extend(process_akram(ds3))

    print("[4/4] mireiaplalis/processed_cadec")
    ds4 = load_dataset("mireiaplalis/processed_cadec", split="train")
    all_sentences.extend(process_mireiaplalis_fixed(ds4))

    print("\nEscribiendo archivo BIO combinado...")
    write_bio_file(all_sentences, output_file)


if __name__ == "__main__":
    main()
