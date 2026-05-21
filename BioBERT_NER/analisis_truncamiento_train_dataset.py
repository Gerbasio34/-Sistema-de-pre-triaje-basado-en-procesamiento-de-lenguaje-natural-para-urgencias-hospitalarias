# analisis_truncamiento_train_dataset.py

# analiza que se pierde por truncamiento en el dataset BIO.
# para cada oracion con mas de max_tokens tokens calcula cuantas
# anotaciones B-SINTOMA/I-SINTOMA caen en la parte truncada.

import statistics
from collections import Counter

input_file = "train_dataset_combined_dedup.txt"
max_tokens = 128


def read_bio_file(path):
    sentences = []
    current = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            token, label = parts
            current.append((token, label))
    if current:
        sentences.append(current)
    return sentences


def analyze_truncation(sentences, max_tokens=max_tokens):
    long_sents = [s for s in sentences if len(s) > max_tokens]
    short_sents = [s for s in sentences if len(s) <= max_tokens]

    # conteos globales
    total_annot       = sum(1 for s in sentences for _, l in s if l != 'O')
    annot_in_short    = sum(1 for s in short_sents for _, l in s if l != 'O')

    annot_visible     = 0
    annot_truncated   = 0
    sents_with_trunc_annot = 0

    lost_per_sent = []
    truncated_positions = []
    truncated_spans = []
    current_span = []
    in_truncated = False

    for sent in long_sents:
        lost = 0
        for i, (token, label) in enumerate(sent):
            in_window = (i < max_tokens)
            if label != 'O':
                if in_window:
                    annot_visible += 1
                else:
                    annot_truncated += 1
                    lost += 1
                    truncated_positions.append(i / len(sent))

        lost_per_sent.append(lost)
        if lost > 0:
            sents_with_trunc_annot += 1

    # reconstruir spans truncados
    for sent in long_sents:
        current_span = []
        for i, (token, label) in enumerate(sent):
            if i < max_tokens:
                continue
            if label == 'B-SINTOMA':
                if current_span:
                    truncated_spans.append(' '.join(current_span))
                current_span = [token]
            elif label == 'I-SINTOMA' and current_span:
                current_span.append(token)
            else:
                if current_span:
                    truncated_spans.append(' '.join(current_span))
                    current_span = []
        if current_span:
            truncated_spans.append(' '.join(current_span))

    # resultados
    print(f"\ntotal oraciones: {len(sentences):,}")
    print(f"oraciones cortas (<={max_tokens}t): {len(short_sents):,} ({len(short_sents)/len(sentences)*100:.1f}%)")
    print(f"oraciones largas (>{max_tokens}t): {len(long_sents):,} ({len(long_sents)/len(sentences)*100:.1f}%)")

    print(f"\ntotal anotaciones (B+I): {total_annot:,}")
    print(f"en oraciones cortas: {annot_in_short:,} ({annot_in_short/total_annot*100:.1f}%)")
    print(f"en oraciones largas, visibles: {annot_visible:,} ({annot_visible/total_annot*100:.1f}%)")
    print(f"truncadas (nunca vistas): {annot_truncated:,} ({annot_truncated/total_annot*100:.1f}%)")

    print(f"\noraciones largas que pierden al menos 1 anotacion: {sents_with_trunc_annot:,} ({sents_with_trunc_annot/len(long_sents)*100:.1f}% de las largas)")

    if lost_per_sent:
        print(f"\nanotaciones perdidas por oracion larga:")
        print(f"  media: {statistics.mean(lost_per_sent):.1f}")
        print(f"  mediana: {statistics.median(lost_per_sent):.1f}")
        print(f"  maximo: {max(lost_per_sent)}")
        cnt = Counter(lost_per_sent)
        print(f"  distribucion (perdidas -> n oraciones):")
        for k in sorted(cnt):
            print(f"    {k} perdidas -> {cnt[k]} oraciones")

    if truncated_positions:
        print(f"\nposicion relativa media de anotaciones truncadas: {statistics.mean(truncated_positions):.2f}")
        print(f"(0=inicio, 1=final de oracion — esperado cercano a 1.0)")

    print(f"\ntotal spans truncados (reconstruidos): {len(truncated_spans):,}")

    if truncated_spans:
        freq = Counter(truncated_spans)
        print(f"\nspans truncados mas frecuentes (top 30):")
        for span, count in freq.most_common(30):
            print(f"  {count}x  {span}")

        span_lengths = [len(s.split()) for s in truncated_spans]
        print(f"\nlongitud media de spans truncados: {statistics.mean(span_lengths):.1f} palabras")
        print(f"longitud maxima: {max(span_lengths)} palabras")

    pct = annot_truncated / total_annot * 100
    if pct < 5:
        conclusion = "baja (<5%): impacto marginal en el entrenamiento."
    elif pct < 15:
        conclusion = "moderada (5-15%): limitacion real pero manejable."
    else:
        conclusion = "alta (>15%): se elimina una fraccion significativa del supervision signal."
    print(f"\nfraccion truncada: {pct:.1f}% -> impacto {conclusion}")


if __name__ == "__main__":
    print(f"leyendo {input_file}...")
    sentences = read_bio_file(input_file)

    lengths = [len(s) for s in sentences]
    print(f"longitud media: {statistics.mean(lengths):.1f} tokens")
    print(f"longitud mediana: {statistics.median(lengths):.1f} tokens")
    print(f"longitud maxima: {max(lengths)} tokens")
    print(f"percentil 95: {sorted(lengths)[int(len(lengths)*0.95)]} tokens")

    analyze_truncation(sentences)