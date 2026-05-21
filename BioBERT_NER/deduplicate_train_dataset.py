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
            if len(parts) == 2:
                current.append((parts[0], parts[1]))
    if current:
        sentences.append(current)
    return sentences

def write_bio_file(sentences, path):
    with open(path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")

sentences = read_bio_file("train_dataset_combined.txt")
print(f"oraciones originales: {len(sentences)}")

seen = set()
unique = []
for sent in sentences:
    key = ' '.join(tok for tok, _ in sent)
    if key not in seen:
        seen.add(key)
        unique.append(sent)

print(f"oraciones unicas: {len(unique)}")
print(f"duplicados eliminados: {len(sentences) - len(unique)}")

write_bio_file(unique, "train_dataset_combined_dedup.txt")
print("guardado en train_dataset_combined_dedup.txt")