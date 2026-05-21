# train_biobert_crf.py
#
# fine-tuning de BioBERT + CRF para extraccion de sintomas (NER) en texto clinico.
# dataset: CADEC en formato BIO (train_dataset.txt)
# output: ./biobert_sintomas_ner_v4_crf/

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall
)
from torchcrf import CRF
import json

# Configuración
bio_file = "train_dataset_combined_dedup.txt"
model_name = "dmis-lab/biobert-base-cased-v1.2"
output_dir = "./biobert_sintomas_ner_v4_crf"

labels_list = ["O", "B-SINTOMA", "I-SINTOMA"]
label2id = {l: i for i, l in enumerate(labels_list)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(labels_list)

print("=" * 60)
print("Entrenamiento BioBERT + CRF — Extraccion de Sintomas")
print("=" * 60)
print(f"Dataset:     {bio_file}")
print(f"Modelo base: {model_name}")
print(f"Salida:      {output_dir}")
print(f"Etiquetas:   {labels_list}")


# Lectura del dataset en formato BIO
def read_bio_file(file_path):
    """Lee un archivo BIO con formato token\\tetiqueta y devuelve
    listas de tokens y etiquetas por oración."""
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        tokens, token_labels = [], []
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    texts.append(tokens)
                    labels.append(token_labels)
                    tokens, token_labels = [], []
                continue
            splits = line.split("\t")
            if len(splits) != 2:
                continue
            token, label = splits
            if label not in label2id:
                label = "O"
            tokens.append(token)
            token_labels.append(label2id[label])
        if tokens:
            texts.append(tokens)
            labels.append(token_labels)
    return texts, labels


print("\nLeyendo dataset...")
tokens_list, labels_list_seq = read_bio_file(bio_file)
print(f"Dataset cargado: {len(tokens_list):,} oraciones, {sum(len(t) for t in tokens_list):,} tokens")

label_counts = {i: 0 for i in range(num_labels)}
for seq in labels_list_seq:
    for label in seq:
        label_counts[label] += 1

print("\nDistribucion de etiquetas:")
total_labels = sum(label_counts.values())
for label_id, count in label_counts.items():
    pct = (count / total_labels) * 100
    print(f"  {id2label[label_id]:12s}: {count:,} ({pct:.2f}%)")


# Split train/val/test (80/10/10)
print("\nDividiendo dataset...")
full_dataset = Dataset.from_dict({"tokens": tokens_list, "labels": labels_list_seq})
train_val_split = full_dataset.train_test_split(test_size=0.2, seed=42)
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": train_val_split["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"]
})

print(f"  Train:      {len(dataset['train']):,} oraciones")
print(f"  Validation: {len(dataset['validation']):,} oraciones")
print(f"  Test:       {len(dataset['test']):,} oraciones")


# Tokenizacion y alineacion de etiquetas
print("\nInicializando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128


def tokenize_and_align_labels(batch):
    """Tokeniza con el tokenizer de BioBERT y alinea las etiquetas BIO
    con los subtokens. El primer subtoken de cada palabra hereda la
    etiqueta original; los subtokens adicionales reciben I-SINTOMA si
    la palabra era B/I-SINTOMA, o O si era O."""
    tokenized_inputs = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    labels = []
    for i, label in enumerate(batch["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                original_label = label[word_idx]
                if original_label == 0:
                    label_ids.append(0)
                else:
                    label_ids.append(2)  # I-SINTOMA para subtokens
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print("Tokenizando dataset...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, desc="Tokenizando")
print("Tokenizacion completada")


# Modelo BioBERT + CRF
print("\nConstruyendo modelo BioBERT + CRF...")


class BioBertCRF(nn.Module):
    """BioBERT con capa CRF para NER.
    El encoder BERT genera emisiones por token que la CRF decodifica
    con el algoritmo de Viterbi garantizando secuencias BIO validas."""

    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # Entrenamiento: calcular loss CRF
            crf_labels = labels.clone()
            crf_mask = (labels != -100)
            crf_labels[~crf_mask] = 0  # placeholder, enmascarado por la CRF
            crf_mask[:, 0] = True      # CRF requiere que el primer paso sea True
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            # Inferencia: decodificacion Viterbi
            mask = attention_mask.bool()
            decoded = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "decoded": decoded}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BioBertCRF(model_name, num_labels, dropout=0.1).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modelo cargado: {trainable_params:,} parametros entrenables")
print(f"CRF: matriz de transiciones {num_labels}x{num_labels}")


# Trainer personalizado para CRF
class CRFTrainer(Trainer):
    """Trainer personalizado que gestiona el forward pass de la CRF.
    El Trainer estandar de HuggingFace no soporta el loss de la CRF
    directamente, por lo que hay que sobreescribir compute_loss y
    prediction_step."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Sobreescribe para usar decodificacion Viterbi en lugar de argmax."""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            outputs_with_loss = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            loss = outputs_with_loss["loss"]

            outputs_decode = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=None
            )
            decoded = outputs_decode["decoded"]

        if prediction_loss_only:
            return (loss, None, None)

        # Convertir listas decodificadas a tensor con one-hot para que
        # el argmax del Trainer estandar funcione correctamente
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[1]
        preds_tensor = torch.zeros(batch_size, seq_len, self.model.num_labels, device=loss.device)

        for i, seq in enumerate(decoded):
            for j, tag_id in enumerate(seq):
                if j < seq_len:
                    preds_tensor[i, j, tag_id] = 1.0

        return (loss, preds_tensor, labels)


# Metricas de evaluacion (token-level y span-level)
def compute_metrics(eval_pred):
    """Calcula metricas a nivel de token (sklearn) y a nivel de span
    (seqeval). El span F1 es la metrica principal para NER y la
    comparable con la literatura."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Token-level
    true_preds_token = []
    true_labels_token = []
    for prediction, label in zip(predictions, labels):
        for pred, lab in zip(prediction, label):
            if lab != -100:
                true_preds_token.append(pred)
                true_labels_token.append(lab)

    precision = precision_score(true_labels_token, true_preds_token, average='weighted', zero_division=0)
    recall = recall_score(true_labels_token, true_preds_token, average='weighted', zero_division=0)
    f1 = f1_score(true_labels_token, true_preds_token, average='weighted', zero_division=0)

    sintoma_mask = [(lab == 1 or lab == 2) for lab in true_labels_token]
    if sum(sintoma_mask) > 0:
        sintoma_f1 = f1_score(
            [l for l, m in zip(true_labels_token, sintoma_mask) if m],
            [p for p, m in zip(true_preds_token, sintoma_mask) if m],
            average='weighted', zero_division=0
        )
    else:
        sintoma_f1 = 0.0

    # Span-level (seqeval)
    true_labels_span = []
    true_preds_span = []
    for prediction, label in zip(predictions, labels):
        seq_true = []
        seq_pred = []
        for pred, lab in zip(prediction, label):
            if lab != -100:
                seq_true.append(id2label[lab])
                seq_pred.append(id2label[pred])
        true_labels_span.append(seq_true)
        true_preds_span.append(seq_pred)

    span_f1 = seqeval_f1(true_labels_span, true_preds_span)
    span_precision = seqeval_precision(true_labels_span, true_preds_span)
    span_recall = seqeval_recall(true_labels_span, true_preds_span)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sintoma_f1": sintoma_f1,
        "span_f1": span_f1,
        "span_precision": span_precision,
        "span_recall": span_recall,
    }


# Configuracion de entrenamiento
print("\nConfigurando entrenamiento...")

training_args = TrainingArguments(
    output_dir=output_dir,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="span_f1",
    greater_is_better=True,

    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    logging_strategy="steps",

    save_total_limit=2,

    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    gradient_accumulation_steps=1,

    seed=42
)

print(f"  learning rate: {training_args.learning_rate}")
print(f"  batch size: {training_args.per_device_train_batch_size}")
print(f"  epocas: {training_args.num_train_epochs}")
print(f"  metrica: span_f1 (seqeval)")
print(f"  gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# Inicializacion del trainer
print("\nInicializando CRF Trainer...")

trainer = CRFTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("CRF Trainer listo")


# Entrenamiento
print("\nIniciando entrenamiento BioBERT + CRF")

trainer.train()
print("\nEntrenamiento completado")


# Evaluacion en test
print("\nEvaluacion en conjunto de test")

test_results = trainer.predict(tokenized_dataset["test"])
test_metrics = test_results.metrics

print("\nMetricas token-level (sklearn):")
print(f"  Precision:   {test_metrics['test_precision']:.4f}")
print(f"  Recall:      {test_metrics['test_recall']:.4f}")
print(f"  F1:          {test_metrics['test_f1']:.4f}")
print(f"  Sintoma F1:  {test_metrics['test_sintoma_f1']:.4f}")

print("\nMetricas span-level (seqeval) — comparables con literatura:")
print(f"  Precision:   {test_metrics['test_span_precision']:.4f}")
print(f"  Recall:      {test_metrics['test_span_recall']:.4f}")
print(f"  F1:          {test_metrics['test_span_f1']:.4f}")

# Classification report token-level
predictions = np.argmax(test_results.predictions, axis=2)
true_labels_tok = []
true_preds_tok = []
for prediction, label in zip(predictions, test_results.label_ids):
    for pred, lab in zip(prediction, label):
        if lab != -100:
            true_preds_tok.append(id2label[pred])
            true_labels_tok.append(id2label[lab])

print("\nReporte token-level por clase:")
print(classification_report(true_labels_tok, true_preds_tok, digits=4))

# Seqeval span-level report
true_labels_span = []
true_preds_span = []
for prediction, label in zip(predictions, test_results.label_ids):
    seq_true, seq_pred = [], []
    for pred, lab in zip(prediction, label):
        if lab != -100:
            seq_true.append(id2label[lab])
            seq_pred.append(id2label[pred])
    true_labels_span.append(seq_true)
    true_preds_span.append(seq_pred)

print("\nReporte span-level (seqeval):")
print(seqeval_report(true_labels_span, true_preds_span, digits=4))


# Guardado del modelo
print("\nGuardando modelo...")

save_path = os.path.join(output_dir, "biobert_crf_model.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": model_name,
    "num_labels": num_labels,
    "labels_list": labels_list,
    "label2id": label2id,
    "id2label": id2label,
}, save_path)

tokenizer.save_pretrained(output_dir)
print(f"Modelo guardado en: {save_path}")

metrics_file = os.path.join(output_dir, "test_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Metricas guardadas en: {metrics_file}")
