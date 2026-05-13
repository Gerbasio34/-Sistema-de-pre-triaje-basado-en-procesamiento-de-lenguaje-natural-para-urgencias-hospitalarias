# test_sapbert_vs_base.py

# compara sapbert base vs fine-tuned en dos baterias de pares,

# test 1: pares coloquial -> termino clinico (base vs fine-tuned)
# test 2: pares nombre limpio vs nombre con qualifier (base vs fine-tuned)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

sapbert_name    = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
classifier_path = "../SAPBERT_RAW/modelo_sapbert_finetuned_v5g_7D_ablation.pt"


# pares coloquial -> termino clinico
pairs_coloquial = [
    ("the runs",                         "diarrhea"),
    ("trouble peeing",                   "dysuria"),
    ("can't breathe",                    "dyspnea"),
    ("belly hurts",                      "abdominal pain"),
    ("heart racing",                     "tachycardia"),
    ("throwing up",                      "vomiting"),
    ("blood in pee",                     "hematuria"),
    ("pins and needles",                 "paresthesia"),
    ("passed out",                       "syncope"),
    ("feel dizzy",                       "vertigo"),
    ("chest feels tight",                "chest pain"),
    ("I fainted",                        "syncope"),
    ("swollen ankles",                   "edema"),
    ("my head is pounding",              "headache"),
    ("can't stop coughing",              "cough"),
    ("feel really weak",                 "asthenia"),
    ("numbness in my hand",              "hand numbness"),
    ("burning when i pee",               "dysuria"),
    ("seeing double",                    "diplopia"),
    ("blood in poo",                     "hematochezia"),
    ("I want to end it all",             "si"),
    ("puking and the runs",              "n/v/d"),
    ("my mom doesn't know where she is", "confusion"),
    ("my blood pressure is really low",  "hypotension"),
    ("my friend is really drunk",        "etoh"),
]

# pares nombre limpio vs nombre con qualifier
pairs_qualifier = [
    ("chest pain",           "chest pain (diagnosis)"),
    ("abdominal pain",       "abdominal pain (diagnosis)"),
    ("diarrhea",             "diarrhea (physical finding)"),
    ("vomiting",             "vomiting (symptom)"),
    ("headache",             "headache (diagnosis)"),
    ("dyspnea",              "dyspnea (observable entity)"),
    ("syncope",              "syncope (diagnosis)"),
    ("edema",                "edema (physical finding)"),
    ("hematuria",            "hematuria (physical finding)"),
    ("vertigo",              "vertigo (diagnosis)"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# cargar sapbert base
tokenizer = AutoTokenizer.from_pretrained(sapbert_name)

bert_base = AutoModel.from_pretrained(sapbert_name)
bert_base.to(device).eval()

# cargar sapbert fine-tuned (solo el encoder bert)
checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
bert_state = {k.replace("bert.", ""): v
              for k, v in checkpoint["model_state"].items()
              if k.startswith("bert.")}
bert_ft = AutoModel.from_pretrained(sapbert_name)
bert_ft.load_state_dict(bert_state)
bert_ft.to(device).eval()


print("modelos cargados\n")


def encode_sapbert(model, texts):
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=64, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


def sim_sapbert(model, a, b):
    embs = encode_sapbert(model, [a, b])
    return cosine_similarity([embs[0]], [embs[1]])[0][0]


# --- test 1: coloquial -> termino clinico ---
print("test 1: coloquial -> termino clinico")
print("par / base / ft / mejora")
print("-" * 60)

gains = []
for a, b in pairs_coloquial:
    s_base = sim_sapbert(bert_base, a, b)
    s_ft   = sim_sapbert(bert_ft,   a, b)
    gain   = s_ft - s_base
    gains.append(gain)
    marker = " +" if gain > 0.05 else (" -" if gain < -0.05 else "")
    print(f"  {a} / {b}: base={s_base:.4f} ft={s_ft:.4f} mejora={gain:+.4f}{marker}")

print(f"\n  media base:        {np.mean([sim_sapbert(bert_base, a, b) for a,b in pairs_coloquial]):.4f}")
print(f"  media fine-tuned:  {np.mean([sim_sapbert(bert_ft,   a, b) for a,b in pairs_coloquial]):.4f}")
print(f"  mejora media:      {np.mean(gains):+.4f}")
print(f"  pares mejorados:   {sum(1 for g in gains if g > 0)} de {len(gains)}")
print(f"  pares empeorados:  {sum(1 for g in gains if g < 0)} de {len(gains)}")


# --- test 2: nombre limpio vs nombre con qualifier ---
print("\ntest 2: nombre limpio vs nombre con qualifier")
print("par / base / ft / mejora")
print("-" * 60)

gains_q = []
for a, b in pairs_qualifier:
    s_base = sim_sapbert(bert_base, a, b)
    s_ft   = sim_sapbert(bert_ft,   a, b)
    gain   = s_ft - s_base
    gains_q.append(gain)
    print(f"  {a} / ...({b.split('(')[1]}: base={s_base:.4f} ft={s_ft:.4f} mejora={gain:+.4f}")

print(f"\n  media base:        {np.mean([sim_sapbert(bert_base, a, b) for a,b in pairs_qualifier]):.4f}")
print(f"  media fine-tuned:  {np.mean([sim_sapbert(bert_ft,   a, b) for a,b in pairs_qualifier]):.4f}")
print(f"  mejora media:      {np.mean(gains_q):+.4f}")

