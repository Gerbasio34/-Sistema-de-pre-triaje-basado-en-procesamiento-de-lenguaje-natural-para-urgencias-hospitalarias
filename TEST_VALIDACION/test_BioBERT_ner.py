# test_BioBERT_ner.py
#
# test exhaustivo del modelo NER — criterio de palabras clave por sintoma
#
# estructura de cada caso:
#   (ID, texto, [[kw_sint1], [kw_sint2], ...], descripcion)
#
# un sintoma se considera detectado (TP) si el NER predice al menos un token
# que contenga alguna de sus palabras clave.
# un sintoma no detectado cuenta como FN.
# una prediccion que no cubre ningun sintoma esperado cuenta como FP.
# si expected=[] y el NER no predice nada -> OK. si predice algo -> FP.

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torchcrf import CRF

ner_model_path = "../BioBERT_NER/biobert_sintomas_ner_v4_crf"
ner_base_model = "dmis-lab/biobert-base-cased-v1.2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- modelo ---

class BioBertCRF(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf        = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs         = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions       = self.classifier(sequence_output)
        if labels is not None:
            crf_labels        = labels.clone()
            crf_mask          = (labels != -100)
            crf_labels[~crf_mask] = 0
            crf_mask[:, 0]    = True
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            mask    = attention_mask.bool()
            decoded = self.crf.decode(emissions, mask=mask)
            return {"decoded": decoded}


def load_model():
    tokenizer  = AutoTokenizer.from_pretrained(ner_model_path)
    checkpoint = torch.load(
        f"{ner_model_path}/biobert_crf_model.pt",
        map_location=device, weights_only=False)
    model = BioBertCRF(ner_base_model, checkpoint['num_labels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, tokenizer


def predict(text, model, tokenizer):
    id2label = {0: "O", 1: "B-SINTOMA", 2: "I-SINTOMA"}
    inputs   = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs   = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        decoded = outputs["decoded"][0]
    tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels  = [id2label[tag] for tag in decoded]
    symptoms, current = [], []
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if label == "B-SINTOMA":
            if current:
                symptoms.append(tokenizer.convert_tokens_to_string(current).strip())
            current = [token]
        elif label == "I-SINTOMA" and current:
            current.append(token)
        else:
            if current:
                symptoms.append(tokenizer.convert_tokens_to_string(current).strip())
                current = []
    if current:
        symptoms.append(tokenizer.convert_tokens_to_string(current).strip())
    return symptoms


# --- criterio de match ---

def symptom_detected(predicted_spans, keywords):
    # true si algun span predicho contiene alguna de las keywords (case-insensitive)
    for span in predicted_spans:
        span_lower = span.lower()
        for kw in keywords:
            if kw.lower() in span_lower:
                return True
    return False


def evaluate_case(predicted, expected_symptoms):
    # devuelve (tp, fp, fn, detalle)
    # detalle: lista de (keywords, detected) para imprimir
    if not expected_symptoms:
        fp = len(predicted)
        return 0, fp, 0, []

    detalle          = []
    matched_pred_idx = set()

    tp = fn = 0
    for kws in expected_symptoms:
        detected = False
        for i, span in enumerate(predicted):
            span_lower = span.lower()
            if any(kw.lower() in span_lower for kw in kws):
                detected = True
                matched_pred_idx.add(i)
                break
        if detected:
            tp += 1
        else:
            fn += 1
        detalle.append((kws, detected))

    fp = len([i for i in range(len(predicted)) if i not in matched_pred_idx])
    return tp, fp, fn, detalle


# --- dataset: 50 casos exhaustivos ---
# formato: (ID, texto, [[kw_sint1], [kw_sint2], ...], descripcion)

test_cases = [
 
    # CARDIACO (12 casos)
    ("CARD-01",
     "I have crushing chest pain that radiates to my left arm and jaw.",
     [["chest", "pain"], ["arm"], ["jaw"]],
     "IAM clasico - 3 sintomas"),
 
    ("CARD-02",
     "My heart is racing, I feel dizzy and I'm sweating a lot.",
     [["racing", "heart"], ["dizzy"], ["sweat"]],
     "Taquicardia + cortejo vegetativo"),
 
    ("CARD-03",
     "I have chest tightness and I can't breathe when I walk upstairs.",
     [["chest", "tightness"], ["breath"]],
     "Angina de esfuerzo"),
 
    ("CARD-04",
     "My legs are swollen and I wake up gasping for air at night.",
     [["swollen", "leg"], ["gasping", "air"]],
     "Insuficiencia cardiaca"),
 
    ("CARD-05",
     "I feel my heart skipping beats and then I get lightheaded.",
     [["skipping", "beats"], ["lightheaded"]],
     "Arritmia + presincope"),
 
    ("CARD-06",
     "I have a sharp pain in my chest that gets worse when I breathe in.",
     [["chest", "pain"], ["breath"]],
     "Pericarditis / pleuritica"),
 
    ("CARD-07",
     "My ankles are swelling and I feel very short of breath even sitting.",
     [["swelling", "ankle"], ["breath"]],
     "ICC descompensada"),
 
    ("CARD-08",
     "I keep getting palpitations and my pulse feels very irregular.",
     [["palpitation"], ["irregular", "pulse"]],
     "FA sintomatica"),
 
    ("CARD-09",
     "My chest feels really heavy and I keep blacking out.",
     [["chest", "heavy"], ["black out", "blackout", "blacking"]],
     "Sincope cardiaco"),
 
    ("CARD-10",
     "I feel like my heart is going to explode and my left arm is tingling.",
     [["heart"], ["arm", "tingle"]],
     "IAM atipico"),
 
    ("CARD-11",
     "I woke up drenched in sweat and my heart was pounding really hard.",
     [["sweat"], ["heart", "pound"]],
     "Episodio nocturno cardiaco"),
 
    ("CARD-12",
     "I get out of breath just going to the bathroom and my feet are puffy.",
     [["breath"], ["feet", "puffy", "swollen"]],
     "ICC severa"),
 
    # NEUROLOGICO (12 casos)
    ("NEURO-01",
     "I can't move my right arm and the left side of my face is drooping.",
     [["arm", "move"], ["face", "droop"]],
     "Ictus hemisférico"),
 
    ("NEURO-02",
     "I have the worst headache of my life and my neck feels very stiff.",
     [["headache"], ["neck", "stiff"]],
     "HSA / meningitis"),
 
    ("NEURO-03",
     "Everything is spinning and I can't walk in a straight line.",
     [["spinning"], ["walk"]],
     "Vertigo con ataxia"),
 
    ("NEURO-04",
     "I'm seeing double and my left hand is completely numb.",
     [["double", "vision", "seeing"], ["numb", "hand"]],
     "Deficit neurologico focal"),
 
    ("NEURO-05",
     "I had a seizure and now I'm confused and don't know where I am.",
     [["seizure"], ["confused"]],
     "Post-ictal"),
 
    ("NEURO-06",
     "My hands are shaking badly and I keep dropping things.",
     [["shaking", "hands"], ["dropping"]],
     "Temblor + debilidad"),
 
    ("NEURO-07",
     "I have electric shocks shooting down my spine when I bend my neck.",
     [["electric", "shock", "spine"]],
     "Signo de Lhermitte"),
 
    ("NEURO-08",
     "My speech is all slurred and I can't find the right words.",
     [["slurred", "speech"], ["words", "speak"]],
     "Afasia + disartria"),
 
    ("NEURO-09",
     "I keep forgetting things mid-sentence and I got lost driving home.",
     [["forget", "memory"], ["lost", "confused"]],
     "Deterioro cognitivo agudo"),
 
    ("NEURO-10",
     "My whole left side went numb for about ten minutes and now it's fine.",
     [["numb", "left"]],
     "AIT hemisférico"),
 
    ("NEURO-11",
     "I blacked out and when I came around I had bitten my tongue.",
     [["blacked out", "blackout", "faint"], ["tongue", "bitten"]],
     "Crisis epileptica"),
 
    ("NEURO-12",
     "I have this weird ringing in my ears and the room keeps spinning.",
     [["ringing", "ear"], ["spinning"]],
     "Vertigo + tinnitus"),
 
    # RESPIRATORIO (10 casos)
    ("RESP-01",
     "I can't breathe and my lips are turning blue.",
     [["breath"], ["lips", "blue"]],
     "Insuficiencia respiratoria"),
 
    ("RESP-02",
     "I've been coughing up blood and I have a sharp pain when I inhale.",
     [["cough", "blood"], ["pain", "inhale", "breath"]],
     "Hemoptisis + dolor pleuritico"),
 
    ("RESP-03",
     "I have a wheezing sound when I breathe and my chest feels very tight.",
     [["wheez"], ["chest", "tight"]],
     "Asma"),
 
    ("RESP-04",
     "I have a fever and I'm bringing up thick green phlegm.",
     [["fever"], ["phlegm", "mucus", "green"]],
     "Neumonia"),
 
    ("RESP-05",
     "I wake up choking every night and I feel exhausted during the day.",
     [["choking"], ["exhausted", "tired"]],
     "Apnea del sueno"),
 
    ("RESP-06",
     "I've been sneezing constantly, my nose is completely blocked and my eyes are watery.",
     [["sneez"], ["nose", "blocked"], ["eyes", "water"]],
     "Rinitis alergica - 3 sintomas"),
 
    ("RESP-07",
     "I can feel fluid gurgling in my chest every time I breathe.",
     [["chest", "fluid", "gurgling"], ["breath"]],
     "Derrame pleural"),
 
    ("RESP-08",
     "I suddenly got a sharp stabbing pain in my side and I can't take a deep breath.",
     [["pain", "side"], ["breath"]],
     "Neumotrax espontaneo"),
 
    ("RESP-09",
     "I've been coughing for three weeks and I'm losing weight without trying.",
     [["cough"], ["weight", "losing"]],
     "TBC / neoplasia pulmonar"),
 
    ("RESP-10",
     "My breathing is really fast and I feel like I can't get enough air.",
     [["breath", "fast"], ["air"]],
     "Hiperventilacion / TEP"),
 
    # GASTROINTESTINAL (12 casos)
    ("GI-01",
     "I've been throwing up blood and I have severe pain in my upper abdomen.",
     [["vomit", "blood", "throwing"], ["pain", "abdomen", "stomach"]],
     "Hematemesis + dolor"),
 
    ("GI-02",
     "I have terrible cramps and watery diarrhea with blood in it.",
     [["cramp"], ["diarrhea"], ["blood"]],
     "Disenteria - 3 sintomas"),
 
    ("GI-03",
     "My belly is swollen, I can't pass gas and I haven't had a bowel movement in days.",
     [["swollen", "belly"], ["gas"], ["bowel"]],
     "Obstruccion intestinal - 3 sintomas"),
 
    ("GI-04",
     "My skin and eyes are turning yellow and my urine is very dark.",
     [["yellow", "skin", "jaundice"], ["yellow", "eyes"], ["urine", "dark"]],
     "Ictericia - 3 sintomas"),
 
    ("GI-05",
     "I have sharp pain in the lower right side of my belly and I feel nauseous.",
     [["pain", "lower right", "belly"], ["nauseous", "nausea"]],
     "Apendicitis"),
 
    ("GI-06",
     "I swallowed a fish bone and now I have pain when I swallow.",
     [["pain", "swallow", "throat"]],
     "Cuerpo extrano - 1 sintoma"),
 
    ("GI-07",
     "My poop is completely black and tarry and my stomach is killing me.",
     [["black", "stool", "poop", "tarry"], ["stomach", "pain"]],
     "Melenas"),
 
    ("GI-08",
     "I have this burning feeling right here in the middle of my chest after eating.",
     [["burn", "chest"], ["eating", "after"]],
     "GERD / pirosis"),
 
    ("GI-09",
     "I can't stop throwing up and I haven't been able to keep anything down for two days.",
     [["vomit", "throwing"], ["keep down"]],
     "Vomitos incoercibles"),
 
    ("GI-10",
     "My right side hurts really bad, especially when I press on it, and I have a fever.",
     [["pain", "right side"], ["fever"]],
     "Colecistitis / apendicitis"),
 
    ("GI-11",
     "I have really bad heartburn going up into my throat and I keep burping acid.",
     [["heartburn", "burn"], ["acid", "burp"]],
     "Reflujo severo"),
 
    ("GI-12",
     "I haven't eaten in days because everything I eat makes me throw up immediately.",
     [["vomit", "throwing", "sick"], ["eat"]],
     "Intolerancia oral"),
 
    # TRAUMA (10 casos)
    ("TRAUMA-01",
     "I fell and my ankle is swollen and I can't put any weight on it.",
     [["ankle", "swollen"], ["weight"]],
     "Fractura tobillo"),
 
    ("TRAUMA-02",
     "I hit my head and now I have a headache and my vision is blurry.",
     [["headache", "head"], ["vision", "blurry"]],
     "TCE"),
 
    ("TRAUMA-03",
     "I burned my hand on the stove and it's covered in blisters.",
     [["burn", "hand"], ["blister"]],
     "Quemadura"),
 
    ("TRAUMA-04",
     "My shoulder popped out and I can't lift my arm at all.",
     [["shoulder", "popped", "dislocated"], ["arm", "lift", "move"]],
     "Luxacion hombro"),
 
    ("TRAUMA-05",
     "I was in a car accident, my neck is very stiff and painful and I have a headache.",
     [["neck", "stiff"], ["neck", "pain"], ["headache"]],
     "Latigazo cervical - 3 sintomas"),
 
    ("TRAUMA-06",
     "I twisted my knee and now it's really swollen and I can't straighten it.",
     [["knee", "twisted"], ["swollen", "knee"]],
     "Lesion ligamentosa rodilla"),
 
    ("TRAUMA-07",
     "I cut my hand pretty deep on broken glass and it won't stop bleeding.",
     [["cut", "hand"], ["bleed"]],
     "Herida incisa"),
 
    ("TRAUMA-08",
     "I fell off my bike and my collarbone is sticking out and hurts like crazy.",
     [["collarbone", "clavicle"], ["pain", "hurt"]],
     "Fractura clavicula"),
 
    ("TRAUMA-09",
     "My back is killing me after I lifted something heavy at work this morning.",
     [["back", "pain"], ["lift"]],
     "Lumbalgia aguda por esfuerzo"),
 
    ("TRAUMA-10",
     "I got hit in the eye and now everything is blurry and it's really swollen.",
     [["eye", "hit"], ["blurry", "vision"], ["swollen", "eye"]],
     "Trauma ocular - 3 sintomas"),
 
    # COLOQUIAL (16 casos)
    ("COLL-01",
     "My back is killing me and I can barely walk.",
     [["back"], ["walk"]],
     "Lumbalgia severa"),
 
    ("COLL-02",
     "I feel like crap, my whole body aches and I'm burning up.",
     [["body", "ache"], ["burn", "fever", "hot"]],
     "Cuadro gripal"),
 
    ("COLL-03",
     "I've got the runs real bad and my tummy is cramping like crazy.",
     [["run", "diarrhea"], ["cramp", "tummy"]],
     "GEA coloquial"),
 
    ("COLL-04",
     "My head is pounding and the light is killing my eyes.",
     [["head", "pound"], ["light", "eye"]],
     "Migrana coloquial - fotofobia"),
 
    ("COLL-05",
     "I'm peeing blood and it burns like crazy.",
     [["pee", "blood", "urine"], ["burn"]],
     "Hematuria + disuria"),
 
    ("COLL-06",
     "My throat is on fire and I can barely swallow my own spit.",
     [["throat"], ["swallow"]],
     "Faringitis severa"),
 
    ("COLL-07",
     "I've been puking my guts out all night and now I feel super weak.",
     [["puke", "vomit"], ["weak"]],
     "Vomitos + astenia"),
 
    ("COLL-08",
     "I woke up and my face was all numb and tingly on one side.",
     [["numb", "face"], ["tingly", "tingling"]],
     "Paralisis facial / ictus"),
 
    ("COLL-09",
     "I feel really out of it, like super confused and I can't think straight.",
     [["confused", "out of it"], ["think"]],
     "Confusion aguda"),
 
    ("COLL-10",
     "My stomach is killing me and I keep running to the bathroom.",
     [["stomach", "pain"], ["bathroom", "diarrhea"]],
     "GEA coloquial 2"),
 
    ("COLL-11",
     "I can't stop shaking and I feel really cold even though my skin is burning.",
     [["shaking", "shiver"], ["cold"], ["burn", "fever"]],
     "Escalofrios + fiebre"),
 
    ("COLL-12",
     "My ear is killing me and I can't hear properly on that side.",
     [["ear", "pain"], ["hear"]],
     "Otitis"),
 
    ("COLL-13",
     "I feel like I'm going to pass out and my heart is going crazy.",
     [["pass out", "faint"], ["heart"]],
     "Presincope + palpitaciones"),
 
    ("COLL-14",
     "My knee is so swollen I can't bend it and it hurts to touch.",
     [["knee", "swollen"], ["pain", "hurt"]],
     "Artritis / hemartrosis"),
 
    ("COLL-15",
     "I've been really itchy all over and I'm breaking out in hives.",
     [["itch"], ["hive", "rash"]],
     "Urticaria generalizada"),
 
    ("COLL-16",
     "I bit my tongue really hard and it won't stop bleeding.",
     [["tongue", "bit"], ["bleed"]],
     "Herida lingual"),
 
    # NEGACIONES (8 casos)
    ("NEG-01",
     "No fever, no cough, but I have a terrible headache.",
     [["headache"]],
     "Negacion doble, 1 sintoma real"),
 
    ("NEG-02",
     "I'm not vomiting but I have constant nausea and bad stomach pain.",
     [["nausea"], ["pain", "stomach"]],
     "Negacion vomito, 2 sintomas reales"),
 
    ("NEG-03",
     "I don't have chest pain, the pain is in my upper back.",
     [["pain", "back"]],
     "Negacion con relocalizacion"),
 
    ("NEG-04",
     "I'm not dizzy but I feel extremely weak and I keep fainting.",
     [["weak"], ["faint"]],
     "Negacion mareo, 2 sintomas reales"),
 
    ("NEG-05",
     "I don't have a rash but my skin is really itchy everywhere.",
     [["itch", "skin"]],
     "Negacion rash, prurito real"),
 
    ("NEG-06",
     "No bleeding but I have really bad cramps and pressure in my pelvis.",
     [["cramp"], ["pressure", "pelvis"]],
     "Negacion sangrado, dolor pelvico real"),
 
    ("NEG-07",
     "I'm not having trouble breathing but my chest feels really tight and uncomfortable.",
     [["chest", "tight"]],
     "Negacion disnea, opresion real"),
 
    ("NEG-08",
     "I don't have a headache but my vision has been really blurry since this morning.",
     [["vision", "blurry"]],
     "Negacion cefalea, vision borrosa real"),
 
    # SIN SINTOMAS (8 casos)
    ("NOSINT-01",
     "I need a refill of my blood pressure medication.",
     [],
     "Solicitud medicacion"),
 
    ("NOSINT-02",
     "My grandmother had breast cancer and I want to get checked.",
     [],
     "Antecedentes familiares"),
 
    ("NOSINT-03",
     "I want to discuss my treatment options for diabetes.",
     [],
     "Consulta tratamiento"),
 
    ("NOSINT-04",
     "I need a sick note for work, I was ill last week.",
     [],
     "Solicitud baja laboral"),
 
    ("NOSINT-05",
     "I'd like to get a flu vaccine today.",
     [],
     "Solicitud vacuna"),
 
    ("NOSINT-06",
     "My doctor told me to come in for a follow-up after my surgery.",
     [],
     "Seguimiento postquirurgico"),
 
    ("NOSINT-07",
     "I want to get my blood test results explained.",
     [],
     "Consulta resultados"),
 
    ("NOSINT-08",
     "I need a referral to see a specialist about my cholesterol.",
     [],
     "Solicitud derivacion"),
 
    # EDGE CASES (12 casos)
    ("EDGE-01",
     "I'm pregnant and having severe abdominal cramps with bleeding.",
     [["cramp", "abdominal"], ["bleed"]],
     "Amenaza de aborto"),
 
    ("EDGE-02",
     "I think I'm having an allergic reaction, my lips and tongue are swelling and I can't breathe.",
     [["lips", "tongue", "swelling"], ["breath"]],
     "Anafilaxia - 2 sintomas criticos"),
 
    ("EDGE-03",
     "My urine smells really strong, looks cloudy and it burns when I go.",
     [["urine", "smell"], ["urine", "cloudy"], ["burn"]],
     "ITU - 3 sintomas"),
 
    ("EDGE-04",
     "I took too many pills by accident and now I feel really sick and my heart is racing.",
     [["pill", "medication", "overdose"], ["sick", "nausea"], ["heart", "racing"]],
     "Sobredosis accidental - 3 sintomas"),
 
    ("EDGE-05",
     "I think I broke my finger, it's pointing the wrong way and I can't move it.",
     [["finger", "broke", "broken"], ["move"]],
     "Fractura falange con deformidad"),
 
    ("EDGE-06",
     "I was stung by a bee and my throat is starting to close up.",
     [["sting", "bee"], ["throat", "closing", "swell"]],
     "Anafilaxia por picadura"),
 
    ("EDGE-07",
     "I haven't urinated in over a day and my lower belly is really painful and swollen.",
     [["urine", "urinate"], ["belly", "pain", "swollen"]],
     "Retencion urinaria aguda"),
 
    ("EDGE-08",
     "I feel really hot, I'm not sweating at all and I'm getting very confused.",
     [["hot", "fever"], ["sweat"], ["confused"]],
     "Golpe de calor - 3 sintomas"),
 
    ("EDGE-09",
     "My tooth is killing me and my whole jaw and face are swollen.",
     [["tooth", "pain"], ["jaw", "face", "swollen"]],
     "Absceso dental con celulitis"),
 
    ("EDGE-10",
     "I got something stuck in my eye and now it's really red and I can't open it.",
     [["eye", "stuck", "foreign"], ["red", "eye"], ["eye", "open"]],
     "Cuerpo extrano ocular"),
 
    ("EDGE-11",
     "I'm eight months pregnant and I suddenly have a really bad headache and my vision is blurry.",
     [["headache"], ["vision", "blurry"]],
     "Preeclampsia - 2 sintomas"),
 
    ("EDGE-12",
     "I fainted at the gym and when I woke up my chest was hurting.",
     [["faint", "blackout", "passed out"], ["chest", "pain"]],
     "Sincope + dolor toracico"),
]


# --- evaluacion ---

def main():
    print("cargando modelo NER...")
    model, tokenizer = load_model()
    print(f"modelo cargado ({device})\n")

    total_tp = total_fp = total_fn = 0
    category_results = {}
    all_results = []

    header = f"{'ID':<12} {'Texto':<52} {'Predichos':<38} {'Sintomas (kw -> detectado)'}"
    print(header)
    print("=" * 130)

    for case_id, text, expected_symptoms, description in test_cases:
        predicted = predict(text, model, tokenizer)
        tp, fp, fn, detalle = evaluate_case(predicted, expected_symptoms)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_results.append((case_id, text, expected_symptoms, description, predicted, tp, fp, fn, detalle))

        prefix = case_id.split("-")[0]
        if prefix not in category_results:
            category_results[prefix] = [0, 0, 0]
        category_results[prefix][0] += tp
        category_results[prefix][1] += fp
        category_results[prefix][2] += fn

        pred_str   = ", ".join(predicted) if predicted else "(ninguno)"
        text_short = text[:50] + ".." if len(text) > 52 else text
        pred_short = pred_str[:36] + ".." if len(pred_str) > 38 else pred_str

        if not expected_symptoms:
            status = "OK" if not predicted else f"FP({fp})"
        else:
            missing = [kws for kws, det in detalle if not det]
            if fn == 0 and fp == 0:
                status = "OK"
            elif fn == 0:
                status = f"OK+FP({fp})"
            elif fp == 0:
                status = f"MISS({'/'.join([kws[0] for kws in missing])})"
            else:
                status = f"MISS({'/'.join([kws[0] for kws in missing])})+FP({fp})"

        kw_summary = " | ".join(
            f"{'ok' if det else '--'}{kws[0]}" for kws, det in detalle
        ) if detalle else "-"

        print(f"{case_id:<12} {text_short:<52} {pred_short:<38} {kw_summary}  [{status}]")

    # metricas globales
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nmetricas globales:")
    print(f"  precision : {precision:.3f}")
    print(f"  recall    : {recall:.3f}")
    print(f"  f1        : {f1:.3f}")
    print(f"  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")

    print("\nmetricas por categoria:")
    print(f"  {'cat':<10} {'prec':>8} {'rec':>8} {'f1':>8} {'tp':>5} {'fp':>5} {'fn':>5}")
    print(f"  {'-'*52}")
    for prefix, (tp, fp, fn) in sorted(category_results.items()):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {prefix:<10} {p:>8.3f} {r:>8.3f} {f:>8.3f} {tp:>5} {fp:>5} {fn:>5}")

    # sintomas no detectados
    print("\nsintomas no detectados (FN por keyword):")
    for case_id, text, expected_symptoms, description, predicted, tp, fp, fn, detalle in all_results:
        if fn > 0:
            missing = [kws for kws, det in detalle if not det]
            print(f"  {case_id:<12} {description}")
            for kws in missing:
                print(f"             -> keyword '{kws[0]}' no detectada  (alternativas: {kws[1:]})")


if __name__ == "__main__":
    main()