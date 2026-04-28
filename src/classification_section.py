# classification_section.py

import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model


# ════════════════════════════════════════════════
# 0. PATH PRO (🔥 FIX PRINCIPAL)
# ════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ════════════════════════════════════════════════
# 1. CACHE GLOBAL
# ════════════════════════════════════════════════

_MODELS = {}


class Classifier:
    def __init__(self, model_path, vect_path, enc_path):

        key = model_path

        if key in _MODELS:
            cached = _MODELS[key]
            self.model = cached["model"]
            self.vectorizer = cached["vectorizer"]
            self.encoder = cached["encoder"]
            return

        print("\n[DEBUG] Loading model:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not os.path.exists(vect_path):
            raise FileNotFoundError(f"Vectorizer not found: {vect_path}")

        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Encoder not found: {enc_path}")

        self.model = load_model(model_path)
        self.vectorizer = joblib.load(vect_path)
        self.encoder = joblib.load(enc_path)

        _MODELS[key] = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "encoder": self.encoder
        }


# ════════════════════════════════════════════════
# 2. SEGMENTATION (🔥 FIX PRINCIPAL QUALITÉ)
# ════════════════════════════════════════════════

def _segmenter_texte(text: str):

    if not text:
        return []

    # structure minimale
    text = text.replace(":", ":\n")
    text = text.replace(". ", ".\n")
    text = text.replace(" - ", "\n- ")

    keywords = [
        "experience", "formation", "competence", "skills",
        "education", "langue", "projet", "contact"
    ]

    for k in keywords:
        text = text.replace(k, f"\n{k}")

    lignes = text.split("\n")

    blocs = []

    for l in lignes:
        l = l.strip()

        if len(l) < 3:
            continue

        # 🔥 découpage OCR long
        if len(l) > 120:
            words = l.split()
            chunk = []

            for w in words:
                chunk.append(w)

                if len(chunk) >= 10:
                    blocs.append(" ".join(chunk))
                    chunk = []

            if chunk:
                blocs.append(" ".join(chunk))
        else:
            blocs.append(l)

    return blocs


# ════════════════════════════════════════════════
# 3. PREDICTION
# ════════════════════════════════════════════════

def _predire_blocs(classifier: Classifier, blocs: list):

    if not blocs:
        return []

    X = classifier.vectorizer.transform(blocs).toarray()
    y_pred = np.argmax(classifier.model.predict(X, verbose=0), axis=1)
    labels = classifier.encoder.inverse_transform(y_pred)

    return list(zip(blocs, labels))


# ════════════════════════════════════════════════
# 4. REGROUPEMENT
# ════════════════════════════════════════════════

def _regrouper_sections(predictions: list, labels_possibles: list):

    sections = {label: [] for label in labels_possibles}

    for texte, label in predictions:
        if label in sections:
            sections[label].append(texte)

    for k in sections:
        sections[k] = "\n".join(sections[k]).strip()

    return sections


# ════════════════════════════════════════════════
# 5. LABELS
# ════════════════════════════════════════════════

CV_LABELS = [
    "HEADER", "EDUCATION", "EXPERIENCE", "SKILL",
    "LOCATION", "LANGUAGE", "INTEREST",
    "CONTACT", "PROJECT", "ACHIEVEMENT"
]

OFFRE_LABELS = [
    "HEADER", "REQUIREMENT", "RESPONSIBILITY",
    "EDUCATION", "EXPERIENCE", "LOCATION",
    "CONTRACT", "SALARY", "LANGUAGE",
    "BENEFIT", "NOT_IMPORTANT"
]


# ════════════════════════════════════════════════
# 6. CLASSIFICATION CV
# ════════════════════════════════════════════════

def classifier_cv(json_cv: dict, debug=False) -> dict:

    classifier = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_cv.h5"),
        vect_path=os.path.join(MODELS_DIR, "vectorizer_cv.pkl"),
        enc_path=os.path.join(MODELS_DIR, "encoder_cv.pkl")
    )

    texte = json_cv.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print("\n[DEBUG BLOCS CV]")
        for b in blocs[:10]:
            print("•", b)

    predictions = _predire_blocs(classifier, blocs)
    sections = _regrouper_sections(predictions, CV_LABELS)

    return {
        "type": "cv",
        "sections": sections
    }


# ════════════════════════════════════════════════
# 7. CLASSIFICATION OFFRE
# ════════════════════════════════════════════════

def classifier_offre(json_offre: dict, debug=False) -> dict:

    classifier = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_offre.h5"),
        vect_path=os.path.join(MODELS_DIR, "vectorizer_offre.pkl"),
        enc_path=os.path.join(MODELS_DIR, "encoder_offre.pkl")
    )

    texte = json_offre.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print("\n[DEBUG BLOCS OFFRE]")
        for b in blocs[:10]:
            print("•", b)

    predictions = _predire_blocs(classifier, blocs)
    sections = _regrouper_sections(predictions, OFFRE_LABELS)

    return {
        "type": "offre",
        "sections": sections
    }

print("\n[DEBUG PATH]")
print(os.path.join(MODELS_DIR, "nn_cv.h5"))