# classification_section.py

import os
import re
import joblib
import numpy as np
from tensorflow.keras.models import load_model


# ════════════════════════════════════════════════
# 0. PATHS
# ════════════════════════════════════════════════

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ════════════════════════════════════════════════
# 1. CACHE GLOBAL
# ════════════════════════════════════════════════

_MODELS = {}


class Classifier:
    def __init__(self, model_path, vect_path, enc_path):
        key = model_path
        if key in _MODELS:
            cached        = _MODELS[key]
            self.model      = cached["model"]
            self.vectorizer = cached["vectorizer"]
            self.encoder    = cached["encoder"]
            return

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(vect_path):
            raise FileNotFoundError(f"Vectorizer not found: {vect_path}")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(f"Encoder not found: {enc_path}")

        self.model      = load_model(model_path)
        self.vectorizer = joblib.load(vect_path)
        self.encoder    = joblib.load(enc_path)

        _MODELS[key] = {
            "model":      self.model,
            "vectorizer": self.vectorizer,
            "encoder":    self.encoder
        }


# ════════════════════════════════════════════════
# 2. NORMALISATION POUR NN
# (le dataset d'entraînement n'a pas d'accents)
# ════════════════════════════════════════════════

def _normaliser_pour_nn(text: str) -> str:
    """
    Normalise le texte UNIQUEMENT pour le NN.
    Le texte original (avec accents) reste intact pour l'affichage.
    """
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "ù": "u", "û": "u", "ü": "u",
        "ô": "o", "ö": "o",
        "î": "i", "ï": "i",
        "ç": "c"
    }
    text = text.lower()
    for accent, lettre in remplacements.items():
        text = text.replace(accent, lettre)
    return text


# ════════════════════════════════════════════════
# 3. SEGMENTATION
# ════════════════════════════════════════════════

def _segmenter_texte(text: str) -> list:
    if not text:
        return []

    # Ajouter sauts de ligne sur séparateurs naturels
    text = text.replace(":", ":\n")
    text = text.replace(". ", ".\n")
    text = text.replace(" - ", "\n- ")

    # Ajouter saut avant mots-clés de sections
    keywords = [
        "experience", "formation", "competence", "competences",
        "skills", "education", "langue", "langues",
        "projet", "projets", "contact", "interet", "interets",
        "achievement", "certification"
    ]
    for k in keywords:
        text = re.sub(rf"\b{k}\b", f"\n{k}", text, flags=re.IGNORECASE)

    lignes = text.split("\n")
    blocs  = []

    for ligne in lignes:
        ligne = ligne.strip()
        if len(ligne) < 3:
            continue

        # Découper les blocs trop longs (OCR parfois colle tout)
        if len(ligne) > 120:
            words = ligne.split()
            chunk = []
            for w in words:
                chunk.append(w)
                if len(chunk) >= 10:
                    blocs.append(" ".join(chunk))
                    chunk = []
            if chunk:
                blocs.append(" ".join(chunk))
        else:
            blocs.append(ligne)

    return blocs


# ════════════════════════════════════════════════
# 4. PRÉDICTION
# ════════════════════════════════════════════════

def _predire_blocs(classifier: Classifier, blocs: list) -> list:
    if not blocs:
        return []

    # Normaliser pour NN (sans accents) mais garder originaux pour affichage
    blocs_normalises = [_normaliser_pour_nn(b) for b in blocs]

    X      = classifier.vectorizer.transform(blocs_normalises).toarray()
    y_pred = np.argmax(classifier.model.predict(X, verbose=0), axis=1)
    labels = classifier.encoder.inverse_transform(y_pred)

    # Retourner texte ORIGINAL + label
    return list(zip(blocs, labels))


# ════════════════════════════════════════════════
# 5. REGROUPEMENT PAR SECTIONS
# ════════════════════════════════════════════════

def _regrouper_sections(predictions: list, labels_possibles: list) -> dict:
    sections = {label: [] for label in labels_possibles}

    for texte, label in predictions:
        if label in sections:
            sections[label].append(texte)

    # Joindre les blocs de chaque section
    for k in sections:
        sections[k] = "\n".join(sections[k]).strip()

    return sections


# ════════════════════════════════════════════════
# 6. LABELS
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
# 7. CLASSIFICATION CV
# ════════════════════════════════════════════════

def classifier_cv(json_cv: dict, debug=False) -> dict:
    classifier = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_cv.h5"),
        vect_path =os.path.join(MODELS_DIR, "vectorizer_cv.pkl"),
        enc_path  =os.path.join(MODELS_DIR, "encoder_cv.pkl")
    )

    texte = json_cv.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print(f"\n[DEBUG] {len(blocs)} blocs CV détectés")
        for b in blocs[:10]:
            print("  •", b)

    predictions = _predire_blocs(classifier, blocs)
    sections    = _regrouper_sections(predictions, CV_LABELS)

    return {"type": "cv", "sections": sections}


# ════════════════════════════════════════════════
# 8. CLASSIFICATION OFFRE
# ════════════════════════════════════════════════

def classifier_offre(json_offre: dict, debug=False) -> dict:
    classifier = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_offre.h5"),
        vect_path =os.path.join(MODELS_DIR, "vectorizer_offre.pkl"),
        enc_path  =os.path.join(MODELS_DIR, "encoder_offre.pkl")
    )

    texte = json_offre.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print(f"\n[DEBUG] {len(blocs)} blocs Offre détectés")
        for b in blocs[:10]:
            print("  •", b)

    predictions = _predire_blocs(classifier, blocs)
    sections    = _regrouper_sections(predictions, OFFRE_LABELS)

    return {"type": "offre", "sections": sections}