import joblib
import numpy as np
from tensorflow.keras.models import load_model

def tester_nn(texte, model_path, vect_path, enc_path, nom=""):
    print(f"\n🔎 Test modèle {nom}...")

    # Charger modèle et outils
    model = load_model(model_path)
    vectorizer = joblib.load(vect_path)
    encoder = joblib.load(enc_path)

    # Prétraitement du texte
    phrases = [p.strip().lower() for p in texte.split("\n") if len(p.strip()) > 3]

    # Vectorisation
    X = vectorizer.transform(phrases).toarray()

    # Prédiction
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    labels = encoder.inverse_transform(y_pred)

    # Résultats
    for phrase, label in zip(phrases, labels):
        print(f"Phrase: {phrase}\n → Label prédit: {label}\n")

# Exemple de test Offre
texte_offre = """
🚀 Développeur ERP IFS – Stagiaire
Missions principales :
- Développer des modules ERP.
- Assurer la maintenance corrective.
Profil recherché :
- Bac+5 en informatique.
📍 Localisation : Rabat
"""

tester_nn(
    texte_offre,
    model_path="../../models/nn_offre.h5",
    vect_path="../../models/vectorizer_offre.pkl",
    enc_path="../../models/encoder_offre.pkl",
    nom="Offres"
)

# Exemple de test CV
texte_cv = """
📢 Curriculum Vitae
Formation :
- Master en Informatique, Université de Rabat
Expérience :
- Développeur Python chez XYZ
Compétences :
- Python, Django, Docker
"""

tester_nn(
    texte_cv,
    model_path="../../models/nn_cv.h5",
    vect_path="../../models/vectorizer_cv.pkl",
    enc_path="../../models/encoder_cv.pkl",
    nom="CV"
)
