import joblib
from tensorflow.keras.models import load_model

# Charger modèle Offres
model_offre = load_model("../../models/nn_offre.h5")
vectorizer_offre = joblib.load("../../models/vectorizer_offre.pkl")
encoder_offre = joblib.load("../../models/encoder_offre.pkl")

# Texte d'offre
texte_offre = """
🚀 Développeur ERP IFS – Stagiaire
Missions principales :
- Développer des modules ERP.
- Assurer la maintenance corrective.
Profil recherché :
- Bac+5 en informatique.
📍 Localisation : Rabat
"""

# Découpage simple
phrases = [p.strip() for p in texte_offre.split("\n") if len(p.strip()) > 3]

# Prédiction
X = vectorizer_offre.transform(phrases).toarray()
y_pred = model_offre.predict(X)
labels = encoder_offre.inverse_transform(y_pred.argmax(axis=1))

print("\n📊 Résultats classification (Offre):")
for phrase, label in zip(phrases, labels):
    print(f"Phrase: {phrase}\n → Label prédit: {label}\n")
