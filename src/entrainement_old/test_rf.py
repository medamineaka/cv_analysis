import joblib

# Charger les modèles et vectorizers
rf_cv = joblib.load("../../models/old_models/rf_cv.pkl")
vectorizer_cv = joblib.load("../../models/vectorizer_cv.pkl")

rf_offre = joblib.load("../../models/old_models/rf_offre.pkl")
vectorizer_offre = joblib.load("../../models/vectorizer_offre.pkl")

# 🔹 Exemples CV marocains (style LinkedIn)
cv_examples = [
    "Diplômé de l'Université Mohammed Premier en Informatique, avec 3 ans d'expérience en développement web.",
    "Ingénieur d'État en Génie Civil formé à l'EMI Rabat, spécialisé en structures et BTP.",
    "Compétences en gestion de projet, communication interculturelle et leadership dans des associations étudiantes.",
    "Stage en marketing digital chez une startup à Casablanca."
]

X_cv_test = vectorizer_cv.transform(cv_examples)
y_cv_pred = rf_cv.predict(X_cv_test)

print("📊 Résultats CV (Maroc) :")
for text, label in zip(cv_examples, y_cv_pred):
    print(f"Texte: {text}\n → Label prédit: {label}\n")

# 🔹 Exemples Offres marocaines (style LinkedIn)
offre_examples = [
    "Société basée à Casablanca recherche un développeur Python avec 2 ans d'expérience.",
    "Poste en contrat CDI à Rabat, salaire compétitif et avantages sociaux.",
    "Responsabilités : encadrer une équipe de 4 ingénieurs et assurer la livraison des projets.",
    "Exigences : diplôme en informatique et maîtrise du français et de l'anglais."
]

X_offre_test = vectorizer_offre.transform(offre_examples)
y_offre_pred = rf_offre.predict(X_offre_test)

print("\n📊 Résultats Offres (Maroc) :")
for text, label in zip(offre_examples, y_offre_pred):
    print(f"Texte: {text}\n → Label prédit: {label}\n")
