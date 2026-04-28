import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# === Évaluation CV ===
df_cv = pd.read_csv("../../data/dataset_cv.csv")[["Text_Block", "Label"]]
train_cv, test_cv = train_test_split(df_cv, test_size=0.2, random_state=42)

vectorizer_cv = TfidfVectorizer(max_features=5000, stop_words=["french","english"])
X_train_cv = vectorizer_cv.fit_transform(train_cv["Text_Block"])
y_train_cv = train_cv["Label"]

X_test_cv = vectorizer_cv.transform(test_cv["Text_Block"])
y_test_cv = test_cv["Label"]

rf_cv = RandomForestClassifier(n_estimators=200, random_state=42)
rf_cv.fit(X_train_cv, y_train_cv)

y_pred_cv = rf_cv.predict(X_test_cv)
print("📊 Rapport CV")
print(classification_report(y_test_cv, y_pred_cv))
print("Matrice de confusion:\n", confusion_matrix(y_test_cv, y_pred_cv))

# === Évaluation Offre ===
df_offre = pd.read_csv("../../data/dataset_offre.csv")[["Text_Block", "Label"]]
train_offre, test_offre = train_test_split(df_offre, test_size=0.2, random_state=42)

vectorizer_offre = TfidfVectorizer(max_features=5000, stop_words=["french","english"])
X_train_offre = vectorizer_offre.fit_transform(train_offre["Text_Block"])
y_train_offre = train_offre["Label"]

X_test_offre = vectorizer_offre.transform(test_offre["Text_Block"])
y_test_offre = test_offre["Label"]

rf_offre = RandomForestClassifier(n_estimators=200, random_state=42)
rf_offre.fit(X_train_offre, y_train_offre)

y_pred_offre = rf_offre.predict(X_test_offre)
print("\n📊 Rapport Offre")
print(classification_report(y_test_offre, y_pred_offre))
print("Matrice de confusion:\n", confusion_matrix(y_test_offre, y_pred_offre))
