import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === CV ===
df_cv = pd.read_csv("../../data/dataset_cv.csv")[["Text_Block", "Label"]]

# Split train/test (5%)
train_cv, test_cv = train_test_split(df_cv, test_size=0.05, random_state=42)

# TF-IDF amélioré
vectorizer_cv = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words=["french","english"]
)
X_train_cv = vectorizer_cv.fit_transform(train_cv["Text_Block"])
X_test_cv = vectorizer_cv.transform(test_cv["Text_Block"])

# Encoder les labels
encoder_cv = LabelEncoder()
y_train_cv = encoder_cv.fit_transform(train_cv["Label"])
y_test_cv = encoder_cv.transform(test_cv["Label"])

# Entraînement XGBoost
xgb_cv = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
xgb_cv.fit(X_train_cv, y_train_cv)

# Sauvegarde
joblib.dump(xgb_cv, "../../models/old_models/xgb_cv.pkl")
joblib.dump(vectorizer_cv, "../../models/vectorizer_cv.pkl")
joblib.dump(encoder_cv, "../../models/encoder_cv.pkl")

# Évaluation
y_pred_cv = xgb_cv.predict(X_test_cv)
print("📊 Rapport CV (XGBoost)")
print(classification_report(y_test_cv, y_pred_cv, target_names=encoder_cv.classes_))
print("Matrice de confusion:\n", confusion_matrix(y_test_cv, y_pred_cv))


# === Offres ===
df_offre = pd.read_csv("../../data/dataset_offre.csv")[["Text_Block", "Label"]]
train_offre, test_offre = train_test_split(df_offre, test_size=0.05, random_state=42)

vectorizer_offre = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words=["french","english"]
)
X_train_offre = vectorizer_offre.fit_transform(train_offre["Text_Block"])
X_test_offre = vectorizer_offre.transform(test_offre["Text_Block"])

encoder_offre = LabelEncoder()
y_train_offre = encoder_offre.fit_transform(train_offre["Label"])
y_test_offre = encoder_offre.transform(test_offre["Label"])

xgb_offre = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
xgb_offre.fit(X_train_offre, y_train_offre)

joblib.dump(xgb_offre, "../../models/old_models/xgb_offre.pkl")
joblib.dump(vectorizer_offre, "../../models/vectorizer_offre.pkl")
joblib.dump(encoder_offre, "../../models/encoder_offre.pkl")

y_pred_offre = xgb_offre.predict(X_test_offre)
print("\n📊 Rapport Offres (XGBoost)")
print(classification_report(y_test_offre, y_pred_offre, target_names=encoder_offre.classes_))
print("Matrice de confusion:\n", confusion_matrix(y_test_offre, y_pred_offre))
