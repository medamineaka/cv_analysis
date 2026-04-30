# kmeans2_train_recommendation.py

import os
import pickle
import re
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data", "kmeans2_domaine_5000.csv")
MODELS = os.path.join(BASE, "..", "models")
os.makedirs(MODELS, exist_ok=True)


# ─────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────
def _clean(t):
    if not t:
        return ""
    t = str(t).lower()

    t = (
        t.replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("à", "a")
         .replace("ù", "u")
         .replace("ô", "o")
    )

    t = re.sub(r"[^a-z0-9\s,/+\-]", " ", t)
    t = re.sub(r"\s+", " ", t)

    return t.strip()


# ─────────────────────────────────────────────
# BUILD TEXT
# ─────────────────────────────────────────────
def build_text(row):
    return " ".join([
        str(row.get("domaine", "")),
        str(row.get("job_title", "")),
        str(row.get("required_skills", "")),
        str(row.get("description", "")),
        str(row.get("level", "")),
    ])


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(k=8):
    print("\n[KMeans#2] Chargement dataset...")
    df = pd.read_csv(DATA)

    # Normalisation des colonnes
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required_cols = ["domaine", "job_title", "required_skills", "description"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le dataset: {col}")

    if "level" not in df.columns:
        df["level"] = ""

    df["text"] = df.apply(build_text, axis=1).map(_clean)

    print(f"[KMeans#2] {len(df)} offres chargées")

    # TF-IDF
    vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vec.fit_transform(df["text"]).toarray()

    # Standardisation
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    df["cluster"] = km.fit_predict(Xs)

    # Label clusters by dominant domain
    cluster_map = {}
    cluster_skills = {}
    terms = np.array(vec.get_feature_names_out())

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        dom = sub["domaine"].value_counts().idxmax()
        cluster_map[int(c)] = dom

        idx = np.where(df["cluster"].values == c)[0]
        center = X[idx].mean(axis=0)

        if hasattr(center, "A1"):
            center = center.A1

        top = np.argsort(center)[-15:][::-1]
        cluster_skills[int(c)] = [t for t in terms[top].tolist() if len(t) > 2]

    # Sauvegarde modèle
    model_path = os.path.join(MODELS, "kmeans2_reco.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "vec": vec,
            "scaler": scaler,
            "km": km,
            "cluster_map": cluster_map,
            "cluster_skills": cluster_skills
        }, f)

    print(f"[KMeans#2] Sauvegardé: {model_path}")
    print("[KMeans#2] Cluster map:", cluster_map)

    return df


# ─────────────────────────────────────────────
# PREDICTION UTILE POUR RECOMMANDATION
# ─────────────────────────────────────────────
def predict_domain_from_text(text: str) -> dict:
    model_path = os.path.join(MODELS, "kmeans2_reco.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modele introuvable: {model_path}. Lance d'abord train()."
        )

    with open(model_path, "rb") as f:
        pack = pickle.load(f)

    vec = pack["vec"]
    scaler = pack["scaler"]
    km = pack["km"]
    cluster_map = pack["cluster_map"]
    cluster_skills = pack["cluster_skills"]

    text_clean = _clean(text)
    X = vec.transform([text_clean]).toarray()
    Xs = scaler.transform(X)

    cluster_id = int(km.predict(Xs)[0])
    domain = cluster_map.get(cluster_id, "Unknown")

    return {
        "cluster_id": cluster_id,
        "domain": domain,
        "skills_cluster": cluster_skills.get(cluster_id, [])
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = train(k=8)
    print(df.head(10).to_string(index=False))