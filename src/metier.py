# metier.py

import os
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "..", "models")
OUT    = os.path.join(BASE, "..", "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("é", "e").replace("è", "e").replace("ê", "e")
    text = text.replace("à", "a").replace("ù", "u").replace("ô", "o")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ══════════════════════════════════════════
# DETECTION DU METIER DE L'OFFRE
# ══════════════════════════════════════════

ROLE_PATTERNS = [
    (r"\bbusiness analyst\b|\bamoa\b", "Business Analyst / AMOA"),
    (r"\bdata scientist\b|\bdata engineer\b", "Data / BI"),
    (r"\bdeveloppeur\b|\bdeveloper\b|\bdevops\b|\bingenieur logiciel\b", "Développement IT"),
    (r"\bcomptable\b|\bfinance\b|\bcontrole de gestion\b", "Finance / Comptabilité"),
    (r"\bingenieur\b|\bingenierie\b", "Ingénierie"),
    (r"\bchef de projet\b|\bproject manager\b|\bpm\b", "Gestion de projet"),
    (r"\bcommercial\b|\bvente\b|\bbusiness developer\b", "Commercial / Vente"),
    (r"\bmarketing\b|\bdigital marketing\b", "Marketing / Communication"),
    (r"\brh\b|\brecrutement\b|\bressources humaines\b", "Ressources Humaines"),
    (r"\bqa\b|\btest\b|\btester\b", "QA / Testing"),
    (r"\bsupport\b|\bhelpdesk\b|\bservice client\b", "Support / Service client"),
]

def detecter_metier_texte(texte: str) -> str:
    t = _normalize_text(texte)

    for pattern, label in ROLE_PATTERNS:
        if re.search(pattern, t):
            return label

    return "Métier non défini"


# ══════════════════════════════════════════
# ENTRAINEMENT — dataset entreprises
# ══════════════════════════════════════════

def entrainer(path, k=3):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["nom"])

    if "prestige_score" not in df.columns:
        raise ValueError("Colonne 'prestige_score' manquante dans le dataset.")

    if "secteur" not in df.columns:
        df["secteur"] = ""

    print(f"[KMeans#1] {len(df)} entreprises")

    def build_text(row):
        nom     = str(row.get("nom", ""))
        secteur = str(row.get("secteur", ""))
        score   = float(row.get("prestige_score", 5))

        if score >= 8:
            desc = "grande entreprise leader national international prestige groupe"
        elif score >= 5:
            desc = "entreprise moyenne etablie regionale locale"
        else:
            desc = "startup pme petite entreprise jeune emergente innovante"

        return f"{nom} {secteur} {desc}"

    df["texte"] = df.apply(build_text, axis=1)

    vec = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        dtype=np.float32
    )
    X_tfidf = vec.fit_transform(df["texte"]).toarray()

    prestige = df["prestige_score"].fillna(5).values.reshape(-1, 1) / 10.0
    X = np.hstack([X_tfidf, prestige * 5])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    inertias, silhouettes = [], []
    for ki in range(2, 8):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lbl = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_sc, lbl, sample_size=min(300, len(df)))
        silhouettes.append(sil)
        print(f"  k={ki} | inertie={km.inertia_:.1f} | silhouette={sil:.4f}")

    km_final = KMeans(n_clusters=k, random_state=42, n_init=20)
    df["cluster"] = km_final.fit_predict(X_sc)

    stats = df.groupby("cluster")["prestige_score"].mean()
    sorted_c = stats.sort_values(ascending=False).index

    noms_map = {c: n for c, n in zip(
        sorted_c,
        ["Prestige", "PME / Locale", "Emergente / Startup"]
    )}

    poids_map = {
        "Prestige": 1.2,
        "PME / Locale": 1.0,
        "Emergente / Startup": 1.1
    }

    df["cluster_nom"] = df["cluster"].map(noms_map)
    df["poids"] = df["cluster_nom"].map(poids_map)

    print("\n", df.groupby("cluster_nom")["prestige_score"]
          .agg(["mean", "min", "max", "count"]).round(2))

    with open(os.path.join(MODELS, "kmeans1.pkl"), "wb") as f:
        pickle.dump({
            "km": km_final,
            "vec": vec,
            "scaler": scaler,
            "noms_map": noms_map,
            "poids_map": poids_map,
        }, f)

    _plot(inertias, silhouettes, k)
    print("[KMeans#1] Modele sauvegarde")
    return df


# ══════════════════════════════════════════
# PREDICTION — depuis texte offre uniquement
# ══════════════════════════════════════════

def predict_depuis_offre(texte_offre: str) -> dict:
    with open(os.path.join(MODELS, "kmeans1.pkl"), "rb") as f:
        data = pickle.load(f)

    km = data["km"]
    vec = data["vec"]
    scaler = data["scaler"]
    noms_map = data["noms_map"]
    poids_map = data["poids_map"]

    texte_lower = _normalize_text(texte_offre)
    X_tfidf = vec.transform([texte_lower]).toarray()

    prestige_neutral = np.array([[0.5 * 5]])
    X = np.hstack([X_tfidf, prestige_neutral])
    X_sc = scaler.transform(X)

    cluster_id = int(km.predict(X_sc)[0])
    cluster_nom = noms_map.get(cluster_id, "PME / Locale")
    poids = poids_map.get(cluster_nom, 1.0)

    return {
        "cluster": cluster_nom,
        "poids": poids
    }


# ══════════════════════════════════════════
# ADAPTER POUR main.py
# ══════════════════════════════════════════

def detecter_metier_cluster(sections: dict) -> dict:
    texte = " ".join([v for v in sections.values() if v])
    cluster_info = predict_depuis_offre(texte)
    metier = detecter_metier_texte(texte)

    return {
        "metier_detecte": metier,
        "cluster_maroc": cluster_info["cluster"],
        "poids": cluster_info["poids"],
        "explication": f"cluster={cluster_info['cluster']} | poids={cluster_info['poids']}"
    }


# ══════════════════════════════════════════
# GRAPHE SOUTENANCE
# ══════════════════════════════════════════

def _plot(inertias, silhouettes, k_opt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    k_range = list(range(2, 8))

    ax1.plot(k_range, inertias, "o-", color="#00d4ff", lw=2)
    ax1.axvline(x=k_opt, color="red", ls="--", label=f"k={k_opt}")
    ax1.set_title("Methode du Coude")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertie")
    ax1.legend()

    ax2.plot(k_range, silhouettes, "s-", color="#ffd700", lw=2)
    ax2.axvline(x=k_opt, color="red", ls="--", label=f"k={k_opt}")
    ax2.set_title("Score Silhouette")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "kmeans1_coude_silhouette.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    entrainer("../data/entreprises_kmeans_train.csv", k=3)

    tests = [
        "Attijariwafa Bank recrute un analyste financier senior",
        "OCP Group cherche ingenieur chimiste experience 5 ans",
        "Startup innovante tech recherche developpeur python junior",
        "Cabinet comptable regional Fes cherche assistant",
        "Nous sommes une PME distribution alimentaire Casablanca",
        "Grande multinationale internationale recrute manager",
    ]

    print(f"\n{'='*60}")
    for t in tests:
        r = predict_depuis_offre(t)
        print(f"{r['cluster']:<22} x{r['poids']}  |  {t[:45]}")