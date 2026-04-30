# scoring.py
"""
Scoring professionnel CV ↔ Offre.
Méthode : SBERT multilingue (similarité sémantique) + TF-IDF fallback + poids KMeans#1.
Compatible avec tout domaine marocain.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── SBERT multilingue — supporte français, arabe, anglais
try:
    from sentence_transformers import SentenceTransformer, util
    _sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    SBERT_OK = True
except Exception:
    SBERT_OK = False


# ════════════════════════════════════════════════
# 1. NETTOYAGE
# ════════════════════════════════════════════════

def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _merge(sections: dict, cles: list) -> str:
    """Fusionne plusieurs sections en un seul texte."""
    return " ".join(
        sections.get(k, "") for k in cles
        if isinstance(sections.get(k), str) and sections.get(k, "").strip()
    )


def _merge_all(sections: dict) -> str:
    return " ".join(
        v for v in sections.values()
        if isinstance(v, str) and v.strip()
    )


# ════════════════════════════════════════════════
# 2. SIMILARITÉ SBERT + TFIDF FALLBACK
# ════════════════════════════════════════════════

def _sim_sbert(text1: str, text2: str) -> float:
    """Similarité sémantique via SBERT multilingue."""
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = _sbert.encode(text1, convert_to_tensor=True)
    emb2 = _sbert.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])


def _sim_tfidf(text1: str, text2: str) -> float:
    """Similarité lexicale TF-IDF — fallback si SBERT absent."""
    t1 = _clean(text1)
    t2 = _clean(text2)
    if not t1 or not t2:
        return 0.0
    try:
        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
        X = vect.fit_transform([t1, t2])
        return float(cosine_similarity(X[0:1], X[1:2])[0][0])
    except Exception:
        return 0.0


def _sim(text1: str, text2: str) -> float:
    """Choisit SBERT si disponible, sinon TF-IDF."""
    if SBERT_OK:
        return _sim_sbert(text1, text2)
    return _sim_tfidf(text1, text2)


# ════════════════════════════════════════════════
# 3. POIDS DYNAMIQUE KMEANS
# ════════════════════════════════════════════════

def _adjust_weight(score_base: float, poids: float) -> float:
    try:
        poids = float(poids)
    except Exception:
        poids = 1.0

    poids = max(0.1, poids)

    if abs(poids - 1.0) < 1e-9:
        return 1.0
    if score_base >= 90 and poids > 1.0:
        return 1.0 + (poids - 1.0) / 10.0
    if score_base >= 75 and poids > 1.0:
        return 1.0 + (poids - 1.0) / 3.0
    return poids


# ════════════════════════════════════════════════
# 4. SCORE PRINCIPAL
# ════════════════════════════════════════════════

def calculer_score(cv_json: dict, offre_json: dict, poids: float = 1.0) -> dict:
    """
    Calcule le score de compatibilité CV ↔ Offre.

    Paramètres :
        cv_json    : {"type": "cv",    "sections": {...}}
        offre_json : {"type": "offre", "sections": {...}}
        poids      : poids KMeans#1 depuis metier.py

    Retourne :
        {
            "score_final"  : 0..100,
            "niveau"       : "Excellent | Très Bon | Bon | Moyen | Faible",
            "methode"      : "SBERT | TF-IDF",
            "detail"       : {...}
        }
    """
    cv    = cv_json.get("sections",    {}) or {}
    offre = offre_json.get("sections", {}) or {}

    # ── Textes clés CV
    cv_skills  = _merge(cv, ["SKILL", "PROJECT", "ACHIEVEMENT"])
    cv_exp     = _merge(cv, ["EXPERIENCE", "PROJECT"])
    cv_edu     = _merge(cv, ["EDUCATION", "HEADER"])
    cv_lang    = _merge(cv, ["LANGUAGE"])
    cv_loc     = _merge(cv, ["LOCATION"])
    cv_global  = _merge_all(cv)

    # ── Textes clés Offre
    offre_req  = _merge(offre, ["REQUIREMENT", "RESPONSIBILITY", "SKILL"])
    offre_exp  = _merge(offre, ["EXPERIENCE",  "RESPONSIBILITY"])
    offre_edu  = _merge(offre, ["EDUCATION",   "REQUIREMENT"])
    offre_lang = _merge(offre, ["LANGUAGE"])
    offre_loc  = _merge(offre, ["LOCATION"])
    offre_glob = _merge_all(offre)

    # ── Calcul similarités par section
    sim_skill = _sim(cv_skills, offre_req)   if cv_skills and offre_req  else 0.0
    sim_exp   = _sim(cv_exp,    offre_exp)   if cv_exp    and offre_exp  else 0.0
    sim_edu   = _sim(cv_edu,    offre_edu)   if cv_edu    and offre_edu  else 0.0
    sim_lang  = _sim(cv_lang,   offre_lang)  if cv_lang   and offre_lang else 0.0
    sim_loc   = _sim(cv_loc,    offre_loc)   if cv_loc    and offre_loc  else 0.0
    sim_glob  = _sim(cv_global, offre_glob)

    # ── Poids sections
    W = {
        "SKILL":      0.40,
        "EXPERIENCE": 0.25,
        "EDUCATION":  0.15,
        "LANGUAGE":   0.05,
        "LOCATION":   0.05,
        "GLOBAL":     0.10,
    }

    score_base = (
        sim_skill * W["SKILL"]      +
        sim_exp   * W["EXPERIENCE"] +
        sim_edu   * W["EDUCATION"]  +
        sim_lang  * W["LANGUAGE"]   +
        sim_loc   * W["LOCATION"]   +
        sim_glob  * W["GLOBAL"]
    ) * 100

    # ── Fallback : si score très bas, utiliser global
    if score_base < 15:
        score_base = max(score_base, sim_glob * 100 * 0.8)

    # ── Poids KMeans#1
    poids_eff   = _adjust_weight(score_base, poids)
    score_final = max(0.0, min(100.0, score_base * poids_eff))

    # ── Niveau
    if score_final >= 80:
        niveau = "Excellent"
    elif score_final >= 65:
        niveau = "Très Bon"
    elif score_final >= 50:
        niveau = "Bon"
    elif score_final >= 35:
        niveau = "Moyen"
    else:
        niveau = "Faible"

    return {
        "score_final": round(score_final, 2),
        "niveau":      niveau,
        "methode":     "SBERT multilingue" if SBERT_OK else "TF-IDF",
        "detail": {
            "SKILL":       round(sim_skill * 100, 2),
            "EXPERIENCE":  round(sim_exp   * 100, 2),
            "EDUCATION":   round(sim_edu   * 100, 2),
            "LANGUAGE":    round(sim_lang  * 100, 2),
            "LOCATION":    round(sim_loc   * 100, 2),
            "global":      round(sim_glob  * 100, 2),
            "score_base":  round(score_base,      2),
            "poids_recu":  poids,
            "poids_effectif": round(poids_eff,    4),
        }
    }