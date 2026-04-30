# recommendation_general.py
import os, re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
PROFILES_PATH = os.path.join(DATA_DIR, "domain_profiles_general.csv")
CATALOG_PATH = os.path.join(DATA_DIR, "skills_catalog_general.csv")

def _clean(text):
    if not text:
        return ""
    text = str(text).lower()
    text = (text.replace("é","e").replace("è","e").replace("ê","e")
                .replace("à","a").replace("ù","u").replace("ô","o")
                .replace("’", "'"))
    text = re.sub(r"[^a-z0-9\s+/\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _split_list(value):
    if not value:
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]

def _load_profiles():
    if not os.path.exists(PROFILES_PATH):
        raise FileNotFoundError(PROFILES_PATH)
    df = pd.read_csv(PROFILES_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df

def _load_catalog():
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(CATALOG_PATH)
    df = pd.read_csv(CATALOG_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df

def _build_vocab(catalog_df):
    skills = set()
    for s in catalog_df["skill"].dropna().astype(str).tolist():
        s = _clean(s)
        if len(s) >= 2:
            skills.add(s)
    return sorted(skills, key=len, reverse=True)

def _extract_skills(text, vocab):
    t = _clean(text)
    found = []
    padded = f" {t} "
    for skill in vocab:
        s = _clean(skill)
        if " " in s:
            if f" {s} " in padded:
                found.append(s)
        else:
            if re.search(rf"\b{re.escape(s)}\b", t):
                found.append(s)
    out, seen = [], set()
    for x in found:
        xx = _clean(x)
        if xx not in seen:
            seen.add(xx)
            out.append(xx)
    return out

def _detect_domain(text, profiles_df):
    texts, domains = [], []
    for _, row in profiles_df.iterrows():
        dom = str(row["domaine"])
        profile_text = " ".join([
            dom,
            str(row.get("keywords", "")),
            str(row.get("core_skills", "")),
            str(row.get("adjacent_skills", "")),
            str(row.get("soft_skills", "")),
            str(row.get("typical_roles", "")),
        ])
        domains.append(dom)
        texts.append(_clean(profile_text))
    query = _clean(text)
    corpus = texts + [query]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    best = int(np.argmax(sims))
    top3_idx = np.argsort(sims)[-3:][::-1]
    top3 = [(domains[i], float(sims[i])) for i in top3_idx]
    return domains[best], float(sims[best]), top3

def _format_text(domain, conf, manquants, bonus, resume):
    lines = ["Recommandations", "", f"Domaine detecte : {domain} (confiance {conf:.2f})", f"Resume : {resume}", ""]
    lines.append("Competences a ajouter dans le CV par rapport a l offre :")
    if manquants:
        lines.extend([f"- {x}" for x in manquants])
    else:
        lines.append("- Aucune competence evidente n'a ete detectee automatiquement")
    lines.append("")
    lines.append("Competences utiles pour progresser dans ce domaine :")
    if bonus:
        lines.extend([f"- {x}" for x in bonus])
    else:
        lines.append("- Aucune suggestion supplementaire")
    lines.append("")
    lines.append("Conseil final :")
    if manquants:
        lines.append("- Ajoutez les competences demandees directement dans votre CV.")
    else:
        lines.append("- Renforcez vos projets, vos resultats et vos experiences concretes.")
    if bonus:
        lines.append("- Ajoutez 1 ou 2 competences adjacentes pour mieux vous differencier.")
    return "\n".join(lines)

def generer_recommandations(cv_sections, offre_sections, score=None):
    try:
        profiles_df = _load_profiles()
        catalog_df = _load_catalog()
        vocab = _build_vocab(catalog_df)

        cv_dict = cv_sections.get("sections", {}) if isinstance(cv_sections, dict) else {}
        off_dict = offre_sections.get("sections", {}) if isinstance(offre_sections, dict) else {}

        cv_text = " ".join([cv_dict.get("SKILL",""), cv_dict.get("EXPERIENCE",""), cv_dict.get("EDUCATION",""), cv_dict.get("PROJECT",""), cv_dict.get("LANGUAGE","")])
        offre_text = " ".join([off_dict.get("HEADER",""), off_dict.get("REQUIREMENT",""), off_dict.get("RESPONSIBILITY",""), off_dict.get("EDUCATION",""), off_dict.get("EXPERIENCE",""), off_dict.get("BENEFIT","")])

        skills_cv = set(_extract_skills(cv_text, vocab))
        skills_offre = set(_extract_skills(offre_text, vocab))

        domain, conf, top3 = _detect_domain(offre_text, profiles_df)
        row = profiles_df[profiles_df["domaine"] == domain].iloc[0]
        core = _split_list(row.get("core_skills", ""))
        adj = _split_list(row.get("adjacent_skills", ""))
        soft = _split_list(row.get("soft_skills", ""))

        manquants = sorted([s for s in (skills_offre - skills_cv) if len(s) >= 2])
        if not manquants:
            for s in core:
                ss = _clean(s)
                if ss and ss not in skills_cv and ss not in skills_offre:
                    manquants.append(ss)
                if len(manquants) >= 5:
                    break

        bonus = []
        for s in core + adj + soft:
            ss = _clean(s)
            if not ss:
                continue
            if ss in skills_cv or ss in skills_offre:
                continue
            if ss in bonus:
                continue
            bonus.append(ss)

        banned = {"junior","senior","management","budget","reporting","maintenance","qualite","pilotage","agile","service","client","analyse","suivi"}
        bonus = [b for b in bonus if b not in banned]
        manquants = manquants[:10]
        bonus = bonus[:10]

        score_val = 0
        if score and isinstance(score, dict):
            try:
                score_val = float(score.get("score_final", 0) or 0)
            except Exception:
                score_val = 0

        if score_val < 30:
            resume = "Votre profil est tres eloigne de l'offre. Priorite aux competences essentielles du poste."
        elif score_val < 60:
            resume = "Votre profil est partiellement adapte. Quelques ameliorations sont recommandees."
        else:
            resume = "Votre profil est proche. Une optimisation legere peut encore renforcer la candidature."

        return {
            "status": "success",
            "cluster": domain,
            "confidence": round(conf, 4),
            "top_domains": top3,
            "manquants": manquants,
            "bonus": bonus,
            "resume": resume,
            "recommendation_text": _format_text(domain, conf, manquants, bonus, resume),
            "blocks": [
                {"title": "Domaine detecte", "value": f"{domain} (confiance {conf:.2f})"},
                {"title": "Competences a ajouter", "items": manquants},
                {"title": "Competences utiles a renforcer", "items": bonus},
                {"title": "Resume", "value": resume},
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "cluster": "", "confidence": 0.0, "top_domains": [], "manquants": [], "bonus": [], "resume": "", "recommendation_text": "", "blocks": []}

if __name__ == "__main__":
    print("Use from main.py")
