# main.py
import os
import json

from extraction import extraire_cv, extraire_offre
from classification_section import classifier_cv, classifier_offre

# ── Modules futurs (activés automatiquement quand livrés)
try:
    from script_metier import detecter_metier_cluster
    METIER_READY = True
except ImportError:
    METIER_READY = False

try:
    from script_score import calculer_score
    SCORE_READY = True
except ImportError:
    SCORE_READY = False

try:
    from script_recommendation import generer_recommandations
    RECO_READY = True
except ImportError:
    RECO_READY = False


# ════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════

def pipeline(cv_path: str, json_offre: str, debug=False) -> dict:
    """
    Point d'entrée unique pour l'API backend.

    Paramètres :
        cv_path    : chemin absolu ou relatif vers le CV (PDF ou DOCX)
        json_offre : string JSON → '{"offre": "texte de l offre"}'
        debug      : affiche les logs intermédiaires

    Retourne :
        {
            "status"          : "success" | "error",
            "metier_detecte"  : "Ingénieur IT / Développement",
            "cluster_maroc"   : "Secteur Bancaire — Poids x1.3",
            "cv"              : { "type": "cv",    "sections": { ... } },
            "offre"           : { "type": "offre", "sections": { ... } },
            "score"           : { ... },
            "recommandations" : [ ... ]
        }
    """
    try:

        # ── 0. SÉCURISER CHEMIN CV
        if not os.path.isabs(cv_path):
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            cv_path  = os.path.abspath(os.path.join(BASE_DIR, cv_path))

        if not os.path.exists(cv_path):
            raise FileNotFoundError(f"CV introuvable : {cv_path}")

        if debug:
            print(f"\n[PIPELINE] CV PATH : {cv_path}")

        # ── 1. EXTRACTION
        if debug: print("\n[PIPELINE] Étape 1 — Extraction...")
        json_cv_brut    = extraire_cv(cv_path)
        json_offre_brut = extraire_offre(json_offre)

        if debug:
            print(f"  CV     ({len(json_cv_brut['content'])} chars)")
            print(f"  Offre  ({len(json_offre_brut['content'])} chars)")

        # ── 2. CLASSIFICATION
        if debug: print("\n[PIPELINE] Étape 2 — Classification sections...")
        cv_sections    = classifier_cv(json_cv_brut,       debug=debug)
        offre_sections = classifier_offre(json_offre_brut, debug=debug)

        # ── 3. MÉTIER + CLUSTER MAROC
        if debug: print("\n[PIPELINE] Étape 3 — Détection métier & cluster...")
        if METIER_READY:
            metier_cluster = detecter_metier_cluster(offre_sections["sections"])
        else:
            metier_cluster = {
                "metier_detecte": "en_cours",
                "cluster_maroc": "en_cours",
                "poids": 1.0
            }
        if debug:
            print(f"  Métier  : {metier_cluster['metier_detecte']}")
            print(f"  Cluster : {metier_cluster['cluster_maroc']}")
            print(f"  Poids   : {metier_cluster['poids']}")

        # ── 4. SCORE
        # Poids KMeans#1 sera injecté ici quand script_score livré
        if debug: print("\n[PIPELINE] Étape 4 — Score...")
        if SCORE_READY:
            score = calculer_score(
                cv_sections,
                offre_sections,
                poids=metier_cluster["poids"]   # ← KMeans#1 ready
            )
        else:
            score = {
                "status":  "en_cours",
                "message": "Module score en développement — KMeans#1 requis",
                "poids_prevu": metier_cluster["poids"]
            }

        # ── 5. RECOMMANDATIONS
        if debug: print("\n[PIPELINE] Étape 5 — Recommandations...")
        if RECO_READY:
            recommandations = generer_recommandations(
                cv_sections,
                offre_sections,
                score
            )
        else:
            recommandations = {
                "status":  "en_cours",
                "message": "Module recommandation en développement — KMeans#2 requis"
            }

        # ── 6. JSON FINAL → BACKEND
        resultat = {
            "status":          "success",
            "metier_detecte":  metier_cluster["metier_detecte"],
            "cluster_maroc":   metier_cluster["cluster_maroc"],
            "cv":              cv_sections,
            "offre":           offre_sections,
            "score":           score,
            "recommandations": recommandations
        }

        if debug:
            print("\n[PIPELINE] ✅ Pipeline terminé avec succès")

        return resultat

    except FileNotFoundError as e:
        return {"status": "error", "code": "FILE_NOT_FOUND", "message": str(e)}
    except ValueError as e:
        return {"status": "error", "code": "VALUE_ERROR",    "message": str(e)}
    except Exception as e:
        return {"status": "error", "code": "UNKNOWN",        "message": str(e)}


# ════════════════════════════════════════════════
# TEST LOCAL
# ════════════════════════════════════════════════

if __name__ == "__main__":

    cv_path = "test/2.pdf"

    offre_json = json.dumps({
        "offre": "Developpeur Python. Competences: Python, Django. Experience 2 ans. Localisation Rabat."
    }, ensure_ascii=False)

    result = pipeline(cv_path, offre_json, debug=True)

    print("\n=== RÉSULTAT FINAL ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))