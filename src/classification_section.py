# classification_section.py
"""
Classification universelle des sections CV et Offre.
Pipeline : Segmentation → Contexte glissant → NN → Post-correction (règles + datasets marocains) → Propagation
Compatible avec tout format de CV (1/2/3 colonnes, toutes structures) et offres LinkedIn.
27 domaines marocains couverts.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ════════════════════════════════════════════════
# 0. PATHS
# ════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "referentiels")


# ════════════════════════════════════════════════
# 1. DATASETS MAROCAINS
# ════════════════════════════════════════════════

def _charger_set(fichier: str, colonne: str) -> set:
    path = os.path.join(DATA_DIR, fichier)
    if not os.path.exists(path):
        print(f"[WARN] Dataset introuvable : {path}")
        return set()
    df = pd.read_csv(path)
    return set(df[colonne].dropna().str.lower().str.strip())


_ECOLES = _charger_set("ecoles.csv", "nom")
_ENTREPRISES = _charger_set("entreprises.csv", "nom")
_FILIERES = _charger_set("filieres.csv", "filiere")
_DIPLOMES = _charger_set("filieres.csv", "diplome")
_METIERS = _charger_set("metiers.csv", "nom")
_VILLES = _charger_set("villes.csv", "nom")

# ════════════════════════════════════════════════
# 2. CACHE MODÈLES
# ════════════════════════════════════════════════

_MODELS = {}


class Classifier:
    def __init__(self, model_path, vect_path, enc_path):
        key = model_path
        if key in _MODELS:
            cached = _MODELS[key]
            self.model = cached["model"]
            self.vectorizer = cached["vectorizer"]
            self.encoder = cached["encoder"]
            return

        for p, nom in [(model_path, "Model"), (vect_path, "Vectorizer"), (enc_path, "Encoder")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{nom} introuvable : {p}")

        self.model = load_model(model_path)
        self.vectorizer = joblib.load(vect_path)
        self.encoder = joblib.load(enc_path)

        _MODELS[key] = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "encoder": self.encoder
        }


# ════════════════════════════════════════════════
# 3. NORMALISATION POUR NN
# ════════════════════════════════════════════════

def _normaliser_pour_nn(text: str) -> str:
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "ù": "u", "û": "u", "ü": "u",
        "ô": "o", "ö": "o",
        "î": "i", "ï": "i",
        "ç": "c"
    }
    text = text.lower()
    for acc, let in remplacements.items():
        text = text.replace(acc, let)
    return text


# ════════════════════════════════════════════════
# 4. TITRES DE SECTIONS
# ════════════════════════════════════════════════

_TITRES_SECTIONS = {
    # CV — contact
    "coordonnees": "CONTACT", "coordonnées": "CONTACT",
    "contact": "CONTACT", "contacts": "CONTACT",
    "informations": "CONTACT", "infos": "CONTACT",
    "mes informations": "CONTACT", "info": "CONTACT",
    "informations personnelles": "CONTACT",
    # CV — formation
    "formation": "EDUCATION", "formations": "EDUCATION",
    "education": "EDUCATION", "etudes": "EDUCATION",
    "études": "EDUCATION", "parcours academique": "EDUCATION",
    "parcours scolaire": "EDUCATION", "diplomes": "EDUCATION",
    # CV — expérience
    "experience": "EXPERIENCE", "experiences": "EXPERIENCE",
    "expérience": "EXPERIENCE", "expériences": "EXPERIENCE",
    "parcours professionnel": "EXPERIENCE",
    "experience professionnelle": "EXPERIENCE",
    # CV — compétences
    "competences": "SKILL", "compétences": "SKILL",
    "skills": "SKILL", "competences techniques": "SKILL",
    "informatique": "SKILL", "outils": "SKILL",
    "technologies": "SKILL", "hard skills": "SKILL",
    "soft skills": "SKILL", "savoir faire": "SKILL",
    # CV — langues
    "langues": "LANGUAGE", "langue": "LANGUAGE",
    "languages": "LANGUAGE", "langues etrangeres": "LANGUAGE",
    # CV — intérêts
    "centres d interet": "INTEREST",
    "centres d intérêt": "INTEREST",
    "interets": "INTEREST", "intérêts": "INTEREST",
    "hobbies": "INTEREST", "loisirs": "INTEREST",
    "activites": "INTEREST", "activités": "INTEREST",
    "interet": "INTEREST", "intérêt": "INTEREST",
    # CV — projets
    "projets": "PROJECT", "projet": "PROJECT",
    "projects": "PROJECT", "projets academiques": "PROJECT",
    "projets personnels": "PROJECT",
    # CV — réalisations
    "certifications": "ACHIEVEMENT", "certification": "ACHIEVEMENT",
    "distinctions": "ACHIEVEMENT", "prix": "ACHIEVEMENT",
    "realisations": "ACHIEVEMENT", "réalisations": "ACHIEVEMENT",
    "accomplissements": "ACHIEVEMENT",
    # Offre — missions
    "missions": "RESPONSIBILITY", "mission": "RESPONSIBILITY",
    "responsabilites": "RESPONSIBILITY", "responsabilités": "RESPONSIBILITY",
    "taches": "RESPONSIBILITY", "tâches": "RESPONSIBILITY",
    "missions principales": "RESPONSIBILITY",
    "votre role": "RESPONSIBILITY", "votre mission": "RESPONSIBILITY",
    # Offre — profil
    "profil": "REQUIREMENT", "profil recherche": "REQUIREMENT",
    "profil requis": "REQUIREMENT", "requis": "REQUIREMENT",
    "requirements": "REQUIREMENT", "exigences": "REQUIREMENT",
    "votre profil": "REQUIREMENT", "nous recherchons": "REQUIREMENT",
    # Offre — avantages
    "avantages": "BENEFIT", "benefits": "BENEFIT",
    "nous offrons": "BENEFIT", "ce que nous offrons": "BENEFIT",
    "avantages sociaux": "BENEFIT",
    # Offre — salaire
    "salaire": "SALARY", "remuneration": "SALARY",
    "rémunération": "SALARY", "package": "SALARY",
    # Offre — contrat
    "contrat": "CONTRACT", "type de contrat": "CONTRACT",
    "type contrat": "CONTRACT",
    # Offre — localisation
    "localisation": "LOCATION", "lieu": "LOCATION",
    "lieu de travail": "LOCATION", "ville": "LOCATION",
    "adresse": "LOCATION",
}


def _detecter_titre_section(texte_norm: str) -> str | None:
    t = texte_norm.strip().rstrip(":")
    for titre, label in _TITRES_SECTIONS.items():
        if t == titre or t.startswith(titre + " ") or t.startswith(titre + ":"):
            return label
    return None


# ════════════════════════════════════════════════
# 5. SEGMENTATION UNIVERSELLE
# ════════════════════════════════════════════════

def _segmenter_texte(text: str) -> list:
    if not text:
        return []

    blocs = []
    for ligne in text.splitlines():
        ligne = ligne.strip()
        if len(ligne) < 2:
            continue

        if len(ligne) > 150:
            words = ligne.split()
            chunk = []
            for w in words:
                chunk.append(w)
                if len(chunk) >= 15:
                    blocs.append(" ".join(chunk))
                    chunk = []
            if chunk:
                blocs.append(" ".join(chunk))
        else:
            blocs.append(ligne)

    return blocs


# ════════════════════════════════════════════════
# 6. CONTEXTE GLISSANT
# ════════════════════════════════════════════════

def _enrichir_avec_contexte(blocs: list) -> list:
    blocs_enrichis = []
    for i, bloc in enumerate(blocs):
        avant = blocs[i - 1] if i > 0 else ""
        apres = blocs[i + 1] if i < len(blocs) - 1 else ""
        contexte = f"{avant} {bloc} {apres}".strip()
        contexte = re.sub(r' {2,}', ' ', contexte)
        blocs_enrichis.append(contexte)
    return blocs_enrichis


# ════════════════════════════════════════════════
# 7. PATTERNS — 27 DOMAINES MAROCAINS
# ════════════════════════════════════════════════

PATTERNS_CONTACT = [
    r'\+?\d{9,13}',
    r'[\w\.\-]+@[\w\.\-]+\.\w+',
    r'linkedin\.com',
    r'github\.com',
    r'portfolio\.',
    r'twitter\.com',
    r'www\.',
]

PATTERNS_LANGUAGE = [
    r'\b(francais|arabe|anglais|espagnol|allemand|italien|amazigh|tamazight|chinois|russe|portugais)\b',
    r'\b(bilingue|multilingue|langue maternelle|intermediaire|courant|natif|avance|debutant|notions)\b',
    r'\b(toefl|toeic|dalf|delf|ielts|cambridge|bulats)\b',
    r'\b(niveau [ab][12]|niveau [bc][12])\b',
]
PATTERNS_INTEREST = [
    r'\b(lecture|sport|football|basketball|voyage|musique|cinema|cuisine|photographie)\b',
    r'\b(robotique|innovation|entrepreneuriat|volontariat|associatif|benevolat)\b',
    r'\b(centres d interet|loisirs|hobbies|activites extra|passion)\b',
]

PATTERNS_SKILL = [
    # IT / Informatique
    r'\b(python|java|sql|docker|git|html|css|javascript|typescript|react|vue|angular|node|php|kotlin|swift|rust|go|c\+\+|c#|ruby|scala|perl|bash)\b',
    r'\b(machine learning|deep learning|nlp|tensorflow|keras|sklearn|pytorch|opencv|pandas|numpy|scipy|matplotlib|seaborn|huggingface)\b',
    r'\b(mysql|mongodb|postgresql|oracle|redis|firebase|elasticsearch|cassandra|sqlite|mariadb|dynamodb)\b',
    r'\b(aws|azure|gcp|kubernetes|jenkins|ci.?cd|devops|terraform|ansible|nginx|apache|linux|ubuntu)\b',
    r'\b(django|flask|fastapi|spring|laravel|symfony|dotnet|express|nextjs|nuxtjs|wordpress|drupal)\b',
    r'\b(jira|confluence|trello|github|gitlab|bitbucket|postman|swagger|figma|notion)\b',
    r'\b(uml|merise|agile|scrum|kanban|lean|itil|cobit|togaf|prince2|pmp)\b',
    # Data / BI
    r'\b(power bi|tableau|looker|qlikview|sas|spss|stata|vba|powerquery|dax)\b',
    r'\b(data warehouse|etl|datamart|hadoop|spark|kafka|airflow|dbt|snowflake|bigquery|databricks)\b',
    # Telecoms & Réseaux
    r'\b(cisco|juniper|mikrotik|routeur|switch|vlan|vpn|firewall|wifi|lte|5g|voip|sip|gsm)\b',
    r'\b(tcp.?ip|bgp|ospf|mpls|sd.?wan|noc|umts|fibre optique|adsl|wimax|wireshark|packet tracer|gns3)\b',
    # Électricité & Énergie
    r'\b(haute tension|basse tension|habilitation|transformateur|onduleur|variateur|disjoncteur|relayage)\b',
    r'\b(automate|plc|scada|supervision|schneider|siemens|abb|legrand|hager|eaton)\b',
    r'\b(photovoltaique|solaire|eolien|stockage energie|smart grid|reseau electrique|poste ht)\b',
    # Électronique & Automatique
    r'\b(arduino|raspberry|pic|stm32|fpga|vhdl|verilog|labview|proteus|eagle|kicad|altium)\b',
    r'\b(microcontroleur|capteur|actionneur|asservissement|regulation|pid|embarque|iot|robotique|drone)\b',
    r'\b(simulink|electronique|circuit|pcb)\b',
    # Génie Mécanique & Industriel
    r'\b(solidworks|catia|inventor|fusion 360|ansys|abaqus|comsol|hypermesh|nastran)\b',
    r'\b(usinage|fraisage|tournage|soudure|fabrication|cfao|cao|impression 3d|prototypage)\b',
    r'\b(maintenance|gmao|tpm|lean manufacturing|kaizen|5s|six sigma|amdec|smed|vsm)\b',
    # Génie Civil & BTP
    r'\b(autocad civil|revit|archicad|sketchup|bim|robot structural|advance design|covadis)\b',
    r'\b(beton|structure|fondations|charpente|coffrage|ferraillage|gros oeuvre|vrd)\b',
    r'\b(topographie|geodesie|gps|station totale|nivellement|sig|arcgis|qgis|mapinfo)\b',
    # Architecture & Urbanisme
    r'\b(rhino|grasshopper|lumion|artlantis|3ds max|vray|enscape|urbanisme|plu)\b',
    # Génie Chimique & Matériaux
    r'\b(chimie|polymeres|composites|metallurgie|corrosion|traitement surface|ceramique)\b',
    r'\b(distillation|extraction|reaction|catalyse|bioprocedes|procedes chimiques)\b',
    r'\b(ndt|essais mecaniques|thermique|caracterisation|drx|meb|spectrometrie)\b',
    # Environnement & Énergie Verte
    r'\b(hse|iso 14001|ohsas|evaluation impact|audit environnemental|bilan carbone)\b',
    r'\b(energie renouvelable|efficacite energetique|developpement durable|empreinte carbone)\b',
    r'\b(traitement eau|assainissement|dechets|recyclage|pollution|ecologie|eie)\b',
    # Aéronautique & Spatial
    r'\b(aeronautique|avionique|navigabilite|part 145|part 66|easa|hydraulique aeronautique)\b',
    r'\b(enovia|plm|composite aeronautique|maintenance aeronautique)\b',
    # Agriculture & Agroalimentaire
    r'\b(agronomie|irrigation|fertilisation|pesticides|semences|phytopathologie|elevage)\b',
    r'\b(haccp|iso 22000|tracabilite|bpm|qualite alimentaire|agroalimentaire)\b',
    # Finance & Banque
    r'\b(comptabilite|audit|controle gestion|tresorerie|fiscalite|consolidation|ifrs|gaap)\b',
    r'\b(sage|sap|cegid|oracle finance|analyse financiere|budget|reporting financier)\b',
    r'\b(banque|credit|risque|bale|conformite|kyc|aml|trading|marches financiers|actuariat)\b',
    # Commerce & Marketing
    r'\b(seo|sea|sem|google ads|facebook ads|crm|salesforce|hubspot|e.?commerce|shopify)\b',
    r'\b(business development|negociation|b2b|b2c|key account|trade marketing|growth hacking)\b',
    r'\b(google analytics|tag manager|mailchimp|hootsuite|content marketing|copywriting)\b',
    # Ressources Humaines
    r'\b(recrutement|onboarding|gpec|sirh|paie|formation rh|evaluation|droit social)\b',
    r'\b(silae|sage paie|adp|peoplesoft|workday|successfactors|taleo|smartrecruiters)\b',
    # Santé & Médecine
    r'\b(his|ris|pacs|dossier patient|imagerie|radiologie|echographie|scanner|irm)\b',
    r'\b(pharmacie|biologie|analyse medicale|sterilisation|bloc operatoire|urgences)\b',
    r'\b(sante publique|epidemiologie|biostatistique|essais cliniques|pharmacovigilance)\b',
    # Droit & Sciences Juridiques
    r'\b(droit des affaires|droit social|droit penal|contentieux|contrats|compliance)\b',
    r'\b(propriete intellectuelle|brevet|arbitrage|mediation|notariat|jurisprudence)\b',
    # Education & Enseignement
    r'\b(pedagogie|didactique|e.?learning|lms|moodle|tutorat|curriculum|apprentissage)\b',
    # Arts & Communication
    r'\b(photoshop|illustrator|indesign|premiere|after effects|davinci|blender|cinema 4d)\b',
    r'\b(montage|motion design|animation|graphisme|identite visuelle|journalisme|redaction)\b',
    r'\b(relations presse|relations publiques|communication institutionnelle)\b',
    # Management & Entrepreneuriat
    r'\b(leadership|strategie|business plan|lean startup|mvp|gouvernance|transformation digitale)\b',
    r'\b(gestion projet|msp|portfolio|programme)\b',
    # Sécurité & Défense
    r'\b(cybersecurite|soc|siem|pentest|ethical hacking|iso 27001|rgpd|forensic|osint)\b',
    r'\b(securite physique|surete|gardiennage|videoprotection|controle acces|incendie)\b',
    # Sport & Education Physique
    r'\b(coaching sportif|entrainement|performance sportive|physiologie|biomecanique|kinesitherapie)\b',
    # Tourisme & Hôtellerie
    r'\b(hotellerie|restauration|accueil|reservation|pms|opera|fidelio|yield management)\b',
    r'\b(tourisme|guide touristique|agence voyage|billetterie|ecotourisme|mice|gastronomie)\b',
    # Transport & Logistique
    r'\b(supply chain|logistique|wms|tms|sap mm|sap wm|sap sd|entrepot|stockage)\b',
    r'\b(affrètement|douane|incoterms|freight|import export|transit)\b',
    # Secteur Public
    r'\b(marche public|appel offre|comptabilite publique|e.?gouvernement|collectivite)\b',
    # Textile & Mode
    r'\b(textile|confection|coupe|patronage|lectra|gerber|modaris|diamino|stylisme)\b',
    # Sciences & Recherche
    r'\b(laboratoire|protocole|publication|article scientifique|peer review|modelisation|simulation)\b',
    # Sciences Humaines
    r'\b(psychologie|sociologie|anthropologie|travail social|enquete|analyse qualitative)\b',
    r'\b(nvivo|atlas.ti|questionnaire|ethnographie|terrain)\b',
]

PATTERNS_EDUCATION = [
    r'\b(bac\s*\+\s*\d)\b',
    r'\b(20\d\d\s*[-–]\s*20\d\d)\b',
    r'\b(these|doctorat|phd)\b',
    r'\b(master|licence|bachelor|ingenieur|diplome|deug|dut|bts|cpge|prepa|dess|dea)\b',
    r'\b(universite|faculte|ecole superieure|institut superieur|grande ecole|classe preparatoire)\b',
]

PATTERNS_HEADER = [
    r'\b(etudiant|ingenieur|developpeur|technicien|manager|directeur|chef|stagiaire|charge)\b',
    r'\b(junior|senior|consultant|analyste|architecte|responsable|coordinateur|lead|expert)\b',
    r'\b(medecin|infirmier|comptable|auditeur|juriste|avocat|pharmacien|enseignant|formateur)\b',
    r'\b(designer|data scientist|chercheur|doctorant|post.?doc|coach|trader|gerant)\b',
]

PATTERNS_LOCATION = [
    r'\b(rue|avenue|bd|boulevard|lot|hay|quartier|cite|n°|appt|immeuble|residence)\b',
    r'\b(douar|mechta|lotissement|commune|cercle|province|prefecture|region)\b',
    r'\b(maroc|france|belgique|canada|emirats|arabie saoudite|tunisie|algerie|senegal)\b',
]

PATTERNS_CONTRACT = [
    r'\b(cdi|cdd|stage|alternance|freelance|interim|temps plein|temps partiel|full.?time|part.?time|mission)\b',
]

PATTERNS_SALARY = [
    r'\b\d{4,6}\s*(dh|mad|eur|euro|dollar|\$|€|k€)\b',
    r'\b(salaire|remuneration|package|compensation|selon profil|a negocier)\b',
]

PATTERNS_RESPONSIBILITY = [
    r'\b(recueillir|rediger|participer|assurer|preparer|contribuer|animer|coordonner)\b',
    r'\b(elaborer|concevoir|developper|mettre en place|piloter|superviser|gerer|realiser)\b',
    r'\b(effectuer|conduire|mener|suivre|controler|valider|produire|livrer|deployer)\b',
]

PATTERNS_NOM_MAJUSCULES = re.compile(r'^[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ\s\-]{4,}$')
PATTERNS_NOM_PROPRE = re.compile(
    r'^[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][a-zàâéèêëîïôùûüç\-]+'
    r'(\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][a-zàâéèêëîïôùûüç\-]+)+$'
)


# ════════════════════════════════════════════════
# 8. MATCH DATASET
# ════════════════════════════════════════════════

def _match_dataset(texte_norm: str, dataset: set, seuil: int = 4) -> bool:
    mots = texte_norm.split()
    for mot in mots:
        if len(mot) >= seuil and mot in dataset:
            return True
    for i in range(len(mots) - 1):
        if mots[i] + " " + mots[i + 1] in dataset:
            return True
    return False


# ════════════════════════════════════════════════
# 9. POST-CORRECTION
# ════════════════════════════════════════════════

def _post_corriger(predictions: list, labels_possibles: list) -> list:
    """
    Ordre de priorité :
    Nom propre > Contact > Language >
    Education (datasets + patterns) > Skill >
    Location > Experience > Responsibility > Contract > Salary > Header

    IMPORTANT : Education avant Skill pour éviter que
    les filières/diplômes soient classés comme Skills.
    """
    corrections = []
    label_set = set(labels_possibles)

    for texte, label in predictions:
        t = _normaliser_pour_nn(texte)
        nouveau = label

        # ── 0. NOM EN MAJUSCULES → HEADER
        if PATTERNS_NOM_MAJUSCULES.match(texte.strip()) and len(texte.strip()) > 4:
            nouveau = "HEADER"

        # ── 0b. NOM PROPRE CAPITALISÉ → HEADER
        elif PATTERNS_NOM_PROPRE.match(texte.strip()) \
                and len(texte.strip().split()) <= 4:
            if label not in ("EDUCATION", "EXPERIENCE", "SKILL", "LOCATION"):
                nouveau = "HEADER"

        # ── 1. CONTACT
        elif any(re.search(p, t) for p in PATTERNS_CONTACT):
            nouveau = "CONTACT"

        # ── 2. LANGUAGE
        elif any(re.search(p, t) for p in PATTERNS_LANGUAGE):
            nouveau = "LANGUAGE"

        # ── 3. EDUCATION — écoles marocaines (AVANT SKILL)
        elif _match_dataset(t, _ECOLES):
            nouveau = "EDUCATION"

        # ── 4. EDUCATION — diplômes marocains (AVANT SKILL)
        elif _match_dataset(t, _DIPLOMES, seuil=3):
            nouveau = "EDUCATION"

        # ── 5. EDUCATION — filières marocaines (AVANT SKILL)
        elif _match_dataset(t, _FILIERES):
            nouveau = "EDUCATION"

        # ── 6. EDUCATION — patterns généraux (AVANT SKILL)
        elif any(re.search(p, t) for p in PATTERNS_EDUCATION):
            nouveau = "EDUCATION"

        # ── 2b. INTEREST — avant SKILL pour éviter confusion
        elif any(re.search(p, t) for p in PATTERNS_INTEREST) \
                and "INTEREST" in label_set:
            nouveau = "INTEREST"

        # ── 7. SKILL — 27 domaines (APRÈS EDUCATION)
        elif any(re.search(p, t) for p in PATTERNS_SKILL):
            nouveau = "SKILL"

        # ── 8. LOCATION — villes marocaines (1408 villes)
        elif _match_dataset(t, _VILLES, seuil=3):
            if label not in ("EDUCATION", "EXPERIENCE", "SKILL"):
                nouveau = "LOCATION"

        # ── 9. LOCATION — adresse physique
        elif any(re.search(p, t) for p in PATTERNS_LOCATION):
            nouveau = "LOCATION"

        # ── 10. EXPERIENCE — entreprises marocaines (2516)
        elif _match_dataset(t, _ENTREPRISES):
            if label not in ("EDUCATION", "SKILL"):
                nouveau = "EXPERIENCE"

        # ── 11. RESPONSIBILITY (offre)
        elif any(re.search(p, t) for p in PATTERNS_RESPONSIBILITY) \
                and "RESPONSIBILITY" in label_set:
            nouveau = "RESPONSIBILITY"

        # ── 12. CONTRACT (offre)
        elif any(re.search(p, t) for p in PATTERNS_CONTRACT) \
                and "CONTRACT" in label_set:
            nouveau = "CONTRACT"

        # ── 13. SALARY (offre)
        elif any(re.search(p, t) for p in PATTERNS_SALARY) \
                and "SALARY" in label_set:
            nouveau = "SALARY"

        # ── 14. HEADER — métiers marocains (1891)
        elif _match_dataset(t, _METIERS):
            if label not in ("EDUCATION", "EXPERIENCE", "SKILL", "LOCATION"):
                nouveau = "HEADER"

        # ── 15. HEADER — titres généraux
        elif any(re.search(p, t) for p in PATTERNS_HEADER):
            if label not in ("EDUCATION", "EXPERIENCE", "SKILL", "LOCATION"):
                nouveau = "HEADER"

        if nouveau not in label_set:
            nouveau = label

        corrections.append((texte, nouveau))

    return corrections


# ════════════════════════════════════════════════
# 10. PROPAGATION SÉQUENTIELLE
# ════════════════════════════════════════════════

def _propager_sections(predictions: list) -> list:
    """
    Propage le label de section aux blocs courts ambigus.
    Exception : les noms propres → toujours HEADER.
    """
    if not predictions:
        return predictions

    resultat = []
    section_active = None

    for texte, label in predictions:
        texte_norm = _normaliser_pour_nn(texte)
        titre_detecte = _detecter_titre_section(texte_norm)

        # Titre de section → activer et ne pas inclure dans contenu
        if titre_detecte:
            section_active = titre_detecte
            continue

        # Détecter nom propre — priorité absolue → HEADER
        est_nom_majuscules = (
                PATTERNS_NOM_MAJUSCULES.match(texte.strip())
                and len(texte.strip()) > 4
        )
        est_nom_propre = (
                PATTERNS_NOM_PROPRE.match(texte.strip())
                and len(texte.strip().split()) <= 4
        )

        if est_nom_majuscules or est_nom_propre:
            resultat.append((texte, "HEADER"))
            section_active = "HEADER"
            continue

        # Bloc court ambigu → hériter section active
        if section_active and len(texte.strip()) < 30:
            resultat.append((texte, section_active))
        else:
            section_active = label
            resultat.append((texte, label))

    return resultat


# ════════════════════════════════════════════════
# 11. PRÉDICTION COMPLÈTE
# ════════════════════════════════════════════════

def _predire(classifier: Classifier, blocs: list,
             labels_possibles: list) -> list:
    if not blocs:
        return []

    blocs_enrichis = _enrichir_avec_contexte(blocs)
    blocs_norm = [_normaliser_pour_nn(b) for b in blocs_enrichis]

    X = classifier.vectorizer.transform(blocs_norm).toarray()
    y_pred = np.argmax(classifier.model.predict(X, verbose=0), axis=1)
    labels = classifier.encoder.inverse_transform(y_pred)

    predictions = list(zip(blocs, labels))
    predictions = _post_corriger(predictions, labels_possibles)
    predictions = _propager_sections(predictions)

    return predictions


# ════════════════════════════════════════════════
# 12. REGROUPEMENT
# ════════════════════════════════════════════════

def _regrouper_sections(predictions: list, labels_possibles: list) -> dict:
    sections = {label: [] for label in labels_possibles}

    for texte, label in predictions:
        if label in sections:
            sections[label].append(texte)

    for k in sections:
        sections[k] = "\n".join(sections[k]).strip()

    return sections


# ════════════════════════════════════════════════
# 13. LABELS
# ════════════════════════════════════════════════

CV_LABELS = [
    "HEADER", "EDUCATION", "EXPERIENCE", "SKILL",
    "LOCATION", "LANGUAGE", "INTEREST",
    "CONTACT", "PROJECT", "ACHIEVEMENT"
]

OFFRE_LABELS = [
    "HEADER", "REQUIREMENT", "RESPONSIBILITY",
    "EDUCATION", "EXPERIENCE", "LOCATION",
    "CONTRACT", "SALARY", "LANGUAGE",
    "BENEFIT", "NOT_IMPORTANT"
]


# ════════════════════════════════════════════════
# 14. CLASSIFICATION CV
# ════════════════════════════════════════════════

def classifier_cv(json_cv: dict, debug=False) -> dict:
    clf = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_cv.h5"),
        vect_path=os.path.join(MODELS_DIR, "vectorizer_cv.pkl"),
        enc_path=os.path.join(MODELS_DIR, "encoder_cv.pkl")
    )

    texte = json_cv.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print(f"\n[DEBUG CV] {len(blocs)} blocs détectés")
        for b in blocs[:12]:
            print(f"  • {b[:80]}")

    predictions = _predire(clf, blocs, CV_LABELS)
    sections = _regrouper_sections(predictions, CV_LABELS)

    if debug:
        print("\n[DEBUG CV] Prédictions :")
        for t, l in predictions[:20]:
            print(f"  [{l:<12}] {t[:70]}")

    return {"type": "cv", "sections": sections}


# ════════════════════════════════════════════════
# 15. CLASSIFICATION OFFRE
# ════════════════════════════════════════════════

def classifier_offre(json_offre: dict, debug=False) -> dict:
    clf = Classifier(
        model_path=os.path.join(MODELS_DIR, "nn_offre.h5"),
        vect_path=os.path.join(MODELS_DIR, "vectorizer_offre.pkl"),
        enc_path=os.path.join(MODELS_DIR, "encoder_offre.pkl")
    )

    texte = json_offre.get("content", "")
    blocs = _segmenter_texte(texte)

    if debug:
        print(f"\n[DEBUG OFFRE] {len(blocs)} blocs détectés")
        for b in blocs[:12]:
            print(f"  • {b[:80]}")

    predictions = _predire(clf, blocs, OFFRE_LABELS)
    sections = _regrouper_sections(predictions, OFFRE_LABELS)

    if debug:
        print("\n[DEBUG OFFRE] Prédictions :")
        for t, l in predictions[:20]:
            print(f"  [{l:<14}] {t[:70]}")

    return {"type": "offre", "sections": sections}