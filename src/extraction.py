# extraction.py
import os
import json
import re
import docx
import fitz  # pymupdf
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ════════════════════════════════════════════════
# 1. DOCX
# ════════════════════════════════════════════════

def _extraire_docx(path: str) -> str:
    doc = docx.Document(path)
    lines = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            lines.append(t)

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))

    return "\n".join(lines)


# ════════════════════════════════════════════════
# 2. PDF — DÉTECTION AUTOMATIQUE
# ════════════════════════════════════════════════

def _extraire_pdf(path: str) -> str:
    """
    Détecte automatiquement si PDF natif ou scanné.
    Natif  → pymupdf + KMeans colonnes
    Scanné → tesseract OCR
    """
    doc = fitz.open(path)

    # ── Test : PDF natif ou scanné ?
    texte_test = ""
    for page in doc:
        texte_test += page.get_text("text")
        if len(texte_test.strip()) > 150:
            break

    doc.close()

    if len(texte_test.strip()) > 150:
        return _extraire_pdf_natif(path)
    else:
        return _extraire_pdf_scanné(path)


# ════════════════════════════════════════════════
# 3. PDF NATIF — pymupdf + KMeans colonnes
# ════════════════════════════════════════════════

def _detecter_nb_colonnes(blocs: list, largeur: float) -> int:
    """
    Détecte automatiquement 1, 2 ou 3 colonnes
    en clusterisant les positions X des blocs.
    """
    if len(blocs) < 3:
        return 1

    centres_x = np.array([
        ((b[0] + b[2]) / 2) / largeur
        for b in blocs
    ]).reshape(-1, 1)

    meilleur_k     = 1
    meilleur_score = -1

    for k in range(2, 4):  # tester 2 et 3 colonnes
        if len(centres_x) <= k:
            break
        try:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(centres_x)
            score  = silhouette_score(centres_x, labels)
            if score > 0.4 and score > meilleur_score:
                meilleur_score = score
                meilleur_k     = k
        except Exception:
            pass

    return meilleur_k


def _trier_blocs_par_colonnes(blocs: list, nb_colonnes: int,
                               largeur: float) -> list:
    """
    Trie les blocs colonne par colonne puis par Y.
    Gère le cas mixte (2 cols en haut, 3 cols en bas).
    """
    if nb_colonnes == 1:
        return sorted(blocs, key=lambda b: b[1])  # tri par Y

    largeur_col = largeur / nb_colonnes

    def cle_tri(bloc):
        centre_x = (bloc[0] + bloc[2]) / 2
        col      = min(int(centre_x // largeur_col), nb_colonnes - 1)
        return (col, bloc[1])  # (colonne, position Y)

    return sorted(blocs, key=cle_tri)


def _extraire_pdf_natif(path: str) -> str:
    doc        = fitz.open(path)
    texte_final = []

    for page in doc:
        largeur = page.rect.width

        # ── Extraire blocs (pymupdf regroupe phrases multi-lignes)
        blocs = page.get_text("blocks")

        # ── Garder uniquement blocs texte non vides
        blocs_texte = [
            b for b in blocs
            if b[6] == 0 and b[4].strip()
        ]

        if not blocs_texte:
            continue

        # ── Détecter nombre de colonnes
        nb_colonnes = _detecter_nb_colonnes(blocs_texte, largeur)

        # ── Trier par colonnes puis Y
        blocs_tries = _trier_blocs_par_colonnes(blocs_texte, nb_colonnes, largeur)

        # ── Extraire texte — phrase sur 2 lignes = 1 bloc = 1 ligne
        for bloc in blocs_tries:
            texte = bloc[4].strip().replace("\n", " ")
            if texte:
                texte_final.append(texte)

    doc.close()
    return "\n".join(texte_final)


# ════════════════════════════════════════════════
# 4. PDF SCANNÉ — tesseract OCR
# ════════════════════════════════════════════════

def _extraire_pdf_scanné(path: str) -> str:
    pages    = convert_from_path(path, dpi=300)
    all_text = []

    for image in pages:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang="fra+eng",
            config="--psm 11"
        )

        blocs = []
        for i in range(len(data["text"])):
            mot = data["text"][i].strip()
            try:
                conf = int(data["conf"][i])
            except Exception:
                conf = 0
            if conf >= 50 and mot:
                blocs.append({
                    "text": mot,
                    "x":    data["left"][i],
                    "y":    data["top"][i],
                })

        if not blocs:
            continue

        # ── Détecter colonnes sur l'image aussi
        largeur_image  = image.width
        centres_x      = np.array([b["x"] for b in blocs])
        nb_colonnes    = _detecter_nb_colonnes_ocr(centres_x, largeur_image)
        largeur_col    = largeur_image / nb_colonnes

        # ── Trier par colonne puis Y
        blocs = sorted(blocs, key=lambda b: (
            min(int(b["x"] // largeur_col), nb_colonnes - 1),
            b["y"]
        ))

        # ── Reconstruire lignes
        lignes    = []
        courante  = []
        y_prev    = None
        SEUIL_Y   = 12

        for b in blocs:
            if y_prev is None or abs(b["y"] - y_prev) <= SEUIL_Y:
                courante.append(b["text"])
            else:
                lignes.append(" ".join(courante))
                courante = [b["text"]]
            y_prev = b["y"]

        if courante:
            lignes.append(" ".join(courante))

        all_text.append("\n".join(lignes))

    return "\n\n".join(all_text)


def _detecter_nb_colonnes_ocr(centres_x: np.ndarray,
                                largeur: float) -> int:
    """Version OCR de la détection colonnes (travaille sur mots, pas blocs)."""
    if len(centres_x) < 3:
        return 1

    x_norm     = (centres_x / largeur).reshape(-1, 1)
    meilleur_k = 1
    meilleur_s = -1

    for k in range(2, 4):
        if len(x_norm) <= k:
            break
        try:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(x_norm)
            score  = silhouette_score(x_norm, labels)
            if score > 0.4 and score > meilleur_s:
                meilleur_s = score
                meilleur_k = k
        except Exception:
            pass

    return meilleur_k


# ════════════════════════════════════════════════
# 5. NETTOYAGE
# ════════════════════════════════════════════════

def _nettoyer(text: str) -> str:
    if not text:
        return ""

    # Correction OCR basique
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'(\S+@\S+)\s*\.\s*(com|fr|ma)', r'\1.\2', text)

    # Nettoyer espaces multiples
    text = re.sub(r' {2,}', ' ', text)

    # Nettoyer lignes vides multiples
    lignes = [l.strip() for l in text.splitlines()]
    propre = []
    vide_precedente = False
    for ligne in lignes:
        if not ligne:
            if not vide_precedente:
                propre.append("")
            vide_precedente = True
        else:
            if len(ligne) >= 2:
                propre.append(ligne)
            vide_precedente = False

    return "\n".join(propre).strip()


# ════════════════════════════════════════════════
# 6. API PUBLIQUE
# ════════════════════════════════════════════════

def extraire_cv(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = _extraire_pdf(path)
    elif ext == ".docx":
        text = _extraire_docx(path)
    else:
        raise ValueError(f"Format non supporté : {ext}")

    return {
        "type":    "cv",
        "content": _nettoyer(text)
    }


def extraire_offre(json_offre: str) -> dict:
    data = json.loads(json_offre)
    text = data.get("offre", "").strip()

    return {
        "type":    "offre",
        "content": _nettoyer(text)
    }