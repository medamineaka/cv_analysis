# extraction.py

import os
import json
import re
import docx
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text


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
# 2. PDF SMART EXTRACTION
# ════════════════════════════════════════════════

def _extraire_pdf(path: str) -> str:

    # ── 1. PDF TEXTE NATIF
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 200:
            return text
    except:
        pass

    # ── 2. OCR
    text = _ocr_pdf(path)

    if not text.strip():
        raise ValueError("Extraction PDF échouée")

    return text


def _ocr_pdf(path: str) -> str:

    pages = convert_from_path(
        path,
        dpi=300,
        poppler_path=r"C:\poppler\Library\bin"
    )

    all_text = []

    for img in pages:

        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            lang="fra+eng",
            config="--psm 4"
        )

        blocks = []

        for i in range(len(data["text"])):
            word = data["text"][i].strip()

            try:
                conf = int(data["conf"][i])
            except:
                conf = 0

            if conf < 40 or not word:
                continue

            blocks.append({
                "text": word,
                "x": data["left"][i],
                "y": data["top"][i]
            })

        # 🔥 tri spatial (clé du succès)
        blocks = sorted(blocks, key=lambda b: (b["y"], b["x"]))

        # 🔥 reconstruction lignes
        lines = []
        current = []
        y_prev = None

        for b in blocks:

            if y_prev is None:
                current.append(b["text"])
                y_prev = b["y"]
                continue

            if abs(b["y"] - y_prev) < 12:
                current.append(b["text"])
            else:
                lines.append(" ".join(current))
                current = [b["text"]]
                y_prev = b["y"]

        if current:
            lines.append(" ".join(current))

        all_text.append("\n".join(lines))

    return "\n\n".join(all_text)


# ════════════════════════════════════════════════
# 3. NETTOYAGE INTELLIGENT
# ════════════════════════════════════════════════

def _normalize(text: str) -> str:
    text = text.lower()

    text = (
        text.replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
            .replace("ù", "u")
            .replace("ô", "o")
    )

    return text


def _fix_common_ocr_errors(text: str) -> str:
    """
    Corrige erreurs fréquentes sans casser les mots
    """

    # espace entre lettres + chiffres
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)

    # email cassé
    text = re.sub(r'(\S+@\S+)\s*\.\s*(com|fr)', r'\1.\2', text)

    return text


def _add_structure(text: str) -> str:
    """
    Ajoute structure pour aider classification
    """

    keywords = [
        "experience", "formation", "competences",
        "langues", "projets", "contact",
        "education", "skills"
    ]

    for k in keywords:
        text = re.sub(rf"\b{k}\b", f"\n{k}", text)

    text = re.sub(r":\s*", ":\n", text)
    text = re.sub(r"\.\s+", ".\n", text)

    return text


def _clean_lines(text: str) -> str:
    lines = text.split("\n")

    clean = []
    for l in lines:
        l = l.strip()

        if len(l) < 2:
            continue

        clean.append(l)

    return "\n".join(clean)


def _nettoyer(text: str) -> str:

    if not text:
        return ""

    text = _normalize(text)
    text = _fix_common_ocr_errors(text)
    text = _add_structure(text)

    # ⚠️ IMPORTANT : ne pas détruire structure
    text = re.sub(r"[^\w\s\-\.:/@]", " ", text)

    text = re.sub(r"\s+", " ", text)

    # remettre lignes
    text = text.replace(" :", ":")
    text = text.replace(". ", ".\n")

    text = _clean_lines(text)

    return text


# ════════════════════════════════════════════════
# 4. API
# ════════════════════════════════════════════════

def extraire_cv(path: str) -> dict:

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = _extraire_pdf(path)
    elif ext == ".docx":
        text = _extraire_docx(path)
    else:
        raise ValueError(f"Format non supporté: {ext}")

    cleaned = _nettoyer(text)

    return {
        "type": "cv",
        "content": cleaned
    }


def extraire_offre(json_offre: str) -> dict:

    data = json.loads(json_offre)
    text = data.get("offre", "").strip()

    return {
        "type": "offre",
        "content": _nettoyer(text)
    }