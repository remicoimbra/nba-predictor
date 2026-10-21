"""
Génère docs/preparation-entretien.pdf à partir de source.html, avec Chrome.

    python docs/entretien/build.py

Étapes : typographie française (espaces insécables) hors code et schémas,
numéros de page du sommaire, impression en PDF par Chrome sans en-tête.

Numéros du sommaire : si pypdfium2 est installé (pip install pypdfium2),
le PDF est imprimé une première fois, la page de chaque titre de chapitre y
est repérée, puis le PDF est réimprimé avec ces numéros. Sinon, on reprend
PAGES, à mettre à jour à la main si la pagination change.
"""

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "preparation-entretien.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Titre de chaque chapitre (texte exact du <h1>, sans le numéro).
HEADINGS = {
    "1": "Le projet en une page", "2": "Vue d'ensemble : le trajet d'une donnée",
    "3": "Les outils de base", "4": "Les données", "5": "Le Machine Learning, simplement",
    "6": "Les variables du projet", "7": "Les modèles", "8": "Évaluer honnêtement",
    "9": "Expliquer une prédiction", "10": "Le score et les statistiques des joueurs",
    "11": "L'API", "12": "Le site (front-end)", "13": "La mise en ligne",
    "14": "Les fichiers clés", "15": "Extraits de code commentés",
    "16": "Questions d'entretien", "17": "Glossaire",
}
# Pagination du 2026-10-21 (28 pages), utilisée sans pypdfium2.
PAGES = {"1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 9, "7": 10, "8": 12, "9": 13,
         "10": 15, "11": 16, "12": 17, "13": 19, "14": 20, "15": 21, "16": 24, "17": 27}

NBSP, NNBSP = "\u00a0", "\u202f"
TEXT_REPLACEMENTS = [
    (" %", NBSP + "%"), (" :", NBSP + ":"), (" ;", NNBSP + ";"),
    (" ?", NNBSP + "?"), (" !", NNBSP + "!"), ("« ", "«" + NBSP), (" »", NBSP + "»"),
]
# Blocs laissés tels quels : code, styles, schémas, balises.
PROTECTED = re.compile(r"(<pre\b.*?</pre>|<style\b.*?</style>|<svg\b.*?</svg>|<[^>]+>)", re.S)


def french(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    return re.sub(r"(\d) (\d{3})\b", r"\1" + NNBSP + r"\2", text)


def print_pdf(html: str, pages: dict) -> None:
    html = re.sub(r"\{\{p(\d+)\}\}", lambda m: str(pages.get(m.group(1), "–")), html)
    # À côté de source.html, pour que les chemins des captures restent valides.
    build = HERE / "build.html"
    build.write_text(html, encoding="utf-8")
    try:
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            "--virtual-time-budget=3000", f"--print-to-pdf={OUT_PDF}", build.as_uri(),
        ], check=True, capture_output=True)
    finally:
        build.unlink(missing_ok=True)


def chapter_pages() -> dict | None:
    """Page de début de chaque chapitre dans le PDF imprimé, ou None sans pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    pdf = pdfium.PdfDocument(OUT_PDF)
    lines = [
        {line.strip() for line in page.get_textpage().get_text_range()
         .replace(NBSP, " ").replace(NNBSP, " ").splitlines()}
        for page in pdf
    ]
    pdf.close()
    found = {}
    for key, title in HEADINGS.items():
        # Couverture et sommaire exclus (le sommaire répète les titres).
        found[key] = next((i for i, page in enumerate(lines[2:], start=3) if title in page), None)
    missing = [HEADINGS[k] for k, v in found.items() if v is None]
    if missing:
        raise SystemExit(f"Titres introuvables dans le PDF : {missing}")
    return found


def main():
    html = (HERE / "source.html").read_text(encoding="utf-8")
    parts = PROTECTED.split(html)
    html = "".join(p if i % 2 else french(p) for i, p in enumerate(parts))

    print_pdf(html, PAGES)
    pages = chapter_pages()
    if pages is None:
        print("pypdfium2 absent : sommaire repris de PAGES, sans vérification.")
    elif pages != PAGES:
        print_pdf(html, pages)
        print("Pagination changée, PAGES à mettre à jour :", pages)
    print(f"PDF : {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
