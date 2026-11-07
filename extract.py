import os
import fitz  # PyMuPDF
from llama_index.core import Document

PDF_PATH = "data/maintenance.pdf"
IMG_DIR = "data/images"
os.makedirs(IMG_DIR, exist_ok=True)

def extract_all():
    doc = fitz.open(PDF_PATH)
    text_docs, table_docs, images = [], [], []

    for num, page in enumerate(doc, start=1):
        # Texte
        text = page.get_text()
        if text.strip():
            text_docs.append(Document(
                text=text,
                metadata={"page": num, "type": "texte"}
            ))

        # Tableaux
        for i, table in enumerate(page.find_tables()):
            md = table.to_markdown()
            if md.strip():
                table_docs.append(Document(
                    text=md,
                    metadata={"page": num, "type": "tableau"}
                ))

        # Images
        for j, img in enumerate(page.get_images(full=True)):
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.width < 100 or pix.height < 100:
                    continue  # on ignore les petites icônes
                if pix.colorspace is None:
                    continue  # masque sans couleur, on ignore
                if pix.colorspace.n not in (1, 3):
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                path = f"{IMG_DIR}/page{num}_img{j}.png"
                pix.save(path)
                images.append({"path": path, "page": num})
            except Exception as e:
                print(f"Image ignorée page {num} : {e}")

    return text_docs, table_docs, images

def render_pages(dpi=130):
    # Chaque page entière en image (pour les schémas vectoriels)
    doc = fitz.open(PDF_PATH)
    pages = []
    for num, page in enumerate(doc, start=1):
        path = f"{IMG_DIR}/page{num}_full.png"
        page.get_pixmap(dpi=dpi).save(path)
        pages.append({"path": path, "page": num})
    return pages

if __name__ == "__main__":
    t, tb, im = extract_all()
    pages = render_pages()
    print(f"{len(t)} pages de texte")
    print(f"{len(tb)} tableaux")
    print(f"{len(im)} images")
    print(f"{len(pages)} pages rendues")