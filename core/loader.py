from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os


class DocumentLoader:

    @staticmethod
    def extract_text_from_pdf(path):

        text = ""
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print("PDF text extraction failed:", e)

        # If no text found -> OCR fallback
        if not text.strip():
            try:
                images = convert_from_path(path)
                for img in images:
                    text += pytesseract.image_to_string(img)
            except Exception as e:
                print("OCR failed:", e)
        return text.strip()


    @staticmethod
    def load_file(path):

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return DocumentLoader.extract_text_from_pdf(path)
        else:
            # normal text file
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
