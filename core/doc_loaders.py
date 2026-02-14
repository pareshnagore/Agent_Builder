"""
Document loaders for Agent_ng.
Supports multiple file formats: PDF, TXT, DOCX, CSV, MD, XLSX.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from abc import ABC, abstractmethod


class LoaderException(Exception):
    """Base exception for loader errors."""
    pass


class DocumentLoader(ABC):
    """Abstract base for document loaders."""

    @abstractmethod
    def load(self, path: str) -> str:
        """Load and return document text."""
        pass

    @abstractmethod
    def supports_paging(self) -> bool:
        """Whether this format supports page numbers."""
        pass


class PDFLoader(DocumentLoader):
    """PDF document loader with OCR fallback."""

    def load(self, path: str) -> str:
        """
        Load PDF and extract text.
        Falls back to OCR if text extraction fails.
        
        Args:
            path: Path to PDF file
        
        Returns:
            Extracted text
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise LoaderException("pypdf not installed. Install with: pip install pypdf")

        text = ""
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"PDF text extraction failed: {e}")

        # OCR fallback for scanned PDFs
        if not text.strip():
            try:
                from pdf2image import convert_from_path
                import pytesseract
                
                images = convert_from_path(path)
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            except Exception as e:
                print(f"OCR fallback failed: {e}")
                raise LoaderException(f"Failed to extract text from PDF {path}: {e}")

        return text.strip()

    def supports_paging(self) -> bool:
        return True


class TextLoader(DocumentLoader):
    """Plain text file loader."""

    def load(self, path: str) -> str:
        """
        Load text file.
        
        Args:
            path: Path to text file
        
        Returns:
            File contents
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise LoaderException(f"Failed to load text file {path}: {e}")

    def supports_paging(self) -> bool:
        return False


class MarkdownLoader(DocumentLoader):
    """Markdown file loader."""

    def load(self, path: str) -> str:
        """
        Load markdown file.
        
        Args:
            path: Path to markdown file
        
        Returns:
            File contents
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise LoaderException(f"Failed to load markdown file {path}: {e}")

    def supports_paging(self) -> bool:
        return False


class DocxLoader(DocumentLoader):
    """Microsoft Word (.docx) loader."""

    def load(self, path: str) -> str:
        """
        Load .docx file.
        
        Args:
            path: Path to docx file
        
        Returns:
            Extracted text from document
        """
        try:
            from docx import Document
        except ImportError:
            raise LoaderException("python-docx not installed. Install with: pip install python-docx")

        try:
            doc = Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])
            
            # Also extract from tables if present
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += "\n" + cell.text
            
            return text
        except Exception as e:
            raise LoaderException(f"Failed to load docx file {path}: {e}")

    def supports_paging(self) -> bool:
        return False


class CSVLoader(DocumentLoader):
    """CSV file loader."""

    def load(self, path: str) -> str:
        """
        Load CSV file and convert to readable text.
        
        Args:
            path: Path to CSV file
        
        Returns:
            CSV contents as formatted text
        """
        try:
            import csv
        except ImportError:
            raise LoaderException("csv module not available")

        try:
            lines = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    lines.append(" | ".join(reader.fieldnames))
                    for row in reader:
                        lines.append(" | ".join(str(v) for v in row.values()))
            return "\n".join(lines)
        except Exception as e:
            raise LoaderException(f"Failed to load CSV file {path}: {e}")

    def supports_paging(self) -> bool:
        return False


class ExcelLoader(DocumentLoader):
    """Excel file loader."""

    def load(self, path: str) -> str:
        """
        Load Excel file and convert to readable text.
        
        Args:
            path: Path to Excel file
        
        Returns:
            Excel contents as formatted text
        """
        try:
            import openpyxl
        except ImportError:
            raise LoaderException("openpyxl not installed. Install with: pip install openpyxl")

        try:
            workbook = openpyxl.load_workbook(path)
            text = ""
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\n--- Sheet: {sheet_name} ---\n"
                
                for row in sheet.iter_rows(values_only=True):
                    text += " | ".join(str(v) if v is not None else "" for v in row) + "\n"
            
            return text
        except Exception as e:
            raise LoaderException(f"Failed to load Excel file {path}: {e}")

    def supports_paging(self) -> bool:
        return False


class DocumentLoaderFactory:
    """Factory for creating appropriate loader based on file type."""

    LOADERS = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".docx": DocxLoader,
        ".doc": DocxLoader,
        ".csv": CSVLoader,
        ".xlsx": ExcelLoader,
        ".xls": ExcelLoader,
    }

    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """
        Get appropriate loader for file type.
        
        Args:
            file_path: Path to document
        
        Returns:
            Loader instance
        
        Raises:
            LoaderException: If file type not supported
        """
        ext = Path(file_path).suffix.lower()
        
        if ext not in DocumentLoaderFactory.LOADERS:
            raise LoaderException(
                f"Unsupported file type: {ext}. Supported: {list(DocumentLoaderFactory.LOADERS.keys())}"
            )
        
        loader_class = DocumentLoaderFactory.LOADERS[ext]
        return loader_class()

    @staticmethod
    def load(file_path: str) -> tuple[str, str, bool]:
        """
        Load document using appropriate loader.
        
        Args:
            file_path: Path to document
        
        Returns:
            Tuple of (text, file_type, supports_paging)
        
        Raises:
            LoaderException: If loading fails
        """
        if not os.path.exists(file_path):
            raise LoaderException(f"File not found: {file_path}")

        loader = DocumentLoaderFactory.get_loader(file_path)
        text = loader.load(file_path)
        file_type = Path(file_path).suffix.lower().lstrip(".")
        
        return text, file_type, loader.supports_paging()

    @staticmethod
    def supported_types() -> list[str]:
        """Return list of supported file extensions."""
        return list(DocumentLoaderFactory.LOADERS.keys())