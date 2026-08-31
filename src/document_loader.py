"""
Phase 1: Technical Document Processing & Ingestion Module
==========================================================
Responsible for loading PDF and text documentation files, stripping formatting noise,
preserving code blocks/syntax, and extracting detailed page-level metadata.
"""

import os
import re
from typing import List, Dict, Any
import pypdf

class DocumentLoader:
    """Document Ingestion Engine for Enterprise Technical Manuals & Specs."""

    def __init__(self, document_dir: str = "data/documents"):
        self.document_dir = document_dir

    def clean_text(self, text: str) -> str:
        """
        Cleans raw extracted text while strictly preserving technical syntax,
        such as code blocks, curly braces, HTTP headers, and JSON schemas.
        """
        if not text:
            return ""

        # 1. Normalize line endings (\r\n -> \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Convert tabs and non-standard spacing to single spaces
        text = re.sub(r'[\t\f\v]', ' ', text)

        # 3. Collapse excessive blank lines (keep max 2 newlines for paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 4. Strip whitespace from individual lines
        lines = [line.strip() for line in text.split('\n')]
        
        return "\n".join(lines).strip()

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF file page by page and extracts clean text alongside page metadata.
        """
        file_name = os.path.basename(file_path)
        documents = []

        try:
            reader = pypdf.PdfReader(file_path)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned_text = self.clean_text(raw_text)

                if cleaned_text:
                    doc_item = {
                        "content": cleaned_text,
                        "metadata": {
                            "source": file_name,
                            "file_path": file_path,
                            "page": page_num,
                            "total_pages": total_pages,
                            "char_count": len(cleaned_text),
                            "file_type": "pdf"
                        }
                    }
                    documents.append(doc_item)

        except Exception as e:
            print(f"Error loading PDF '{file_path}': {str(e)}")

        return documents

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Scans the document directory and loads all supported technical documents (.pdf, .txt, .md).
        """
        all_docs = []

        if not os.path.exists(self.document_dir):
            print(f"Directory '{self.document_dir}' does not exist.")
            return all_docs

        for root, _, files in os.walk(self.document_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                if file.endswith('.pdf'):
                    pdf_docs = self.load_pdf(file_path)
                    all_docs.extend(pdf_docs)
                elif file.endswith(('.txt', '.md')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        cleaned_text = self.clean_text(raw_text)
                        if cleaned_text:
                            all_docs.append({
                                "content": cleaned_text,
                                "metadata": {
                                    "source": file,
                                    "file_path": file_path,
                                    "page": 1,
                                    "total_pages": 1,
                                    "char_count": len(cleaned_text),
                                    "file_type": file.split('.')[-1]
                                }
                            })
                    except Exception as e:
                        print(f"Error loading text file '{file_path}': {str(e)}")

        print(f"Phase 1 Complete: Successfully ingested {len(all_docs)} pages/sections from '{self.document_dir}'.")
        return all_docs


if __name__ == "__main__":
    # Test execution
    loader = DocumentLoader("data/documents")
    docs = loader.load_all_documents()
    print("\n--- EXTRACTED METADATA SAMPLE ---")
    if docs:
        print("Metadata:", docs[0]["metadata"])
        print("\nCleaned Text Sample:\n", docs[0]["content"][:350])