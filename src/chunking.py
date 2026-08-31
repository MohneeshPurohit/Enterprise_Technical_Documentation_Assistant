"""
Phase 2: Code-Aware Text Chunking Module
========================================
Splits extracted technical documentation pages into smaller, semantically coherent chunks.
Uses recursive character splitting to preserve code blocks, JSON schemas, and API parameters,
while maintaining full metadata traceability.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any

class CodeAwareTextSplitter:
    """
    Splits text recursively using a priority list of technical separators.
    Priority order: Double newline ('\n\n'), Single newline ('\n'), Space (' '), Character ('').
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Recursively splits text into chunks of maximum length `chunk_size`
        with `chunk_overlap` overlap between consecutive chunks.
        """
        if not text:
            return []

        # Find the highest-priority separator present in the text
        chosen_sep = ""
        for sep in self.separators:
            if sep in text:
                chosen_sep = sep
                break

        raw_splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in raw_splits:
            split_len = len(split) + (len(chosen_sep) if current_chunk else 0)

            if current_length + split_len <= self.chunk_size:
                current_chunk.append(split)
                current_length += split_len
            else:
                if current_chunk:
                    joined_chunk = chosen_sep.join(current_chunk).strip()
                    if joined_chunk:
                        chunks.append(joined_chunk)

                # Maintain chunk overlap window
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_chunk = []
                    overlap_len = 0
                    for prev_item in reversed(current_chunk):
                        if overlap_len + len(prev_item) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_item)
                            overlap_len += len(prev_item) + len(chosen_sep)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0

                # Handle edge cases where a single split exceeds chunk_size
                if len(split) > self.chunk_size:
                    sub_splitter = CodeAwareTextSplitter(self.chunk_size, self.chunk_overlap)
                    sub_chunks = sub_splitter.split_text(split)
                    chunks.extend(sub_chunks[:-1])
                    if sub_chunks:
                        current_chunk = [sub_chunks[-1]]
                        current_length = len(sub_chunks[-1])
                else:
                    current_chunk.append(split)
                    current_length += len(split)

        if current_chunk:
            joined_chunk = chosen_sep.join(current_chunk).strip()
            if joined_chunk:
                chunks.append(joined_chunk)

        return chunks

class TechnicalChunker:
    """Processes document objects from Phase 1 into rich, metadata-backed chunks."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.splitter = CodeAwareTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of document items from Phase 1 and converts them into chunk objects.
        Propagates original metadata and appends chunk-specific tracking IDs.
        """
        all_chunks = []
        global_chunk_id = 0

        for doc in documents:
            content = doc.get("content", "")
            parent_meta = doc.get("metadata", {})

            text_chunks = self.splitter.split_text(content)

            for idx, text in enumerate(text_chunks):
                global_chunk_id += 1
                chunk_meta = parent_meta.copy()
                chunk_meta.update({
                    "chunk_id": f"chk_{global_chunk_id:04d}",
                    "chunk_index": idx + 1,
                    "total_chunks_in_page": len(text_chunks),
                    "chunk_char_count": len(text)
                })

                all_chunks.append({
                    "chunk_id": f"chk_{global_chunk_id:04d}",
                    "content": text,
                    "metadata": chunk_meta
                })

        print(f"Phase 2 Complete: Created {len(all_chunks)} chunks from {len(documents)} document pages.")
        return all_chunks


if __name__ == "__main__":
    from src.document_loader import DocumentLoader

    loader = DocumentLoader("data/documents")
    docs = loader.load_all_documents()

    chunker = TechnicalChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)

    print("\n--- SAMPLE GENERATED CHUNKS ---")
    for chk in chunks[:2]:
        print(f"\n[ID: {chk['chunk_id']} | Source: {chk['metadata']['source']} | Page: {chk['metadata']['page']}]")
        print(f"Length: {chk['metadata']['chunk_char_count']} chars")
        print("Content:\n" + "-"*40)
        print(chk["content"])
        print("-" * 40)