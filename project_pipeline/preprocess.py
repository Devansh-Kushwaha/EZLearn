# # nlp_pipeline/preprocess.py
# import re
# from typing import List

# def clean_text(text: str) -> str:
#     """
#     Cleans text by normalizing whitespace.
#     """
#     return re.sub(r'\s+', ' ', text).strip()


# def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
#     """
#     Splits long text into overlapping chunks to feed into an embedding model or LLM.

#     Prevents infinite loops by ensuring 'start' always increases.
#     """
#     if not text:
#         return []

#     chunks = []
#     text_len = len(text)
#     start = 0

#     while start < text_len:
#         end = min(start + max_chars, text_len)
#         chunks.append(text[start:end])

#         # Advance start safely; ensure we always move forward
#         if end == text_len:
#             break
#         start = max(0, end - overlap)

#         # If overlap is too large (e.g., > max_chars), force move forward
#         if start >= end:
#             start = end

#     return chunks


# nlp_pipeline/preprocess.py
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Cleans text by normalizing whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200) -> List[str]:
    """
    Fixed-size chunking that NEVER produces tiny chunks.
    Designed for PDFs, PPTs, academic notes, and technical content.

    - Avoids sentence splitting (PDFs break sentences).
    - Every chunk is ~max_chars large.
    - Overlap is preserved for summarization consistency.
    """
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()

        # Skip accidental tiny chunks (< 300 chars)
        if len(chunk) < 300 and chunks:
            chunks[-1] += " " + chunk
        else:
            chunks.append(chunk)

        # Prepare next chunk
        if end == length:
            break
        start = end - overlap

    return chunks
