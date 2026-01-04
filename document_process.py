
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """Structured chunk with source metadata for citations."""
    text: str
    source_file: str
    chunk_index: int
    page: Optional[int] = None  # Only for PDF


def num_tokens_from_string(string: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _get_text_splitter(
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    length_function: Callable[[str], int] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a consistent text splitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function or num_tokens_from_string,
    )


def _load_pdf(file_path: str) -> List[tuple[str, Optional[int]]]:
    """Load PDF and return list of (text, page_number) tuples."""
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    results = []
    for doc in pages:
        page_num = doc.metadata.get("page", None)
        if doc.page_content:
            results.append((doc.page_content, page_num))
    return results


def _load_txt(file_path: str) -> List[tuple[str, Optional[int]]]:
    """Load TXT file and return list of (text, None) tuples."""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    text = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise ValueError(f"Could not decode file {file_path} with any known encoding")
    return [(text, None)] if text.strip() else []


def _load_docx(file_path: str) -> List[tuple[str, Optional[int]]]:
    """Load DOCX file and return list of (text, None) tuples."""
    try:
        import docx2txt
    except ImportError:
        raise ImportError("docx2txt is required for DOCX support. Install with: pip install docx2txt")
    
    text = docx2txt.process(file_path)
    return [(text, None)] if text and text.strip() else []


def load_and_chunk(
    file_path: str,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    length_function: Callable[[str], int] | None = None,
) -> List[DocumentChunk]:
    """
    Load a document (PDF, TXT, or DOCX) and split into structured chunks.
    
    Args:
        file_path: Path to the document file
        chunk_size: Maximum chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        length_function: Function to measure text length (defaults to token count)
    
    Returns:
        List[DocumentChunk]: Structured chunks with source metadata
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    source_file = os.path.basename(file_path)
    
    # Load based on file type
    if ext == '.pdf':
        raw_texts = _load_pdf(file_path)
    elif ext == '.txt':
        raw_texts = _load_txt(file_path)
    elif ext == '.docx':
        raw_texts = _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .txt, .docx")
    
    if not raw_texts:
        return []
    
    # Create splitter
    splitter = _get_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    
    # Split and create structured chunks
    chunks: List[DocumentChunk] = []
    chunk_index = 0
    
    for text, page_num in raw_texts:
        if not text.strip():
            continue
        split_texts = splitter.split_text(text)
        for split_text in split_texts:
            if split_text.strip():
                chunks.append(DocumentChunk(
                    text=split_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    page=page_num,
                ))
                chunk_index += 1
    
    return chunks


def chunk_pdf_texts(
    file_path: str,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    length_function: Callable[[str], int] | None = None,
) -> List[str]:
    """
    Load a PDF and split into text chunks using LangChain.
    
    Returns: List[str] (each item is a chunk of text)
    
    Note: This is kept for backward compatibility. 
          For new code, prefer load_and_chunk() which returns structured chunks.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    textsplit = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function or num_tokens_from_string,
    )
    chunks = textsplit.split_documents(pages)
    return [c.page_content for c in chunks if c.page_content]


if __name__ == '__main__':
    # Demo: load and chunk a PDF
    texts = chunk_pdf_texts('Speech and Language Processing.pdf')
    print(f"Chunks (legacy): {len(texts)}\n")
    
    # Demo: load_and_chunk with structured output
    chunks = load_and_chunk('Speech and Language Processing.pdf')
    print(f"Chunks (structured): {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:")
        print(f"  source: {chunk.source_file}")
        print(f"  page: {chunk.page}")
        print(f"  text: {chunk.text[:100]}...")
        print('=' * 40)
