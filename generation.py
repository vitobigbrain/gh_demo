"""
Answer Generation Module

Generates answers based on retrieved context using OpenAI API with:
- Numbered citations in context
- Retry with exponential backoff
- Structured response with answer and sources
- Clean error handling
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from faiss_demo import load_meta, load_index, search, INDEX_FILE_PATH, META_FILE_PATH, DEFAULT_TOP_K


# ========= Configuration =========
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 16.0  # seconds


class GenerationError(Exception):
    """Custom exception for generation errors."""
    pass


@dataclass
class GenerationResult:
    """Structured result from answer generation."""
    answer_text: str
    used_sources: List[Dict[str, Any]]
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "used_sources": self.used_sources,
            "model": self.model,
        }


def _get_openai_client() -> OpenAI:
    """Get OpenAI client, checking for API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise GenerationError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY=your-key"
        )
    return OpenAI(api_key=api_key)


def build_cited_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Build context string with numbered citations.
    
    Each chunk is prefixed with [1], [2], etc. for citation reference.
    Includes source file and page info when available.
    """
    if not search_results:
        return ""
    
    parts = []
    for i, r in enumerate(search_results, 1):
        source_info = f"来源: {r.get('source_file', 'unknown')}"
        if r.get('page') is not None:
            source_info += f", 第{r['page']+1}页"
        
        parts.append(f"[{i}] ({source_info})\n{r['text']}")
    
    return "\n\n".join(parts)


def generate_answer(
    context_text: str,
    user_question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Generate an answer based on context (legacy API for backward compatibility).
    
    Args:
        context_text: Context information (retrieved document chunks)
        user_question: User's question
        model: Model to use, default "gpt-4o"
        temperature: Temperature parameter, default 0.7
        max_retries: Maximum retry attempts
    
    Returns:
        Generated answer text
    
    Raises:
        GenerationError: If generation fails after all retries
    """
    client = _get_openai_client()
    
    system_prompt = "你是一个专业的问答助手。请基于提供的上下文信息准确回答用户的问题。"
    
    user_prompt = f"""请基于以下上下文信息回答用户的问题。

上下文信息：
{context_text}

用户问题：{user_question}

要求：
1. 只基于提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请明确说明"根据提供的资料，无法找到相关信息"
3. 答案要准确、简洁、有条理
4. 在回答中引用来源，使用 [1], [2] 等标注引用了哪些文档片段
5. 用中文回答

请回答："""

    last_error: Exception | None = None
    backoff = INITIAL_BACKOFF
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )
            
            answer = response.choices[0].message.content
            return answer or ""
            
        except RateLimitError as e:
            last_error = GenerationError(f"OpenAI rate limit exceeded: {e}")
        except APIConnectionError as e:
            last_error = GenerationError(f"Cannot connect to OpenAI API: {e}")
        except APIError as e:
            last_error = GenerationError(f"OpenAI API error: {e}")
        except Exception as e:
            last_error = GenerationError(f"Generation failed: {type(e).__name__}: {e}")
        
        # Retry with exponential backoff
        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
    
    raise last_error or GenerationError("Generation failed after all retries")


def generate_answer_with_sources(
    search_results: List[Dict[str, Any]],
    user_question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> GenerationResult:
    """
    Generate an answer with citation support and return structured result.
    
    Args:
        search_results: List of search results from FAISS search
        user_question: User's question
        model: Model to use
        temperature: Temperature parameter
        max_retries: Maximum retry attempts
    
    Returns:
        GenerationResult with answer_text, used_sources, and model info
    
    Raises:
        GenerationError: If generation fails
    """
    if not search_results:
        return GenerationResult(
            answer_text="没有找到相关的文档内容，无法回答该问题。请先上传相关文档。",
            used_sources=[],
            model=model,
        )
    
    # Build cited context
    context_text = build_cited_context(search_results)
    
    # Generate answer
    answer_text = generate_answer(
        context_text=context_text,
        user_question=user_question,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )
    
    return GenerationResult(
        answer_text=answer_text,
        used_sources=search_results,
        model=model,
    )


def rag_query(
    index: faiss.Index,
    meta: Dict[int, Any],
    question: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> GenerationResult:
    """
    Full RAG pipeline: search + generate.
    
    Args:
        index: FAISS index
        meta: Metadata store
        question: User's question
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity score filter
        model: LLM model to use
        temperature: Generation temperature
    
    Returns:
        GenerationResult with answer and sources
    """
    # Search for relevant chunks
    search_results = search(
        index=index,
        meta=meta,
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    
    # Generate answer with sources
    return generate_answer_with_sources(
        search_results=search_results,
        user_question=question,
        model=model,
        temperature=temperature,
    )


if __name__ == "__main__":
    # Demo: Full RAG query
    question = "解释一下SEMANTIC ROLE LABELING"
    
    # Check for index files
    if not os.path.exists(INDEX_FILE_PATH) or not os.path.exists(META_FILE_PATH):
        print(f"错误：找不到 index 文件。请先运行 faiss_demo.py 创建索引。")
        print(f"Index 路径: {INDEX_FILE_PATH}")
        print(f"Meta 路径: {META_FILE_PATH}")
    else:
        print("加载 faiss index...")
        index = load_index(INDEX_FILE_PATH)
        meta = load_meta(META_FILE_PATH)
        
        print(f"检索问题: {question}\n")
        
        try:
            # Use the new RAG query function
            result = rag_query(
                index=index,
                meta=meta,
                question=question,
                top_k=5,
            )
            
            print("=" * 60)
            print(f"检索到 {len(result.used_sources)} 个相关片段:\n")
            
            for i, src in enumerate(result.used_sources, 1):
                print(f"[{i}] 来源: {src['source_file']}, 相似度: {src['score']:.4f}")
                if src.get('page') is not None:
                    print(f"    页码: {src['page']+1}")
                print(f"    内容: {src['text'][:200]}...")
                print()
            
            print("=" * 60)
            print(f"\n问题: {question}")
            print(f"\n答案 (模型: {result.model}):\n{result.answer_text}")
            
        except GenerationError as e:
            print(f"生成错误: {e}")
