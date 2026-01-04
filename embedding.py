"""
Embedding Service Module

Provides text embedding via a local HTTP embedding service with:
- Configurable timeout
- Retry with exponential backoff
- Response validation
- Clean error messages for UI
"""

from __future__ import annotations

import time
from typing import List

import requests

# ========= Configuration =========
EMBEDDING_URL = "http://test.2brain.cn:9800/v1/emb"

# Request settings
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 16.0  # seconds


class EmbeddingServiceError(Exception):
    """Custom exception for embedding service errors."""
    pass


def local_embedding(
    inputs: List[str],
    *,
    timeout: float = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> List[List[float]]:
    """
    Get embeddings from the local embedding service.
    
    Args:
        inputs: List of text strings to embed
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of embedding vectors (each is a list of floats)
    
    Raises:
        EmbeddingServiceError: If the service is unreachable or returns invalid data
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not inputs:
        return []  
    
    if not all(isinstance(t, str) for t in inputs):
        raise ValueError("All inputs must be strings")
    
    headers = {"Content-Type": "application/json"}
    data = {"texts": inputs}
    
    last_error: Exception | None = None
    backoff = INITIAL_BACKOFF
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                EMBEDDING_URL,
                headers=headers,
                json=data,
                timeout=timeout,
            )
            
            # Check HTTP status
            if response.status_code != 200:
                raise EmbeddingServiceError(
                    f"Embedding service returned status {response.status_code}: {response.text[:200]}"
                )
            
            # Parse response
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError as e:
                raise EmbeddingServiceError(
                    f"Embedding service returned invalid JSON: {response.text[:200]}"
                ) from e
            
            # Validate response structure
            if not isinstance(result, dict):
                raise EmbeddingServiceError(
                    f"Expected dict response, got {type(result).__name__}"
                )
            
            if "data" not in result:
                raise EmbeddingServiceError(
                    f"Response missing 'data' field. Keys: {list(result.keys())}"
                )
            
            data_field = result["data"]
            if not isinstance(data_field, dict) or "text_vectors" not in data_field:
                raise EmbeddingServiceError(
                    f"Response 'data' missing 'text_vectors' field"
                )
            
            vectors = data_field["text_vectors"]
            
            # Validate vectors
            if not isinstance(vectors, list):
                raise EmbeddingServiceError(
                    f"Expected list of vectors, got {type(vectors).__name__}"
                )
            
            if len(vectors) != len(inputs):
                raise EmbeddingServiceError(
                    f"Expected {len(inputs)} vectors, got {len(vectors)}"
                )
            
            # Validate each vector is a list of numbers
            for i, vec in enumerate(vectors):
                if not isinstance(vec, list):
                    raise EmbeddingServiceError(
                        f"Vector {i} is not a list: {type(vec).__name__}"
                    )
                if not vec:
                    raise EmbeddingServiceError(f"Vector {i} is empty")
            
            return vectors
            
        except requests.exceptions.Timeout as e:
            last_error = EmbeddingServiceError(
                f"Embedding service timed out after {timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            last_error = EmbeddingServiceError(
                f"Cannot connect to embedding service at {EMBEDDING_URL}. "
                f"Please check if the service is running."
            )
        except EmbeddingServiceError:
            raise
        except Exception as e:
            last_error = EmbeddingServiceError(
                f"Embedding request failed: {type(e).__name__}: {e}"
            )
        
        # Retry with exponential backoff
        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
    
    # All retries exhausted
    raise last_error or EmbeddingServiceError("Embedding request failed after all retries")


if __name__ == '__main__':
    # Test the embedding service
    inputs = ["Hello, world!"]
    try:
        output = local_embedding(inputs)[0]
        print(f"Embedding successful!")
        print(f"Dimension: {len(output)}")
        print(f"First 5 values: {output[:5]}")
    except EmbeddingServiceError as e:
        print(f"Error: {e}")
