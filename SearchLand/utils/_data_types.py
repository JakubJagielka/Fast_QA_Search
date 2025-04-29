from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: int
    file_name: str
    bm25_score: float
    semantic_score: Optional[float]
    combined_score: float
    positions: Dict
    all_terms_present: bool
    chunk_id: Optional[int] = None
    chunk_text: Optional[List[str]] = None

class SearchMode(Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    
class EmbeddingType(Enum):
    OPENAI_SMALL = 1536 
    OPENAI_LARGE = 3072 
    MINILM_L6 = 384
    
