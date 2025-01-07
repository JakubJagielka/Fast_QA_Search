import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from data_storage import DataStorage, EmbeddingType
from _data_types import SearchResult, SearchMode
from os.path import isdir
from llm_qa import LLM_QA
from openai import AsyncOpenAI
import os 


class SearchEngine:
    """A hybrid search engine combining semantic and keyword-based search approaches.
    
    This engine provides multiple search modes:
    - Hybrid: Combines both semantic and keyword search results
    - Semantic: Uses embeddings for meaning-based search
    - Keyword: Uses BM25 algorithm for keyword matching
    
    Attributes:
        semantic_weight (float): Weight for semantic search scores (0-1)
        keyword_weight (float): Weight for keyword search scores (0-1)
        data_storage (DataStorage): Storage handler for documents and embeddings
    """

    def __init__(self, 
                 path: Optional[str] = None, 
                 model_type: EmbeddingType = EmbeddingType.MINILM_L6, 
                 openai_apikey = None,
                 temp: bool = False
                 ) -> None:
        """Initialize the search engine with specified parameters.
        
        Args:
            path: Path to file or directory containing documents
            model_type: Type of embedding model to use
            temp: Whether to use temporary storage
        """
        self.semantic_weight = 0.5
        self.keyword_weight = 0.5
        self.openai_client = AsyncOpenAI(api_key=openai_apikey) if openai_apikey else None
            
        self.data_storage = DataStorage(model_type, self.openai_client, temp)
        if path:
            self._initialize_storage(path)
            if self.data_storage.changed:
                self.data_storage.save_vector_store()

        self.llm_qa = LLM_QA(self.openai_client) 
        
               

    def _initialize_storage(self, path: str) -> None:
        """Initialize data storage based on provided path.
        
        Args:
            path: Path to file or directory
        """
        if isdir(path):
            self.data_storage.read_from_folder(path)
        else:
            self.data_storage.read_from_file(path)
        


    def search(self, query: str, k: int = 3, mode: SearchMode = SearchMode.HYBRID) -> List[SearchResult]:
        """Execute search query using specified mode.
        
        Args:
            query: Search query string
            k: Number of top results to return
            mode: Search mode (hybrid, semantic, or keyword)
            
        Returns:
            List of ranked SearchResult objects
        """
        search_methods = {
            SearchMode.HYBRID: self._hybrid_search,
            #SearchMode.SEMANTIC: self._semantic_search,
            #SearchMode.KEYWORD: self._keyword_search
        }
        
        results = search_methods[mode](query, k)
        self.display_results(results)
        return results
    
    
    def llm_qa_search(self, query: str, model: str = 'gpt-4o-mini', k=3) -> str:
        results = self.data_storage.vector_store.search(
            self.data_storage.create_embeddings(query), 'files', k*2
        )
        
        return asyncio.run(self.llm_qa.answer(query, [self.retrive_chunk(results[2][i], results[1][i]) for i in range(len(results[1]))], model=model))


    def _hybrid_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform hybrid search combining keyword and semantic results.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            Combined and ranked search results
        """
        keyword_results = self.data_storage.index.search(query, k)
        semantic_results = self.data_storage.vector_store.search(
            self.data_storage.create_embeddings(query), 'files', k*2
        )
        return self._combine_results(keyword_results, semantic_results, k)

    def _combine_results(self, 
                        keyword_results: List[Dict], 
                        semantic_results: tuple, 
                        k: int) -> List[SearchResult]:
        """Combine and normalize keyword and semantic search results.
        
        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            k: Number of results to return
            
        Returns:
            Combined and normalized search results
        """
        combined_results = []
        semantic_scores = dict(zip(semantic_results[2], semantic_results[0]))
        
        # Process each keyword result
        for result in keyword_results:
            doc_id = result['doc_id']
            semantic_score = semantic_scores.get(doc_id, 0.0)
            
            # Calculate normalized scores
            max_bm25 = max(r['bm25_score'] for r in keyword_results) or 1
            normalized_bm25 = result['bm25_score'] / max_bm25
            combined_score = (
                self.keyword_weight * normalized_bm25 +
                self.semantic_weight * semantic_score
            )
            
            # Find relevant chunk IDs
            chunk_ids = [
                semantic_results[1][idx] 
                for idx, sem_doc_id in enumerate(semantic_results[2]) 
                if sem_doc_id == doc_id
            ]
            
            combined_results.append(self._create_search_result(
                result, doc_id, semantic_score, combined_score, chunk_ids
            ))

        return sorted(combined_results, key=lambda x: x.combined_score, reverse=True)[:k]

    def _create_search_result(self, 
                            result: Dict, 
                            doc_id: str, 
                            semantic_score: float, 
                            combined_score: float, 
                            chunk_ids: List[str]) -> SearchResult:
        """Create a SearchResult object from search components.
        
        Args:
            result: Original search result
            doc_id: Document identifier
            semantic_score: Semantic search score
            combined_score: Combined search score
            chunk_ids: List of relevant chunk IDs
            
        Returns:
            Formatted SearchResult object
        """
        return SearchResult(
            doc_id=doc_id,
            file_name=result['file_name'].decode() if isinstance(result['file_name'], bytes) else result['file_name'],
            bm25_score=result['bm25_score'],
            semantic_score=semantic_score,
            combined_score=combined_score,
            positions=result['positions'],
            all_terms_present=result['all_terms_present'],
            chunk_id=chunk_ids,
            chunk_text=[self.retrive_chunk(doc_id, chunk) for chunk in chunk_ids] if chunk_ids else []
        )

    # Utility Methods
    def display_results(self, results: List[SearchResult], detailed: bool = True) -> None:
        """Display formatted search results.
        
        Args:
            results: List of search results to display
            detailed: Whether to show detailed information
        """
        for i, result in enumerate(results, 1):
            print(f"\n=== Result {i} ===")
            print(f"File: {result.file_name}")
            print(f"Combined Score: {result.combined_score:.4f}")
            print(f"BM25 Score: {result.bm25_score:.4f}")
            print(f"Semantic Score: {result.semantic_score:.4f}")
            print(f"All Terms Present: {result.all_terms_present}")
            print(f"{len(result.chunk_id)} Relevant Chunks Found")
            
            if detailed:
                self._display_detailed_results(result)

    def _display_detailed_results(self, result: SearchResult) -> None:
        """Display detailed information for a search result.
        
        Args:
            result: SearchResult object to display
        """
        print("\nToken Positions:")
        for token, info in result.positions.items():
            print(f"  {token}: {info['positions']}")
            
        if not result.chunk_id:
            print("\nNo relevant chunks found for this document.")
        
        for i, chunk_text in enumerate(result.chunk_text):
            print("\nRelevant Chunk:")
            print(chunk_text, "\n")


    def add_source(self, item: str) -> None:
        """
        Adds one of the following to the search engine:
        - A text string
        - A file
        - A directory
        - A URL
        
        Args:
            item: String to process (text, file path, directory path, or URL)
        """
        
        if item.startswith(('http://', 'https://')):
            success = self.data_storage.read_from_url(item)
            if success and self.data_storage.changed:
                self.data_storage.save_vector_store()
                
        elif os.path.isdir(item):
            success = self.data_storage.read_from_folder(item)
            if success and self.data_storage.changed:
                self.data_storage.save_vector_store()
                
        elif os.path.isfile(item):
            success = self.data_storage.read_from_file(item)
            if success and self.data_storage.changed:
                self.data_storage.save_vector_store()
                
        else:
            success = self.data_storage.read_from_text(item)
            if success and self.data_storage.changed:
                self.data_storage.save_vector_store()
    
    def set_search_weights(self, semantic_weight: float = 0.5, keyword_weight: float = 0.5) -> None:
        """Configure weights for hybrid search scoring.
        
        Args:
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Raises:
            AssertionError: If weights are invalid
        """
        assert 0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1
        assert abs(semantic_weight + keyword_weight - 1.0) < 1e-6
        
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

   
    def retrive_chunk(self, doc_id: str, chunk_id: str) -> str:
        """Retrieve specific chunk text from document.
        
        Args:
            doc_id: Document identifier
            chunk_id: Chunk identifier
            
        Returns:
            Chunk text content
        """
        return self.data_storage.index.return_chunk(doc_id, chunk_id)
    
    def retrive_close_chunks(self, doc_id: str, chunk_id: str, k: int = 3) -> str:
        """Retrieve nearby chunks from document.
        
        Args:
            doc_id: Document identifier
            chunk_id: Reference chunk identifier
            k: Number of nearby chunks to retrieve
            
        Returns:
            Combined nearby chunks text
        """
        return self.data_storage.index.return_close_chunks(doc_id, chunk_id, k)
    
    def retrive_document_text(self, doc_id: str) -> str:
        """Retrieve full document text.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Complete document text
        """
        return self.data_storage.index.return_doc(doc_id)


def simple_search(path: Union[str, List[str]], 
                 query: str, 
                 k: int = 3,
                 embedding_model: EmbeddingType = EmbeddingType.MINILM_L6,
                 mode: SearchMode = SearchMode.HYBRID) -> List[SearchResult]:
    """Perform one-time search operation on specified paths.
    
    Args:
        path: Path or list of paths to search
        query: Search query string
        k: Number of results to return
        embedding_model: Embedding model type
        mode: Search mode
        
    Returns:
        Ranked search results
    """
    search_engine = SearchEngine(path, model_type=embedding_model, temp=True)
    return search_engine.search(query, k=k, mode=mode)


def simple_qa(query: str, path: Union[str, List[str]], k: int = 3, model: str = 'gpt-4o-mini',
                embedding_model: EmbeddingType = EmbeddingType.MINILM_L6,
                 ) -> List[SearchResult]:
    """Perform one-time searcch with question answer by llm.
    
    Args:
        path: Path or list of paths to search
        query: Search query string
        k: Number of results to return
        embedding_model: Embedding model type
        mode: Search mode
        
    Returns:
        Ranked search results
    """
    search_engine = SearchEngine(path, model_type=embedding_model, temp=True)
    return search_engine.llm_qa_search(query,model=model, k=k)

# Example usage
if __name__ == "__main__":
    search_engine = SearchEngine('files', EmbeddingType.MINILM_L6)
    results = search_engine.search("allocation memory", k=3, mode=SearchMode.HYBRID)
    #search_engine.add_source("My name is Adam Smith.")
    llm_qa = search_engine.llm_qa_search("What is my name? Tell me all posibilites.")
    print(llm_qa)
    
