import asyncio
from dataclasses import dataclass
import shutil
from typing import List, Dict, Optional, Any, Union
from os.path import isdir
from openai import AsyncOpenAI
import os
from .data_storage import DataStorage
from .utils._data_types import SearchResult, SearchMode, EmbeddingType
from .utils._logger import get_logger
from .llm_qa import LLM_QA
from .faiss_search import VectorStore
from .cython_files.Data_Struct import InvertedIndex
from dotenv import load_dotenv
load_dotenv()
logger = get_logger()

class SearchEngine:
    def __init__(self, path: Optional[str] = None, model_type: EmbeddingType = EmbeddingType.MINILM_L6,
                 openai_apikey = None,
                 temp: bool = False
                 ) -> None:
        self.path = path
        self.semantic_weight = 0.5
        self.keyword_weight = 0.5
        self.openai_client = AsyncOpenAI(api_key=openai_apikey) if openai_apikey else None

        self.data_storage = DataStorage(model_type, self.openai_client, temp)
        if path:
            self._initialize_storage(path)
            if self.data_storage.changed:
                self.data_storage.save_vector_store()
                self.data_storage.document_tracker.save_metadata() # Ensure metadata saved on init changes

        self.llm_qa = LLM_QA(self.openai_client)

    def search(self, query: str, k: int = 3, mode: SearchMode = SearchMode.HYBRID,
               vector_store_path: str = 'data'
               ) -> List[SearchResult]:

        search_methods = {
            SearchMode.HYBRID:    self._hybrid_search,
            SearchMode.SEMANTIC:  self._semantic_search,
            SearchMode.KEYWORD:   self._keyword_search,
        }
        search_func = search_methods.get(mode, self._hybrid_search)
        results = search_func(query, k, vector_store_path)
        self.display_results(results)
        return results
    
    def add_source(self, item: str, domain=False) -> bool:
        success = False
        if item.startswith(('http://', 'https://')):
            if domain:
                success = self.data_storage.read_from_domain(item)
            else:
                success = self.data_storage.read_from_url(item)
        elif os.path.isdir(item):
            success = self.data_storage.read_from_folder(item)
        elif os.path.isfile(item):
            success = self.data_storage.read_from_file(item)
        else:
            # Assume it's text if it's not a recognized path or URL
            success = self.data_storage.read_from_text(item)

        if success and self.data_storage.changed:
            logger.info("Data changed, saving vector store and metadata.")
            self.data_storage.save_vector_store()
            self.data_storage.document_tracker.save_metadata()
            self.data_storage.changed = False # Reset flag after saving
        elif not success:
            logger.error(f"Failed to add source: {item}")

        return success

    def set_search_weights(self, semantic_weight: float = 0.5, keyword_weight: float = 0.5) -> None:
        if not (0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1):
            raise ValueError("Weights must be between 0 and 1.")
        if abs(semantic_weight + keyword_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.")

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        logger.info(f"Search weights updated: Semantic={semantic_weight}, Keyword={keyword_weight}")

    def retrive_chunk(self, doc_id: int, chunk_id: int) -> str:
        try:
            return self.data_storage.index.return_chunk(doc_id, chunk_id)
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id} for doc {doc_id}: {e}")
            return f"[Error retrieving chunk {chunk_id}]"

    def retrive_close_chunks(self, doc_id: int, chunk_id: int, k: int = 3) -> str:
        try:
            return self.data_storage.index.return_close_chunks(doc_id, chunk_id, k)
        except Exception as e:
            logger.error(f"Error retrieving close chunks for chunk {chunk_id}, doc {doc_id}: {e}")
            return f"[Error retrieving close chunks for {chunk_id}]"

    def retrive_document_text(self, doc_id: int) -> str:
        try:
            return self.data_storage.index.return_doc(doc_id)
        except Exception as e:
            logger.error(f"Error retrieving document text for doc {doc_id}: {e}")
            return f"[Error retrieving document {doc_id}]"

    def list_sources(self) -> List[Dict[str, Any]]:
        """Returns a list of indexed sources with their metadata."""
        return self.data_storage.list_tracked_documents()
    
    def display_results(self, results: List[SearchResult], detailed: bool = True) -> None:
        if len(results) == 0:
            print("No relevant results found.")
            return

        for i, result in enumerate(results, 1):
            print(f"\n=== Result {i} ===")
            print(f"File: {result.file_name} (Doc ID: {result.doc_id})")
            print(f"Combined Score: {result.combined_score:.4f}")
            print(f"BM25 Score: {result.bm25_score:.4f}")
            print(f"Semantic Score: {result.semantic_score:.4f}")
            print(f"All Terms Present: {result.all_terms_present}")
            print(f"{len(result.chunk_id)} Relevant Chunks Found: {result.chunk_id}")

            if detailed:
                self._display_detailed_results(result)

    def get_source_content(self, doc_id: int) -> str:
        """Retrieves the full text content of a source by its doc_id."""
        logger.info(f"Retrieving content for doc_id: {doc_id}")
        return self.retrive_document_text(doc_id)

    def delete_source(self, doc_id: int) -> bool:
        """Deletes a source from the index, vector store, and tracker."""
        logger.warning(f"Attempting to delete source with doc_id: {doc_id}")
        success = self.data_storage.delete_source_by_id(doc_id)
        if success:
            logger.info(f"Successfully deleted source with doc_id: {doc_id}. Saving changes.")
            # Save changes after deletion
            self.data_storage.save_vector_store()
            self.data_storage.document_tracker.save_metadata()
        else:
            logger.error(f"Failed to delete source with doc_id: {doc_id}")
        return success
    
    def reembed_document(self, doc_id: int) -> bool:
        """
        Fully re‐processes a tracked document: rebuilds its chunks & re‐embeds.
        """
        file_path = self.data_storage.document_tracker.get_filepath_for_docid(doc_id)
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Cannot re-embed: file for doc_id {doc_id} not found.")
            return False
        # force DataStorage to treat it as modified
        self.data_storage.document_tracker.documents[file_path].chunk_ids = set()
        success = self.data_storage.read_from_file(file_path)
        if success:
            self.data_storage.save_vector_store()
            self.data_storage.document_tracker.save_metadata()
        return success

    def clear_index(self) -> bool:
        """
        Deletes every tracked source, empties both vector store & inverted index.
        """

        self.data_storage.document_tracker.reset()
        self.data_storage.index = InvertedIndex()
        self.data_storage.vector_store.delete_from_disk()
        self.data_storage.vector_store = VectorStore(
            self.data_storage.vector_store.dimension,
            path=self.data_storage.vector_store.store_path,
            temp=self.data_storage.vector_store.is_temp
        )
        if os.path.exists(self.path):
            try:
                [shutil.rmtree(self.path)]
            except Exception as e:
                logger.error(f"Error clearing data folder: {e}")
                return False
        os.makedirs(self.path, exist_ok=True)
        logger.info("Index cleared successfully.")
        
        return True

    def llm_qa_search(self, query: str, model: str = 'gpt-4.1-mini', k=3, vector_store_path: str ='data') -> str:
        context = self._hybrid_search(query, k)
        texts = [result.chunk_text for result in context]
        texts = [text for sublist in texts for text in sublist]  # Flatten the list of lists
        texts = [text for text in texts if text]  # Remove empty strings
        if not texts:
            return "I couldn't find relevant information to answer your question."

        coro = self.llm_qa.answer(query, texts, model=model)
        try:
            loop = asyncio.get_running_loop()
            # we *are* in a loop
            return loop.run_until_complete(coro)
        except RuntimeError:
            # no running loop
            return asyncio.run(coro)
      
    def _initialize_storage(self, path: str) -> None:
        if isdir(path):
            logger.info(f"Reading from directory: {path}")
            self.data_storage.read_from_folder(path)
        else:
            logger.info(f"Reading from file: {path}")
            self.data_storage.read_from_file(path)
     
    def _semantic_search(self, query: str, k: int, index_name: str = 'data') -> List[SearchResult]:
        # 1) embed
        logger.info(f"Semantic search start: query={query!r}, k={k}")
        vec = self.data_storage.create_embeddings(query)
        distances, chunk_ids, doc_ids = self.data_storage.vector_store.search(vec, index_name, k)
        results = []
        for score, cid, did in zip(distances, chunk_ids, doc_ids):
            file_name = self.data_storage.get_filename_for_docid(did) or "Unknown"
            snippet = self.retrive_chunk(did, cid)
            results.append(SearchResult(
                doc_id=did,
                file_name=file_name,
                bm25_score=0.0,
                semantic_score=score,
                combined_score=score,
                positions={},
                all_terms_present=False,
                chunk_id=[cid],
                chunk_text=[snippet]
            ))
        logger.info(f"Semantic search: returning {len(results)} results.")
        return results

    def _keyword_search(self, query: str, k: int, _unused: str = '') -> List[SearchResult]:
        logger.info(f"Keyword search start: query={query!r}, k={k}")
        hits: List[Dict] = self.data_storage.index.search(query, k)
        results = []
        # hits are dicts with keys doc_id, bm25_score, positions, all_terms_present, file_name, probably chunk_id?
        for r in hits:
            doc_id = r['doc_id']
            file_name = (r.get('file_name') or b'').decode() if isinstance(r.get('file_name'), bytes) else r.get('file_name','Unknown')
            results.append(SearchResult(
                doc_id=doc_id,
                file_name=file_name,
                bm25_score=r.get('bm25_score', 0.0),
                semantic_score=0.0,
                combined_score=r.get('bm25_score', 0.0),
                positions=r.get('positions', {}),
                all_terms_present=r.get('all_terms_present', False),
                chunk_id=r.get('chunk_id', []),
                chunk_text=[ self.retrive_chunk(doc_id, cid) for cid in r.get('chunk_id',[]) ]
            ))
        # sort descending by bm25
        logger.info(f"Keyword search: returning {len(results)} results.")
        return sorted(results, key=lambda x: x.bm25_score, reverse=True)
   
    def _hybrid_search(self, query: str, k: int, path: str = 'data') -> List[SearchResult]:
        """
        Hybrid search combining BM25 and semantic scores.
        Returns top-k documents, each with up to 3 distinct context blocks
        fetched via retrive_close_chunks(..., k=3), skipping chunk‐IDs within ±1.
        """
        logger.info(f"Hybrid search start: query={query!r}, k={k}")

        # 1) Keyword Search
        kw_raw = self.data_storage.index.search(query, k * 10)
        if not kw_raw:
            logger.warning("No keyword hits.")
            return []
        kw_by_doc: Dict[int, Dict] = {}
        for hit in kw_raw:
            did = hit['doc_id']
            if did not in kw_by_doc or hit['bm25_score'] > kw_by_doc[did]['bm25_score']:
                kw_by_doc[did] = hit
        if not kw_by_doc:
            logger.warning("No valid keyword hits after filtering.")
            return []

        # 2) Semantic Search
        q_vec = self.data_storage.create_embeddings(query)
        sem_scores, sem_cids, sem_dids = self.data_storage.vector_store.search(
            q_vec, path, k * 10
        )
        # collect all positive‐score hits per doc
        sem_hits: Dict[int, List[Dict[str, Union[float,int]]]] = {}
        for score, cid, did in zip(sem_scores, sem_cids, sem_dids):
            if score < 0: 
                continue
            sem_hits.setdefault(did, []).append({'score': score, 'chunk_id': cid})
        # keep top-3 semantic hits per doc
        for did, hits in sem_hits.items():
            hits.sort(key=lambda h: h['score'], reverse=True)
            sem_hits[did] = hits[:3]

        # 3) Combine Scores
        all_dids = set(kw_by_doc) | set(sem_hits)
        max_bm25 = max((h['bm25_score'] for h in kw_by_doc.values()), default=1.0) or 1.0

        combined: Dict[int, Dict] = {}
        for did in all_dids:
            kw = kw_by_doc.get(did, {})
            sem_list = sem_hits.get(did, [])
            bm25 = kw.get('bm25_score', 0.0)
            sem_score = (sum(h['score'] for h in sem_list) / len(sem_list)) if sem_list else 0.0
            norm_bm25 = bm25 / max_bm25
            combined_score = self.keyword_weight * norm_bm25 + self.semantic_weight * sem_score

            # chunk candidates: up to 3 semantic, or fallback to one keyword chunk_id
            if sem_list:
                cands = [h['chunk_id'] for h in sem_list]
            else:
                cid = kw.get('chunk_id')
                cands = [cid] if cid is not None else []

            combined[did] = {
                'doc_id': did,
                'bm25_score': bm25,
                'semantic_score': sem_score,
                'combined_score': combined_score,
                'chunk_candidates': cands,
                'positions': kw.get('positions', {}),
                'all_terms_present': kw.get('all_terms_present', False),
                'file_name': kw.get('file_name') or b'Unknown',
            }

        # 4) Build final SearchResult list
        final: List[SearchResult] = []
        sorted_docs = sorted(combined.values(),
                            key=lambda x: x['combined_score'],
                            reverse=True)

        for entry in sorted_docs[:k]:
            did = entry['doc_id']
            fn = entry['file_name']
            if isinstance(fn, bytes):
                fn = fn.decode('utf-8', 'replace')

            texts: List[str] = []
            chunk_ids: List[int] = []

            # dedupe chunk_ids within ±1, keep up to 3
            raw_cids = entry['chunk_candidates']
            filtered: List[int] = []
            for cid in raw_cids:
                if not any(abs(cid - prev) <= 1 for prev in filtered):
                    filtered.append(cid)
                if len(filtered) == 3:
                    break

            # fetch context for each filtered cid
            for cid in filtered:
                try:
                    block = self.retrive_close_chunks(did, cid, k=3).strip()
                    if block:
                        texts.append(block)
                        chunk_ids.append(cid)
                except Exception as e:
                    logger.error(f"Error close_chunks doc={did} cid={cid}: {e}")
                    # fallback to single chunk
                    try:
                        single = self.retrive_chunk(did, cid).strip() or "[Empty chunk]"
                        texts.append(single)
                        chunk_ids.append(cid)
                    except Exception as e2:
                        logger.error(f"Error single_chunk doc={did} cid={cid}: {e2}")
                        continue

            # final fallback if nothing retrieved
            if not texts:
                texts = ["No relevant chunk identified"]
                chunk_ids = []

            final.append(SearchResult(
                doc_id             = did,
                file_name          = fn,
                bm25_score         = entry['bm25_score'],
                semantic_score     = entry['semantic_score'],
                combined_score     = entry['combined_score'],
                positions          = entry.get('positions', {}),
                all_terms_present  = entry.get('all_terms_present', False),
                chunk_id           = chunk_ids,
                chunk_text         = texts
            ))

        logger.info(f"Hybrid search: returning {len(final)} results.")
        return final
        
    def _create_search_result(self,
                            result: Dict,
                            doc_id: str,
                            semantic_score: float,
                            combined_score: float,
                            chunk_ids: List[str]) -> SearchResult:

        file_name = result.get('file_name', b'Unknown File')
        if isinstance(file_name, bytes):
            file_name = file_name.decode()

        chunk_texts = []
        if chunk_ids:
            try:
                chunk_texts = [self.retrive_chunk(doc_id, chunk) for chunk in chunk_ids]
            except Exception as e:
                logger.error(f"Error retrieving chunks for doc {doc_id}, chunks {chunk_ids}: {e}")
                chunk_texts = ["Error retrieving chunk text."] * len(chunk_ids)


        return SearchResult(
            doc_id=doc_id,
            file_name=file_name,
            bm25_score=result.get('bm25_score', 0.0),
            semantic_score=semantic_score,
            combined_score=combined_score,
            positions=result.get('positions', {}),
            all_terms_present=result.get('all_terms_present', False),
            chunk_id=chunk_ids,
            chunk_text=chunk_texts
        )

    def _display_detailed_results(self, result: SearchResult) -> None:
        print("\nToken Positions:")
        for token, info in result.positions.items():
            print(f"  {token}: {info.get('positions', 'N/A')}")

        if not result.chunk_id:
            print("\nNo relevant chunks found for this document.")
        else:
            for i, chunk_text in enumerate(result.chunk_text):
                print(f"\nRelevant Chunk (ID: {result.chunk_id[i]}):")
                print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text) # Display snippet


def simple_search(path: Union[str, List[str]],
                 query: str,
                 k: int = 3,
                 embedding_model: EmbeddingType = EmbeddingType.MINILM_L6,
                 mode: SearchMode = SearchMode.HYBRID) -> List[SearchResult]:
    search_engine = SearchEngine(path, model_type=embedding_model, temp=True)
    return search_engine.search(query, k=k, mode=mode)

def simple_qa(query: str, path: Union[str, List[str]], k: int = 3, model: str = 'gpt-4o-mini',
                embedding_model: EmbeddingType = EmbeddingType.MINILM_L6,
                 ) -> str: # Return type is string (answer)
    search_engine = SearchEngine(path, model_type=embedding_model, temp=True)
    return search_engine.llm_qa_search(query, model=model, k=k)


if __name__ == "__main__":
    logger.info("Przykład użycia aplikacji w kodzie.")
    
    openai_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    if openai_key == "YOUR_API_KEY_HERE":
        logger.warning("OpenAI API key not set. QA functionality might be limited.")
        openai_key = None

    search_engine = SearchEngine(path='data', model_type= EmbeddingType.MINILM_L6, openai_apikey=openai_key)

    search_engine.add_source("https://pl.wikipedia.org/wiki/Papież", domain=True)
    search_engine.search("Obecny papież", k=3)
    
    search_engine.add_source("C:/Users/Jakub/Documents/Wzor_pracy_dyplomowej_AK.pdf")
    search_engine.search("Jak powinno wyglądać zakończenie pracy.", k=3)
    
    odpowiedz = search_engine.llm_qa_search("Jakie porady dodatkowe dotyczące pracy dyplomowej możesz mi dać?",
                                            model='gpt-4.1-mini', k=3)
    print(f"\nOdpowiedź modelu LLM: {odpowiedz}")
    
    logger.info("Zakończono działanie aplikacji.")