import numpy as np
import faiss
import os
from typing import List, Union, Optional, Tuple, Dict, Set
import json
from .utils._logger import get_logger
from pathlib import Path

logger = get_logger()

class VectorStore:
    def __init__(self, dimension: int, path: str = 'vector_store', temp: bool = False):
        logger.info(f"Initializing VectorStore (Dimension: {dimension}, Path: {path}, Temp: {temp})")
        if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            logger.warning("Setting KMP_DUPLICATE_LIB_OK=TRUE.")

        self.dimension = dimension
        self.indexes: Dict[str, faiss.IndexIDMap] = {}
        self.mapping_chunk_to_doc: Dict[np.int64, int] = {}
        self.store_path = path
        self.is_temp = temp
        self.metadata_filename = "document_metadata.json" 

        if not self.is_temp:
            os.makedirs(self.store_path, exist_ok=True)
            self.load(self.store_path)
        else:
            logger.info("Using temporary vector store. Data will not be persisted.")
            if 'data' not in self.indexes:
                self.add_index('data')


    def add_index(self, index_name: str) -> None:
        if index_name in self.indexes:
            logger.warning(f"Index '{index_name}' already exists. Skipping creation.")
            return

        try:
            base_index = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIDMap(base_index)
            self.indexes[index_name] = index
            logger.info(f"Successfully created index '{index_name}' with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}': {str(e)}", exc_info=True)
            raise


    def _get_metadata_path(self) -> str:
         return self.metadata_filename

    def save(self, directory: Optional[str] = None) -> None:
        if self.is_temp:
            logger.info("Skipping save for temporary vector store.")
            return

        save_dir = directory or self.store_path
        try:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving {len(self.indexes)} index(es) to {save_dir}")
            for index_name, index in self.indexes.items():
                index_path = os.path.join(save_dir, f"{index_name}.faiss")
                faiss.write_index(index, index_path)
                logger.info(f"Saved index '{index_name}' ({index.ntotal} vectors) to {index_path}")
                
        except Exception as e:
            logger.error(f"Failed to save indexes or metadata to {save_dir}: {str(e)}", exc_info=True)


    def load(self, directory: Optional[str] = None) -> None:
        if self.is_temp:
            logger.info("Skipping load for temporary vector store.")
            return

        load_dir = directory or self.store_path
        logger.info(f"Attempting to load indexes and metadata from {load_dir}")
        loaded_index_count = 0
        try:
            if not os.path.exists(load_dir):
                logger.warning(f"Directory {load_dir} does not exist. Cannot load.")
                if 'data' not in self.indexes:
                     self.add_index('data')
                return

            faiss_files = [f for f in os.listdir(load_dir) if f.endswith('.faiss')]
            if not faiss_files:
                logger.info(f"No FAISS index files found in {load_dir}.")
            else:
                for file in faiss_files:
                    index_name = Path(file).stem
                    index_path = os.path.join(load_dir, file)
                    try:
                        index = faiss.read_index(index_path)
                        if index.d != self.dimension:
                             logger.error(f"Dimension mismatch loading index '{index_name}' from {index_path}. Expected {self.dimension}, got {index.d}. Skipping.")
                             continue
                        if not isinstance(index, faiss.IndexIDMap):
                             logger.error(f"Loaded index '{index_name}' is not an IndexIDMap. Type: {type(index)}. Skipping.")
                             continue

                        self.indexes[index_name] = index
                        logger.info(f"Loaded index '{index_name}' ({self.indexes[index_name].ntotal} vectors) from {index_path}")
                        loaded_index_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load index file {index_path}: {e}. Skipping.", exc_info=True)

            logger.info(f"Finished loading {loaded_index_count} FAISS index(es).")

            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        documents = metadata.get("documents", {})
                        for doc_info in documents.values():
                            doc_id = doc_info.get("doc_id")
                            chunk_ids = doc_info.get("chunk_ids", [])
                            for chunk_id in chunk_ids:
                                self.mapping_chunk_to_doc[int(chunk_id)] = doc_id
                    logger.info(f"Loaded {len(self.mapping_chunk_to_doc)} chunk-to-doc mappings from {metadata_path}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in vector store metadata file: {metadata_path}. Mapping reset.")
                    self.mapping_chunk_to_doc = {}
                except Exception as e:
                    logger.error(f"Failed to load vector store metadata from {metadata_path}: {e}", exc_info=True)
                    self.mapping_chunk_to_doc = {}
            else:
                logger.warning(f"Vector store metadata file not found: {metadata_path}. Mapping may be incomplete or empty.")
                self.mapping_chunk_to_doc = {}

            if 'data' not in self.indexes:
                 logger.info("Default 'data' index not found after loading/initialization, creating it.")
                 self.add_index('data')

        except Exception as e:
            logger.error(f"General error during load process from {load_dir}: {str(e)}", exc_info=True)
            if 'data' not in self.indexes:
                 self.add_index('data')


    def reconstruct_by_id(self, index_name: str, chunk_id: int) -> Optional[np.ndarray]:
        """
        Reconstructs a vector from the specified FAISS index using its chunk_id.
        Returns the L2-normalized vector or None if not found or on error.
        """
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' not found for reconstruction.")
            return None

        index = self.indexes[index_name]
        try:
            vector_id = int(chunk_id)

            vector = index.reconstruct(vector_id) 

            if vector is not None and vector.size > 0:
                 vector_2d = vector.reshape(1, -1)
                 faiss.normalize_L2(vector_2d)
                 return vector_2d[0] 
            else:
                 logger.warning(f"Vector for chunk_id {chunk_id} (ID: {vector_id}) could not be reconstructed or was empty from index '{index_name}'.")
                 return None
        except TypeError as te:
            logger.error(f"TypeError during reconstruction for chunk_id {chunk_id} (ID: {vector_id}) in index '{index_name}': {te}. This might indicate a deeper binding issue.", exc_info=True)
            return None
        except RuntimeError:
            logger.warning(f"chunk_id {chunk_id} (ID: {vector_id}) not found in FAISS index '{index_name}' for reconstruction (RuntimeError).")
            return None
        except ValueError as ve:
            logger.error(f"ValueError converting chunk_id '{chunk_id}' to int for reconstruction: {ve}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reconstructing vector for chunk_id {chunk_id} (ID: {vector_id}) in index '{index_name}': {e}", exc_info=True)
            return None

    def add_vectors(self,
                   vectors: Union[np.ndarray, List[List[float]]],
                   doc_id: int,
                   chunk_ids: Union[List[int], np.ndarray],
                   index_name: str) -> None:
        if index_name not in self.indexes:
            logger.warning(f"Index '{index_name}' not found. Creating it now.")
            self.add_index(index_name)
            if index_name not in self.indexes:
                 logger.error(f"Failed to create index '{index_name}'. Cannot add vectors.")
                 return

        try:
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)

            if vectors.ndim == 1:
                if vectors.shape[0] == self.dimension:
                    vectors = vectors.reshape(1, -1)
                else:
                     raise ValueError(f"Single vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[0]}")

            if vectors.shape[0] == 0:
                 logger.warning(f"Attempted to add 0 vectors to index '{index_name}'. Skipping.")
                 return

            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}"
                )

            ids_array = np.array(chunk_ids, dtype=np.int64)

            if len(ids_array) != len(vectors):
                raise ValueError(
                    f"Number of chunk IDs ({len(ids_array)}) does not match "
                    f"number of vectors ({len(vectors)})"
                )

            faiss.normalize_L2(vectors)

            self.indexes[index_name].add_with_ids(vectors, ids_array)

            for chunk_id_np64 in ids_array:
                self.mapping_chunk_to_doc[chunk_id_np64] = doc_id

            logger.info(
                f"Added {len(vectors)} vectors with {len(ids_array)} IDs to index '{index_name}'. Total vectors: {self.indexes[index_name].ntotal}"
            )

        except Exception as e:
            logger.error(f"Failed to add vectors to index '{index_name}': {str(e)}", exc_info=True)


    def delete_vectors(self, index_name: str, doc_id: Optional[int] = None, chunk_ids: Optional[List[int]] = None):
        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' not found for deletion.")
            return 0

        if doc_id is None and chunk_ids is None:
            logger.warning("Deletion requires either doc_id or chunk_ids.")
            return 0

        try:
            ids_to_delete_set: Set[np.int64] = set()

            if chunk_ids is not None:
                ids_to_delete_set.update(np.int64(c) for c in chunk_ids)
                logger.info(f"Attempting deletion of specific chunk IDs: {list(ids_to_delete_set)}")
            elif doc_id is not None:
                ids_to_delete_set.update(k for k, v in self.mapping_chunk_to_doc.items() if v == doc_id)
                logger.info(f"Attempting deletion of all chunks associated with doc_id {doc_id}: {list(ids_to_delete_set)}")

            if not ids_to_delete_set:
                logger.warning(f"No vectors found to delete in index '{index_name}' based on criteria (doc_id={doc_id}, chunk_ids={chunk_ids}).")
                return 0

            ids_to_delete_array = np.array(list(ids_to_delete_set), dtype=np.int64)

            num_removed = self.indexes[index_name].remove_ids(ids_to_delete_array)

            deleted_count_map = 0
            for chunk_id_np64 in ids_to_delete_array:
                if chunk_id_np64 in self.mapping_chunk_to_doc:
                    del self.mapping_chunk_to_doc[chunk_id_np64]
                    deleted_count_map += 1

            logger.info(
                f"Attempted deletion of {len(ids_to_delete_array)} IDs. "
                f"FAISS reported {num_removed} vectors removed from index '{index_name}'. "
                f"Removed {deleted_count_map} entries from chunk-to-doc mapping. "
                f"Index size now: {self.indexes[index_name].ntotal}"
            )
            return num_removed

        except Exception as e:
            logger.error(f"Failed to delete vectors from index '{index_name}': {str(e)}", exc_info=True)
            return 0


    def search(self,
              query_vector: Union[List[float], np.ndarray],
              index_name: str,
              k: int = 5) -> Tuple[List[float], List[int], List[int]]:

        if index_name not in self.indexes:
            logger.error(f"Index '{index_name}' not found for search.")
            return [], [], []

        index = self.indexes[index_name]
        if index.ntotal == 0:
             logger.warning(f"Index '{index_name}' is empty. Cannot perform search.")
             return [], [], []

        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)

            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch. Expected {self.dimension}, got {query_vector.shape[1]}"
                )

            faiss.normalize_L2(query_vector)

            actual_k = min(k, index.ntotal)
            if actual_k <= 0:
                logger.warning(f"Search k is {actual_k} (original k={k}, index size={index.ntotal}), returning empty results.")
                return [], [], []

            distances, chunk_indices = index.search(query_vector, actual_k)

            distances_flat = distances[0]
            chunk_ids_flat_np64 = chunk_indices[0] 

            matched_doc_ids = []
            valid_chunk_ids_final = [] 
            valid_distances_final = []

            for i, chunk_id_np64 in enumerate(chunk_ids_flat_np64):
                if chunk_id_np64 == -1:
                    continue

                doc_id = self.mapping_chunk_to_doc.get(chunk_id_np64)

                if doc_id is not None:
                    matched_doc_ids.append(doc_id)
                    valid_chunk_ids_final.append(int(chunk_id_np64)) 
                    valid_distances_final.append(float(distances_flat[i]))
                else:
                    logger.warning(f"Found chunk_id {chunk_id_np64} in FAISS search results for index '{index_name}', but it's missing from the chunk-to-doc mapping. Skipping this result.")

            return valid_distances_final, valid_chunk_ids_final, matched_doc_ids

        except Exception as e:
            logger.error(f"Search failed in index '{index_name}': {str(e)}", exc_info=True)
            return [], [], []
        
        
    def delete_from_disk(self) -> None:
        """
        Deletes the vector store from disk if it is not temporary.
        """
        if self.is_temp:
            logger.info("Temporary vector store. No action taken.")
            return

        try:
            if os.path.exists(self.store_path):
                import shutil
                shutil.rmtree(self.store_path)
                logger.info(f"Deleted vector store directory: {self.store_path}")
            else:
                logger.warning(f"Vector store directory does not exist: {self.store_path}")

        except Exception as e:
            logger.error(f"Failed to delete vector store directory: {self.store_path}. Error: {str(e)}", exc_info=True)