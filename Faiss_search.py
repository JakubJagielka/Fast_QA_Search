import numpy as np
import faiss
import os
from typing import List, Union, Optional
import logging
import warnings
import json
class VectorStore:
    def __init__(self, dimension: int = 1536, path: str = './vector_store', temp: bool = False):
        """
        Initialize the vector store with specified dimension.
        Uses IndexIDMap to maintain custom IDs efficiently.
        
        Args:
            dimension: Dimensionality of vectors to store
            path: Directory path for storing/loading indexes
        """
        # Handle OpenMP conflict
        if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            warnings.warn(
                "Setting KMP_DUPLICATE_LIB_OK=TRUE to handle OpenMP conflicts. "
                "Consider addressing this at the environment level for production use."
            )
            
        self.dimension = dimension
        self.indexes = {}
        self.mapping_to_doc = {}
        # Coafigure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if temp:
            # For temporary store, always create a fresh index
            self.add_index('temporary_store')
        else:
            # Load existing indexes if path exists
            if os.path.isdir(path) and len(os.listdir(path)) > 0:
                self.load(path)
            else:
                self.logger.info(f"No existing vector store found at {path}")

    def add_index(self, index_name: str) -> None:
        """
        Add a new index with specified name.
        
        Args:
            index_name: Name of the new index
        """
        if index_name in self.indexes:
            self.logger.warning(f"Index '{index_name}' already exists. Skipping creation.")
            return
            
        try:
            base_index = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIDMap(base_index)
            self.indexes[index_name] = index
            self.logger.info(f"Successfully created index '{index_name}'")
        except Exception as e:
            self.logger.error(f"Failed to create index '{index_name}': {str(e)}")
            raise

    def add_vectors(self, 
                   vectors: Union[List[float], np.ndarray],
                   doc_id: int, 
                   ids: Union[List[int], np.ndarray, int], 
                   index_name: str) -> None:
        """
        Add vectors with specified IDs.
        
        Args:
            vectors: Vector or array of vectors to add
            ids: Corresponding IDs for the vectors
            index_name: Name of the index to add vectors to
        """
        try:
            # Convert vectors to correct format
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)
            
            # Handle single vector case
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                if not isinstance(ids, (list, np.ndarray)):
                    ids = [ids]
            
            # Validate dimensions
            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch. Expected {self.dimension}, "
                    f"got {vectors.shape[1]}"
                )
            
            # Ensure index exists
            if index_name not in self.indexes:
                self.logger.info(f"Creating new index '{index_name}'")
                self.add_index(index_name)
            
            # Convert IDs to numpy array
            ids_array = np.array(ids, dtype=np.int64)
            
            if len(ids_array) != len(vectors):
                raise ValueError(
                    f"Number of IDs ({len(ids_array)}) does not match "
                    f"number of vectors ({len(vectors)})"
                )
            
            for j in ids_array:
                self.mapping_to_doc[j] = doc_id
                
            self.indexes[index_name].add_with_ids(vectors, ids_array)
            self.logger.info(
                f"Successfully added {len(vectors)} vectors to index '{index_name}'"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to index '{index_name}': {str(e)}")
            raise

    def search(self, 
              query_vector: Union[List[float], np.ndarray], 
              index_name: str, 
              k: int = 5) -> tuple[list[float], list[int]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Vector to search for
            index_name: Name of the index to search in
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, matched_ids)
        """
        try:
            if index_name not in self.indexes:
                raise KeyError(f"Index '{index_name}' not found")
                
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
                
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
                
            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch. Expected {self.dimension}, "
                    f"got {query_vector.shape[1]}"
                )
            
            distances, indices = self.indexes[index_name].search(query_vector, k)
            # Filter out -1 indices (not found)
            valid_mask = indices[0] != -1
            distances = distances[0][valid_mask]
            matched_ids = indices[0][valid_mask]
            
            doc_ids = [self.mapping_to_doc[i] for i in matched_ids]           
            
            return distances.tolist(), matched_ids.tolist(), doc_ids
            
        except Exception as e:
            self.logger.error(f"Search failed in index '{index_name}': {str(e)}")
            raise

    def save(self, directory: str) -> None:
        """
        Save the vector stores to disk.
        
        Args:
            directory: Directory to save the indexes in
        """
        try:
            os.makedirs(directory, exist_ok=True)
            for index_name, index in self.indexes.items():
                index_path = os.path.join(directory, f"{index_name}.faiss")
                faiss.write_index(index, index_path)
                self.logger.info(f"Successfully saved index '{index_name}' to {index_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save indexes to {directory}: {str(e)}")
            raise

    def load(self, directory: str) -> None:
        """
        Load vector stores from disk.
        
        Args:
            directory: Directory to load the indexes from
        """
        try:
            faiss_files = [f for f in os.listdir(directory) if f.endswith('.faiss')]
            for file in faiss_files:
                index_name = file.split('.')[0]
                index_path = os.path.join(directory, file)
                self.indexes[index_name] = faiss.read_index(index_path)
                self.logger.info(f"Successfully loaded index '{index_name}' from {index_path}")
            
            metadata = json.load(open("document_metadata.json", 'r'))
            for file_info in metadata['documents'].values():
                if type(file_info) == str:
                    continue
                doc_id = file_info['doc_id']
                chunk_ids = file_info['chunk_ids']
                for chunk_id in chunk_ids:
                    self.mapping_to_doc[chunk_id] = doc_id
                                    
                
        except Exception as e:
            self.logger.error(f"Failed to load indexes from {directory}: {str(e)}")
            raise
        
        

            