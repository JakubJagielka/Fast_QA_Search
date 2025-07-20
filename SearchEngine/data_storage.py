import re
import numpy as np
from .cython_files.Data_Struct import InvertedIndex 
from .embeddingModels_api import EmbeddingOpenAI
from .embeddingModels import MiniLM_L6
import os
import asyncio
from typing import List, Dict, Any, Optional
from .faiss_search import VectorStore
from .document_tracker import DocumentTracker, DocumentMetadata
from .file_handlers import FileHandlerFactory
from .utils._data_types import *
from .utils._logger import get_logger


logger = get_logger()

class DataStorage():
    def __init__(self, model_type: EmbeddingType = EmbeddingType.MINILM_L6, openai_client = None, temp: bool = False):
        logger.info("Initializing DataStorage")
        self.openai_client = openai_client
        self.changed = False
        logger.info("Creating InvertedIndex")
        self.index = InvertedIndex()
        self.model_type = model_type
        self.embedding_model = self.choose_embedding_model(model_type)
        self.vector_store_path = 'vector_store' # Hardcoded path to create a vector store folder
        self.vector_store = VectorStore(model_type.value, path=self.vector_store_path, temp=temp)
        self.document_tracker = DocumentTracker(model_name=model_type)
        self.file_handler_factory = FileHandlerFactory()
        self._load_existing_index_data()
        logger.info("DataStorage initialized successfully")

    def _load_existing_index_data(self):
        logger.info("Loading existing document data into InvertedIndex...")
        loaded_count = 0
        missing_files = []
        for file_path, metadata in self.document_tracker.documents.items():
            try:
                # Check if file exists, unless it's a pseudo-path (like URL or text input)
                is_pseudo_path = os.path.basename(file_path).startswith(("url_", "text_input_"))
                if is_pseudo_path or os.path.exists(file_path):
                    # Only add to index if it doesn't exist there already
                    if not self.index.has_doc(metadata.doc_id):
                        content = self.file_handler_factory.get_handler(file_path)
                        if content:
                            # Use add_txt to populate the inverted index only
                            self.index.add_txt(content.decode('utf-8', 'replace') , metadata.doc_id, os.path.basename(file_path), list(metadata.chunk_ids)[0])
                            loaded_count += 1
                        else:
                            logger.warning(f"Could not retrieve content for {file_path}, skipping index load.")
                else:
                    logger.warning(f"Tracked file not found, skipping index load: {file_path}")
                    missing_files.append(file_path)
            except Exception as e:
                logger.error(f"Error loading file into index {file_path}: {str(e)}")

        # Optional: Clean up tracker for missing files found during load
        # for missing_path in missing_files:
        #     doc_id_to_remove = self.document_tracker.get_docid_for_filepath(missing_path)
        #     if doc_id_to_remove is not None:
        #         logger.warning(f"Removing missing file {missing_path} (Doc ID: {doc_id_to_remove}) from tracker during startup.")
        #         self.delete_source_by_id(doc_id_to_remove) # Be careful with recursive calls or state changes here

        logger.info(f"Loaded content for {loaded_count} documents into InvertedIndex.")

    def choose_embedding_model(self, model_type: EmbeddingType):
        logger.info(f"Choosing embedding model: {model_type}")
        if model_type == EmbeddingType.OPENAI_SMALL:
            if not self.openai_client: raise ValueError("API key required for OpenAI embeddings")
            logger.info("Using OpenAI small embedding model")
            return EmbeddingOpenAI(self.openai_client, "text-embedding-3-small")
        elif model_type == EmbeddingType.OPENAI_LARGE:
            if not self.openai_client: raise ValueError("API key required for OpenAI embeddings")
            logger.info("Using OpenAI large embedding model")
            return EmbeddingOpenAI(self.openai_client, "text-embedding-3-large")
        elif model_type == EmbeddingType.MINILM_L6:
            logger.info("Using MiniLM_L6 embedding model")
            return MiniLM_L6()
        else:
            raise NotImplementedError(f"Embedding model type {model_type} not supported.")

    def read_from_file(self, file_path, from_folder: bool = False) -> bool:
        try:
            file_name = os.path.basename(file_path)
            index_name = 'data'

            # Check existence first, handle missing files by attempting deletion
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}. Attempting to remove from tracking.")
                doc_id_to_remove = self.document_tracker.get_docid_for_filepath(file_path)
                if doc_id_to_remove is not None:
                    self.delete_source_by_id(doc_id_to_remove)
                else:
                     logger.warning(f"Could not find doc_id for missing file: {file_path}")
                return False # Indicate failure for this specific file

            # Determine if processing is needed (new, modified, or model change)
            should_process, doc_id = self.document_tracker.should_process_file(file_path, self.model_type)

            if should_process:
                logger.info(f"Processing new/modified file: {file_name} (Doc ID: {doc_id})")
                content = self.file_handler_factory.get_handler(file_path)
                if not content:
                     logger.warning(f"No content retrieved from file: {file_name}. Skipping processing.")
                     # Track the file even if empty? Yes, tracker handles this.
                     self.document_tracker.add_chunk_ids(file_path, set()) # Track with empty chunks
                     self.changed = True # Mark as changed to save metadata
                     return True # File was handled (even if empty)

                # Delete existing vectors if reprocessing
                existing_chunks = self.document_tracker.get_chunk_ids(file_path)
                if existing_chunks:
                    logger.info(f"Deleting existing {len(existing_chunks)} vectors for reprocessing doc_id {doc_id}")
                    self.vector_store.delete_vectors(index_name=index_name, chunk_ids=list(existing_chunks))
                    # Note: delete_vectors also updates the chunk->doc mapping

                # Add to inverted index and get chunks/IDs
                processed_chunks, chunk_ids_list, _ = self.index.add_txt(content.decode('utf-8', 'replace') , doc_id, file_name)
                chunk_ids_set = set(chunk_ids_list) # Convert list from C++ to set

                if processed_chunks:
                    embeddings = self.create_embeddings(processed_chunks)
                    if embeddings.size > 0:
                        self.vector_store.add_vectors(embeddings, doc_id, chunk_ids_list, index_name)
                        self.document_tracker.add_chunk_ids(file_path, chunk_ids_set)
                        self.changed = True
                        logger.info(f"Added {len(processed_chunks)} chunks with embeddings for {file_name}")
                    else:
                        logger.error(f"Failed to generate embeddings for {file_name}. No vectors added.")
                        self.document_tracker.add_chunk_ids(file_path, set()) # Track with empty chunks if embedding failed
                        self.changed = True
                else:
                    logger.warning(f"No processable content found in file: {file_name}")
                    self.document_tracker.add_chunk_ids(file_path, set())
                    self.changed = True

                if not from_folder:
                    self.document_tracker.save_metadata() # Save tracker state after processing

            else:
                logger.info(f"Skipping unchanged file: {file_name} (Doc ID: {doc_id})")
                # Ensure content is loaded into inverted index if not already present
                if not self.index.has_doc(doc_id):
                    logger.info(f"Loading content for unchanged file into index: {file_name}")
                    content = self.file_handler_factory.get_handler(file_path)
                    if content:
                        self.index.add_txt(content.decode('utf-8', 'replace') , doc_id, file_name)
                    else:
                        logger.warning(f"Could not load content for unchanged file: {file_name}")

            logger.debug(f"Finished processing file: {file_name}")
            return True # Indicate success for this file

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return False # Indicate failure for this file

    def read_from_folder(self, folder_path):
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return False
        try:
            logger.info(f"Processing folder: {folder_path}")
            norm_folder_path = os.path.normpath(folder_path)

            tracked_files_in_folder = {
                fp: meta for fp, meta in self.document_tracker.documents.items()
                if os.path.normpath(os.path.dirname(fp)).startswith(norm_folder_path)
            }
            logger.debug(f"Tracked files potentially in folder: {list(tracked_files_in_folder.keys())}")

            current_files_in_folder = set()
            for root, _, files in os.walk(folder_path):
                for file in files:
                    try:
                        # Attempt to get handler to filter unsupported types early
                        full_path = os.path.normpath(os.path.join(root, file))
                        if self.file_handler_factory.is_supported(full_path):
                             current_files_in_folder.add(full_path)
                        else:
                             logger.debug(f"Skipping unsupported file type: {full_path}")
                    except Exception as e:
                         logger.warning(f"Error checking file support for {full_path}: {e}")

            logger.debug(f"Current supported files on disk: {list(current_files_in_folder)}")

            files_processed_or_skipped = set()
            any_change_detected = False

            # Process tracked files (check modifications / deletions)
            for file_path in tracked_files_in_folder.keys():
                if file_path in current_files_in_folder:
                    if self.read_from_file(file_path, True): # read_from_file returns True on success/skip, False on error
                         files_processed_or_skipped.add(file_path)
                         if self.changed: # Check if read_from_file set the changed flag
                              any_change_detected = True
                              self.changed = False # Reset flag after acknowledging change
                    else:
                         logger.error(f"Failed to process tracked file: {file_path}")
                         # Decide if we should remove it from tracking on error? Maybe not automatically.
                         files_processed_or_skipped.add(file_path) # Mark as handled (even if error)
                else:
                    logger.warning(f"Tracked file missing, removing: {file_path}")
                    doc_id_to_remove = self.document_tracker.get_docid_for_filepath(file_path)
                    if doc_id_to_remove is not None:
                        if self.delete_source_by_id(doc_id_to_remove):
                             any_change_detected = True
                    files_processed_or_skipped.add(file_path)

            # Process new files
            new_files = current_files_in_folder - files_processed_or_skipped
            logger.info(f"Found {len(new_files)} new files to process.")
            for file_path in new_files:
                if self.read_from_file(file_path, True):
                     if self.changed:
                          any_change_detected = True
                          self.changed = False # Reset flag
                else:
                     logger.error(f"Failed to process new file: {file_path}")

            self.changed = any_change_detected
            self.save_vector_store() 
            self.document_tracker.save_metadata() 
        

            logger.info(f"Finished processing folder: {folder_path}. Overall change detected: {self.changed}")
            return True

        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {str(e)}", exc_info=True)
            return False

    def read_from_text(self, text: str, source_label: str = "text_input") -> bool:
        logger.info(f"Processing text input: {source_label}")
        try:
            content_bytes = text.encode('utf-8')
            text_hash = self.document_tracker.get_content_hash(content_bytes)
            # Use a more descriptive pseudo-path including the hash
            pseudo_file_path = os.path.normpath(f"text_input/{text_hash}.txt")
            file_identifier = os.path.basename(pseudo_file_path) # text_hash.txt

            # Check if this specific text content (by hash) needs processing
            # Pass content directly to should_process_file for hash check
            should_process, doc_id = self.document_tracker.should_process_file(
                pseudo_file_path, self.model_type, content=text
            )

            if should_process:
                logger.info(f"Adding new text content as pseudo-file: {pseudo_file_path} (Doc ID: {doc_id})")

                # Delete existing vectors if reprocessing (unlikely for text, but possible if model changes)
                existing_chunks = self.document_tracker.get_chunk_ids(pseudo_file_path)
                if existing_chunks:
                    logger.info(f"Deleting existing {len(existing_chunks)} vectors for reprocessing text doc_id {doc_id}")
                    self.vector_store.delete_vectors(index_name='data', chunk_ids=list(existing_chunks))

                # Add to inverted index
                processed_chunks, chunk_ids_list, _ = self.index.add_txt(content_bytes.decode('utf-8', 'replace') , doc_id, file_identifier)
                chunk_ids_set = set(chunk_ids_list)

                if processed_chunks:
                    embeddings = self.create_embeddings(processed_chunks)
                    if embeddings.size > 0:
                        self.vector_store.add_vectors(embeddings, doc_id, chunk_ids_list, 'data')
                        self.document_tracker.add_chunk_ids(pseudo_file_path, chunk_ids_set)
                        self.changed = True
                        logger.info(f"Added {len(processed_chunks)} chunks for text input: {file_identifier}")
                    else:
                        logger.error(f"Failed to generate embeddings for text input {file_identifier}. No vectors added.")
                        self.document_tracker.add_chunk_ids(pseudo_file_path, set())
                        self.changed = True
                else:
                    logger.warning(f"No processable content generated for text input: {file_identifier}")
                    self.document_tracker.add_chunk_ids(pseudo_file_path, set())
                    self.changed = True

                self.document_tracker.save_metadata() # Save tracker state
                return True
            else:
                logger.info(f"Text content already indexed and unchanged: {pseudo_file_path} (Doc ID: {doc_id})")
                # Ensure content is loaded into inverted index if not already present
                if not self.index.has_doc(doc_id):
                     logger.info(f"Loading existing text content into index: {pseudo_file_path}")
                     self.index.add_txt(content_bytes.decode('utf-8', 'replace') , doc_id, file_identifier)
                return True

        except Exception as e:
            logger.error(f"Error processing text input: {e}", exc_info=True)
            return False

    def read_from_url(self, url: str) -> bool:
        logger.info(f"Processing URL: {url}")
        try:
            content_bytes = self.file_handler_factory.get_handler(url)
            if not content_bytes:
                logger.error(f"Failed to fetch content from URL: {url}")
                return False

            text = content_bytes.decode('utf-8', errors='replace')
            # Use read_from_text logic for processing and deduplication
            return self.read_from_text(text, source_label=url)

        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}", exc_info=True)
            return False

    def read_from_domain(self, url: str, save_to_disk: bool = False, max_pages: int = 20, save_dir: str = "data/crawled") -> bool:
        logger.info(f"Processing domain crawl for: {url}")
        try:
            domain_content = self.file_handler_factory.get_domain(url, max_pages)
            if not isinstance(domain_content, dict):
                logger.error(f"Domain handler did not return a dict for {url}")
                return False

            processed_any = False
            any_change_detected = False

            if save_to_disk:
                os.makedirs(save_dir, exist_ok=True)

            for page_url, text in domain_content.items():
                logger.debug(f"Processing crawled page: {page_url}")
                content_bytes = text.encode('utf-8')
                content_hash = re.sub(r'[\W_]+', '_', page_url)

                if save_to_disk:
                    # Save to disk, use hash for unique filename within save_dir
                    file_name = f"{content_hash}.txt"
                    file_path = os.path.normpath(os.path.join(save_dir, file_name))
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        # Process using the saved file path
                        if self.read_from_file(file_path, True):
                             processed_any = True
                             if self.changed:
                                  any_change_detected = True
                                  self.changed = False # Reset flag
                        else:
                             logger.error(f"Failed processing saved crawled page: {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving or processing crawled page {page_url} to {file_path}: {e}")
                else:
                    # Process directly using read_from_text logic (in-memory)
                    if self.read_from_text(text, source_label=page_url):
                        processed_any = True
                        if self.changed:
                            any_change_detected = True
                            self.changed = False # Reset flag
                    else:
                        logger.error(f"Failed processing crawled page in memory: {page_url}")

            self.changed = any_change_detected # Set overall change flag
            self.save_vector_store() # Save vector store after processing
            self.document_tracker.save_metadata() # Save tracker state
            logger.info(f"Finished processing domain crawl for: {url}. Overall change detected: {self.changed}")
            return processed_any # Return True if at least one page was processed successfully

        except Exception as e:
            logger.error(f"Error processing domain crawl {url}: {e}", exc_info=True)
            return False

    def save_vector_store(self, directory=None):
        save_dir = directory or self.vector_store_path
        logger.info(f"Saving vector store to {save_dir}")
        self.vector_store.save(save_dir)

    def create_embeddings(self, text_or_list):
        if not text_or_list:
            return np.array([])

        is_single_string = isinstance(text_or_list, str)
        texts = [text_or_list] if is_single_string else text_or_list

        # Filter out empty strings which can cause issues with some models
        if not texts:
             logger.warning("Input text list contains only empty strings. Returning empty embeddings.")
             return np.array([]) if is_single_string else np.empty((0, self.embedding_model.dim))

        try:
            if isinstance(self.embedding_model, EmbeddingOpenAI):
                embeddings = asyncio.run(self.embedding_model.get_embeddings(texts))
            else:
                embeddings = self.embedding_model.get_embeddings(texts)

            # Handle case where input was single string but invalid
            if is_single_string and embeddings.shape[0] == 0:
                 return np.array([])
            # Handle case where input was list but all invalid
            if not is_single_string and embeddings.shape[0] == 0:
                 return np.empty((0, self.embedding_model.dim))

            # If input was single string and valid, return the single embedding vector
            if is_single_string:
                 return embeddings[0]

            # If input was list, we need to return an array corresponding to the original list length,
            # potentially inserting zero vectors for the filtered-out empty strings.
            # This is complex. For now, we return embeddings only for valid texts.
            # Caller needs to be aware the returned array might be smaller than input list.
            # TODO: Consider returning a masked array or mapping if exact correspondence is needed.
            if len(texts) != len(texts):
                 logger.warning(f"Generated embeddings for {len(texts)} non-empty strings out of {len(texts)} total.")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}", exc_info=True)
            return np.array([]) if is_single_string else np.empty((0, self.embedding_model.dim))

    def list_tracked_documents(self) -> List[Dict[str, Any]]:
        docs = []
        for metadata in self.document_tracker.documents.values():
            file_path = metadata.file_path
            doc_type = "unknown"
            base_name = os.path.basename(file_path)
            if base_name.startswith("url_") or file_path.startswith("http"): # Check prefix and path start
                 doc_type = "url"
            elif base_name.startswith("text_input_") or file_path.startswith("text_input/"):
                 doc_type = "text"
            elif os.path.exists(file_path): # Check if it's an actual file path that exists
                 doc_type = "file"
            elif "/" in file_path or "\\" in file_path: # Assume file if it looks like a path but doesn't exist
                 doc_type = "file (missing)"
            # Add more specific checks if needed (e.g., based on save_dir for crawled files)

            docs.append({
                "doc_id": metadata.doc_id,
                "file_path": metadata.file_path, # Keep original path for identification
                "file_size": metadata.file_size,
                "last_modified": metadata.last_modified,
                "chunk_count": len(metadata.chunk_ids) if metadata.chunk_ids else 0,
                "type": doc_type,
                "model": metadata.embedding_model_name
            })
        return sorted(docs, key=lambda x: x['file_path'])
    
    def delete_source_by_id(self, doc_id: int) -> bool:
        logger.info(f"Attempting deletion for doc_id: {doc_id}")
        file_path = self.document_tracker.get_filepath_for_docid(doc_id)

        if file_path is None:
            logger.error(f"Cannot delete: No file path found for doc_id {doc_id}.")
            return False

        try:
            # 1. Remove from Document Tracker (returns chunk_ids)
            # delete_file now also saves metadata
            chunk_ids_to_delete = self.document_tracker.delete_file(file_path)
            if chunk_ids_to_delete is None: # Indicates deletion failed in tracker
                 logger.error(f"Failed to remove doc_id {doc_id} ({file_path}) from tracker.")
                 return False # Stop deletion process if tracker fails
            logger.info(f"Removed doc_id {doc_id} ({file_path}) from tracker. Associated chunks: {chunk_ids_to_delete}")

            # 2. Remove from Vector Store
            if chunk_ids_to_delete:
                index_name = 'data'
                num_vec_removed = self.vector_store.delete_vectors(index_name=index_name, chunk_ids=list(chunk_ids_to_delete))
                logger.info(f"Requested deletion of {len(chunk_ids_to_delete)} vectors for doc_id {doc_id} from index '{index_name}'. Actual removed: {num_vec_removed}.")
                # Note: delete_vectors handles updating the chunk->doc mapping internally
            else:
                 logger.info(f"No chunk IDs associated with doc_id {doc_id}, skipping vector deletion.")

            # 3. Remove from Inverted Index
            try:
                 if hasattr(self.index, 'delete_doc') and self.index.has_doc(doc_id):
                     if self.index.delete_doc(doc_id):
                         logger.info(f"Removed doc_id {doc_id} from InvertedIndex.")
                     else:
                         logger.warning(f"InvertedIndex.delete_doc reported failure for doc_id {doc_id}.")
                 elif not hasattr(self.index, 'delete_doc'):
                     logger.warning(f"InvertedIndex class does not have 'delete_doc' method. Keyword index for doc_id {doc_id} may remain.")
                 else:
                      logger.info(f"Doc_id {doc_id} not found in InvertedIndex, skipping deletion.")
            except Exception as e:
                 logger.error(f"Error removing doc_id {doc_id} from InvertedIndex: {e}")

            # 4. Mark state as changed (caller might need to save vector store)
            self.changed = True

            # 5. Persist changes (Tracker metadata saved by delete_file)
            # Caller (e.g., SearchEngine) should handle saving the vector store if needed

            logger.info(f"Successfully processed deletion request for doc_id: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error during deletion process for doc_id {doc_id} ({file_path}): {e}", exc_info=True)
            return False

    def get_filename_for_docid(self, doc_id: int) -> Optional[str]:
        file_path = self.document_tracker.get_filepath_for_docid(doc_id)
        return os.path.basename(file_path) if file_path else None