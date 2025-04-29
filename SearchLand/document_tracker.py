import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Tuple
from pathlib import Path
import os
import time
from .utils._data_types import EmbeddingType
from .utils._logger import get_logger

logger = get_logger()

@dataclass
class DocumentMetadata:
    file_path: str
    file_hash: str # Content hash for files, URLs, text
    file_size: int
    last_modified: float # File mtime or processing time for text/URL
    doc_id: int
    chunk_ids: Set[int] = field(default_factory=set)
    embedding_model_name: Optional[str] = None


class DocumentTracker:
    _next_doc_id = 0

    def __init__(self, metadata_path: str = "document_metadata.json", model_name: Optional[EmbeddingType] = None):
        logger.info("Initializing DocumentTracker")
        self.current_embedding_model = model_name
        self.metadata_path = metadata_path
        self.documents: Dict[str, DocumentMetadata] = {} # norm_path -> metadata
        self._doc_id_to_path: Dict[int, str] = {}
        self._hash_to_doc_id: Dict[str, int] = {}
        self.load_metadata()


    def load_metadata(self):
        if Path(self.metadata_path).exists():
            logger.info(f"Loading metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    stored_model_name = data.get("embedding_model")

                    loaded_docs = {}
                    max_doc_id = -1 # Start at -1 to handle empty metadata correctly
                    for path_key, v in data.get("documents", {}).items():
                        # Normalize path from stored key
                        norm_path_key = os.path.normpath(path_key)
                        v['file_path'] = norm_path_key # Ensure file_path in dict matches normalized key

                        if 'chunk_ids' in v and isinstance(v['chunk_ids'], list):
                            v['chunk_ids'] = set(v['chunk_ids'])
                        else:
                             v['chunk_ids'] = set(v.get('chunk_ids', []))

                        v['embedding_model_name'] = v.get('embedding_model_name', stored_model_name)

                        try:
                            # Ensure all required fields are present before creating dataclass instance
                            required_fields = ['file_path', 'file_hash', 'file_size', 'last_modified', 'doc_id']
                            if not all(field in v for field in required_fields):
                                logger.error(f"Error loading metadata for key '{path_key}': Missing required fields. Skipping entry. Data: {v}")
                                continue

                            metadata = DocumentMetadata(**v)
                            loaded_docs[norm_path_key] = metadata
                            self._doc_id_to_path[metadata.doc_id] = metadata.file_path
                            # Only add hash mapping if hash is valid
                            if metadata.file_hash and metadata.file_hash != "hash_error":
                                self._hash_to_doc_id[metadata.file_hash] = metadata.doc_id
                            if metadata.doc_id > max_doc_id:
                                max_doc_id = metadata.doc_id
                        except TypeError as te:
                             logger.error(f"Error loading metadata for key '{path_key}': Type mismatch - {te}. Skipping entry. Data: {v}")
                        except Exception as e:
                             logger.error(f"Error creating DocumentMetadata for key '{path_key}': {e}. Skipping entry. Data: {v}")

                    self.documents = loaded_docs
                    DocumentTracker._next_doc_id = max_doc_id + 1
                    logger.info(f"Loaded {len(self.documents)} documents. Next Doc ID: {DocumentTracker._next_doc_id}")

                    if stored_model_name and self.current_embedding_model:
                        try:
                            # Allow comparison even if stored name isn't in Enum (e.g., custom name)
                            if stored_model_name != self.current_embedding_model.name:
                                logger.warning(f"Embedding model mismatch! Metadata uses '{stored_model_name}', current is '{self.current_embedding_model.name}'. Files may be reprocessed.")
                        except Exception as e:
                             logger.error(f"Error comparing embedding model names: {e}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {self.metadata_path}. Starting fresh.")
                self.reset()
            except Exception as e:
                 logger.error(f"Unexpected error loading metadata: {e}", exc_info=True)
                 # Reset state to be safe
                 self.reset()
        else:
            logger.info(f"Metadata file {self.metadata_path} not found. Starting fresh.")
            DocumentTracker._next_doc_id = 0


    def reset(self) -> None:
        logger.info("Resetting DocumentTracker state.")
        self.documents = {}
        self._doc_id_to_path = {}
        self._hash_to_doc_id = {}
        DocumentTracker._next_doc_id = 0
        if os.path.exists(self.metadata_path):
            try:
                os.remove(self.metadata_path)
                logger.info(f"Metadata file {self.metadata_path} removed.")
            except FileNotFoundError:
                logger.warning(f"Metadata file {self.metadata_path} not found for removal.")
            except Exception as e:
                logger.error(f"Error removing metadata file: {e}")

    def save_metadata(self) -> None:
        logger.info(f"Saving metadata for {len(self.documents)} documents to {self.metadata_path}")
        try:
            with open(self.metadata_path, 'w') as f:
                serializable_docs = {}
                for k, v in self.documents.items():
                    doc_dict = v.__dict__.copy()
                    doc_dict['chunk_ids'] = sorted(list(v.chunk_ids))
                    serializable_docs[k] = doc_dict

                metadata = {
                    "embedding_model": self.current_embedding_model.name if self.current_embedding_model else None,
                    "documents": serializable_docs,
                    "last_saved": time.time()
                }
                json.dump(metadata, f, indent=2)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)


    def add_chunk_ids(self, file_path: str, chunk_ids: Set[int]) -> None:
        norm_path = os.path.normpath(file_path)
        if norm_path in self.documents:
            self.documents[norm_path].chunk_ids = chunk_ids # Overwrite with the latest set
            logger.debug(f"Set chunk IDs for {norm_path}: {chunk_ids}")
        else:
            logger.warning(f"Attempted to add chunk IDs for untracked file: {norm_path}")

    def get_chunk_ids(self, file_path: str) -> Optional[Set[int]]:
        """Returns the set of chunk IDs for a tracked file, or None if not tracked."""
        norm_path = os.path.normpath(file_path)
        metadata = self.documents.get(norm_path)
        return metadata.chunk_ids if metadata else None


    def get_content_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except FileNotFoundError:
            logger.error(f"File not found while calculating hash: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""


    def should_process_file(
        self,
        file_path: str,
        embedding_model: EmbeddingType,
        content: Optional[str] = None
    ) -> Tuple[bool, int]:
        norm_path = os.path.normpath(file_path)
        is_pseudo_path = norm_path.startswith(("text_input/", "url_")) or not ("/" in norm_path or "\\" in norm_path)

        current_time = time.time()
        content_hash = ""
        file_size = 0
        last_modified = current_time

        if is_pseudo_path:
            if content is None:
                 logger.error(f"Content must be provided for pseudo-path: {norm_path}")
                 # Cannot determine status without content, assume needs processing if untracked
                 existing_meta = self.documents.get(norm_path)
                 if existing_meta:
                      return False, existing_meta.doc_id # Assume no change if tracked
                 else:
                      # Need to generate an ID but cannot register properly
                      return True, self.generate_doc_id() # Risky, might lead to orphans

            content_bytes = content.encode('utf-8')
            content_hash = self.get_content_hash(content_bytes)
            file_size = len(content_bytes)
            # Check hash first for quick exit
            existing_doc_id_by_hash = self._hash_to_doc_id.get(content_hash)
            if existing_doc_id_by_hash is not None:
                 # Found by hash, check if the path mapping and model also match
                 existing_path = self._doc_id_to_path.get(existing_doc_id_by_hash)
                 if existing_path == norm_path:
                      metadata = self.documents.get(norm_path)
                      if metadata and metadata.embedding_model_name == embedding_model.name:
                           logger.debug(f"Pseudo-path content unchanged and model matches: {norm_path}")
                           return False, existing_doc_id_by_hash
                      else:
                           # Path matches, but model changed or metadata missing (shouldn't happen)
                           reason = "embedding model changed" if metadata else "metadata inconsistency"
                           logger.info(f"Pseudo-path needs reprocessing ({reason}): {norm_path}")
                           # Update existing metadata or register if missing
                           if metadata:
                                metadata.embedding_model_name = embedding_model.name
                                metadata.last_modified = current_time
                                metadata.chunk_ids = set() # Clear old chunks
                                return True, metadata.doc_id
                           else:
                                # Register with existing ID but new metadata
                                self._register_document(norm_path, content_hash, file_size, last_modified, existing_doc_id_by_hash, embedding_model)
                                return True, existing_doc_id_by_hash
                 else:
                      # Hash collision or same content added via different pseudo-path?
                      # Treat as new registration for this specific pseudo-path
                      logger.info(f"Content hash matches existing doc ({existing_doc_id_by_hash}), but path differs. Registering new pseudo-path: {norm_path}")
                      doc_id = self.generate_doc_id()
                      self._register_document(norm_path, content_hash, file_size, last_modified, doc_id, embedding_model)
                      return True, doc_id
            else:
                 # Hash not found, definitely new content for this pseudo-path
                 logger.info(f"New pseudo-path or content detected: {norm_path}")
                 doc_id = self.generate_doc_id()
                 self._register_document(norm_path, content_hash, file_size, last_modified, doc_id, embedding_model)
                 return True, doc_id

        else: # Handle real files
            path = Path(norm_path)
            if not path.exists():
                logger.warning(f"File does not exist: {norm_path}. Cannot determine processing status.")
                # If it was tracked, signal for deletion elsewhere. If not tracked, ignore.
                existing_meta = self.documents.get(norm_path)
                return False, existing_meta.doc_id if existing_meta else -1 # Return existing ID if known

            try:
                content_hash = self.get_file_hash(norm_path)
                file_stat = path.stat()
                file_size = file_stat.st_size
                last_modified = file_stat.st_mtime
            except Exception as e:
                logger.error(f"Error accessing file stats for {norm_path}: {e}. Assuming needs processing.")
                existing_meta = self.documents.get(norm_path)
                doc_id = existing_meta.doc_id if existing_meta else self.generate_doc_id()
                if not existing_meta:
                    # Register with error state if new
                    self._register_document(norm_path, "hash_error", 0, current_time, doc_id, embedding_model)
                else:
                     # Update existing if stats failed
                     existing_meta.file_hash = "hash_error"
                     existing_meta.last_modified = current_time
                     existing_meta.embedding_model_name = embedding_model.name
                     existing_meta.chunk_ids = set()
                return True, doc_id

            if norm_path in self.documents:
                metadata = self.documents[norm_path]
                model_matches = (metadata.embedding_model_name == embedding_model.name)

                # Check hash first, then modification time as a fallback/secondary check
                hash_matches = (metadata.file_hash == content_hash)
                # mtime_matches = (abs(metadata.last_modified - last_modified) < 1e-6) # Compare floats carefully

                if hash_matches and model_matches:
                    logger.debug(f"File unchanged (hash match) and model matches: {norm_path}")
                    # Optionally update mtime if it differs but hash is same?
                    # if not mtime_matches:
                    #     metadata.last_modified = last_modified
                    return False, metadata.doc_id

                # Determine reason for reprocessing
                if not hash_matches:
                     reason = "content hash changed"
                elif not model_matches:
                     reason = "embedding model changed"
                # elif not mtime_matches:
                #      reason = "modification time changed (hash identical)" # Less common trigger
                else:
                     reason = "unknown state change" # Should not happen

                logger.info(f"File needs reprocessing ({reason}): {norm_path}")
                # Update metadata for reprocessing
                metadata.file_hash = content_hash
                metadata.file_size = file_size
                metadata.last_modified = last_modified
                metadata.embedding_model_name = embedding_model.name
                metadata.chunk_ids = set() # Clear old chunks
                # Update hash mapping
                if content_hash and content_hash != "hash_error":
                     self._hash_to_doc_id[content_hash] = metadata.doc_id
                return True, metadata.doc_id

            else:
                # New file
                logger.info(f"New file detected: {norm_path}")
                doc_id = self.generate_doc_id()
                self._register_document(norm_path, content_hash, file_size, last_modified, doc_id, embedding_model)
                return True, doc_id

    def _register_document(self, file_path, file_hash, file_size, last_modified, doc_id, embedding_model):
        norm_path = os.path.normpath(file_path)
        if not file_hash:
             logger.warning(f"Registering document {norm_path} (ID: {doc_id}) with empty hash.")
             file_hash = f"empty_hash_{doc_id}" # Use a placeholder

        metadata = DocumentMetadata(
            file_path=norm_path,
            file_hash=file_hash,
            file_size=file_size,
            last_modified=last_modified,
            doc_id=doc_id,
            chunk_ids=set(),
            embedding_model_name=embedding_model.name if embedding_model else None
        )
        self.documents[norm_path] = metadata
        self._doc_id_to_path[doc_id] = norm_path
        # Only map valid hashes
        if file_hash and file_hash != "hash_error" and not file_hash.startswith("empty_hash_"):
             self._hash_to_doc_id[file_hash] = doc_id
        logger.debug(f"Registered document: {norm_path} (ID: {doc_id})")


    def delete_file(self, file_path: str) -> Optional[Set[int]]:
        norm_path = os.path.normpath(file_path)
        metadata = self.documents.pop(norm_path, None) # Use pop to get and remove

        if metadata is None:
            logger.warning(f"File not found in metadata for deletion: {norm_path}")
            return None # Indicate not found

        doc_id = metadata.doc_id
        chunk_ids = metadata.chunk_ids
        file_hash = metadata.file_hash

        if doc_id in self._doc_id_to_path:
            del self._doc_id_to_path[doc_id]

        # Remove from hash map only if the hash exists and maps to the deleted doc_id
        if file_hash and file_hash in self._hash_to_doc_id and self._hash_to_doc_id[file_hash] == doc_id:
             del self._hash_to_doc_id[file_hash]

        logger.info(f"Deleted metadata for: {norm_path} (Doc ID: {doc_id})")
        try:
            self.save_metadata()
            return chunk_ids
        except Exception as e:
             logger.error(f"Failed to save metadata after deleting {norm_path}: {e}")
             # Should we rollback? Difficult. Return chunks but log error.
             return chunk_ids # Return chunks even if save failed, but state is inconsistent


    def generate_doc_id(self) -> int:
        doc_id = DocumentTracker._next_doc_id
        DocumentTracker._next_doc_id += 1
        return doc_id

    def get_processed_files(self) -> Set[str]:
        return set(self.documents.keys())

    def get_filepath_for_docid(self, doc_id: int) -> Optional[str]:
        return self._doc_id_to_path.get(doc_id)

    def get_docid_for_filepath(self, file_path: str) -> Optional[int]:
        norm_path = os.path.normpath(file_path)
        metadata = self.documents.get(norm_path)
        return metadata.doc_id if metadata else None

    def find_doc_by_hash(self, content_hash: str) -> Optional[int]:
        return self._hash_to_doc_id.get(content_hash)