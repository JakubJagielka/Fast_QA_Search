import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Set
from pathlib import Path
import sys
from _data_types import EmbeddingType

@dataclass
class DocumentMetadata:
    file_path: str
    file_hash: str
    file_size: int
    last_modified: float
    doc_id: int
    chunk_ids: Set[int] 

class DocumentTracker:
    _next_doc_id = 0  # Class variable to track the next available doc_id

    def __init__(self, metadata_path: str = "document_metadata.json", model_name: EmbeddingType = None):
        self.current_embedding_model = model_name
        self.metadata_path = metadata_path
        self.documents: Dict[str, DocumentMetadata] = {}
        self.load_metadata()

                
    def load_metadata(self):
        if Path(self.metadata_path).exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                stored_model_name = data.get("embedding_model")
                self.documents = {
                    k: DocumentMetadata(**v) for k, v in data["documents"].items()
                }
                # Update _next_doc_id to be higher than any existing doc_id
                DocumentTracker._next_doc_id = max(
                    (doc.doc_id for doc in self.documents.values()),
                    default=0
                ) + 1
                
                if stored_model_name and self.current_embedding_model:
                    stored_model = EmbeddingType[stored_model_name]
                    if stored_model != self.current_embedding_model:
                        print("New embedding model detected. This will re-vectorize all documents.")
                        print(f"Old model: {stored_model.name}")
                        print(f"New model: {self.current_embedding_model.name}")
                        a = input("Are you sure you want to continue? (y/n)")
                        if a.lower() == 'y':
                            self.current_embedding_model = stored_model
                        else:
                            sys.exit()

    def save_metadata(self) -> None:
        with open(self.metadata_path, 'w') as f:
            metadata = {
                "embedding_model": self.current_embedding_model.name if self.current_embedding_model else None,
                "documents": {k: v.__dict__ for k, v in self.documents.items()}
            }
            json.dump(metadata, f, indent=2)

    def add_chunk_ids(self, file_path: str, chunk_ids: Set[int]) -> None:
        self.documents[file_path].chunk_ids = list(chunk_ids)

    def get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def should_process_file(self, file_path: str, embedding_model: EmbeddingType) -> tuple[bool, int]:
        if self.current_embedding_model != embedding_model:
            return True, self.documents[file_path].doc_id if file_path in self.documents else self.generate_doc_id()

        path = Path(file_path)
        if not path.exists():
            return True, self.generate_doc_id()

        file_hash = self.get_file_hash(file_path)
        file_size = path.stat().st_size
        last_modified = path.stat().st_mtime

        if file_path in self.documents:
            metadata = self.documents[file_path]
            if (metadata.file_hash == file_hash and 
                metadata.file_size == file_size and 
                metadata.last_modified == last_modified):
                return False, metadata.doc_id

        doc_id = self.generate_doc_id()
        self.documents[file_path] = DocumentMetadata(
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            last_modified=last_modified,
            doc_id=doc_id,
            chunk_ids=[]
        )
        return True, doc_id
    
    def generate_doc(self, file_path: str, content: str) -> int:
        """Generate a new document entry and save content to file.
        
        Args:
            file_path: Path or identifier (e.g. URL, "text_input")
            content: Text content to save
            
        Returns:
            doc_id: Generated document ID
        """
        doc_id = self.generate_doc_id()

        # For text input and URLs, create a file to store the content
        if file_path in ["text_input"] or file_path.startswith(("http://", "https://")):
            # Create files directory if it doesn't exist
            Path("files").mkdir(exist_ok=True)
            
            # Generate filename based on source
            if file_path == "text_input":
                stored_path = Path("files") / f"text_input.txt"
            else:
                # For URLs, use sanitized filename
                url_filename = file_path.split("/")[-1][:50]  # Take last part of URL, limit length
                stored_path = Path("files") / f"url_{url_filename}.txt"
        else:
            stored_path = Path(file_path)

        # Save content to file
        with open(stored_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Create metadata entry
        self.documents[str(stored_path)] = DocumentMetadata(
            file_path=str(stored_path),
            file_hash=self.get_file_hash(stored_path),
            file_size=stored_path.stat().st_size,
            last_modified=stored_path.stat().st_mtime,
            doc_id=doc_id,
            chunk_ids=[]
        )
        
        return doc_id


    def generate_doc_id(self) -> int:
        doc_id = DocumentTracker._next_doc_id
        DocumentTracker._next_doc_id += 1
        return doc_id

    def get_processed_files(self) -> Set[str]:
        return set(self.documents.keys())