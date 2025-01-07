import datetime
import json
from cython_files.Data_Struct import InvertedIndex
from embeddingModels_api import EmbeddingOpenAI
from embeddingModels import MiniLM_L6
import os
import asyncio
from Faiss_search import VectorStore
from DocumentTracker import DocumentTracker
from file_handlers import FileHandlerFactory
from _data_types import *


class DataStorage():

    def __init__(self, model_type: EmbeddingType = EmbeddingType.MINILM_L6, openai_client = None, temp: bool = False):
        self.openai_client = openai_client
        self.changed = False
        self.index = InvertedIndex()
        self.model_type = model_type
        self.embedding_model = self.choose_embedding_model(model_type)
        self.vector_store = VectorStore(model_type.value ,temp=temp)
        self.document_tracker = DocumentTracker(model_name=model_type)



    def choose_embedding_model(self, model_type: EmbeddingType):
        match(model_type):
            case EmbeddingType.OPENAI_SMALL:
                if self.openai_client is None:
                    raise ValueError("API key required for OpenAI embeddings")
                return EmbeddingOpenAI(self.openai_client, "text-embedding-3-small")
            
            case EmbeddingType.OPENAI_LARGE:
                if self.openai_client is None:
                    raise ValueError("API key required for OpenAI embeddings")
                return EmbeddingOpenAI(self.openai_client, "text-embedding-3-large")
            
            case EmbeddingType.MINILM_L6:
                return MiniLM_L6()
        
        
    def read_from_file(self, file_path):
        """Process a single file and add it to the index and vector store"""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
                
            should_process, doc_id = self.document_tracker.should_process_file(file_path, self.model_type)
            folder_path = os.path.dirname(file_path)
            print(file_path, os.path.basename(file_path))
            file_name = os.path.basename(file_path)

            if should_process:
                print(f"Processing new/modified file: {file_name}")
                content = FileHandlerFactory().get_handler(file_path)
                text, chunk_ids, _ = self.index.add_txt(content, doc_id, file_name)
                embeddings = self.create_embeddings(text)
                self.vector_store.add_vectors(embeddings, doc_id, chunk_ids, folder_path)
                self.document_tracker.add_chunk_ids(file_path, chunk_ids)
                self.changed = True
            else:
                print(f"Skipping unchanged file: {file_name}")
                content = FileHandlerFactory().get_handler(file_path)
                self.index.add_txt(content, doc_id, file_name)

            self.document_tracker.save_metadata()
            return True
                
        except Exception as e:
            print(f"Error processing file: {file_name} - {e}")
            return False

    def read_from_folder(self, folder_path):
        """Process all files in a folder"""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return False
            
        try:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                self.read_from_file(file_path)
                
            self.document_tracker.save_metadata()
            return True
                
        except Exception as e:
            print(f"Error processing folder {folder_path}: {str(e)}")
            return False
        
        
    def read_from_text(self, text: str, json_path: str = "files/text_content.json") -> bool:
        """Process raw text string and add it to the index and vector store. 
        Saves/loads content from JSON file.
        
        Args:
            text: Text content to process
            json_path: Path to JSON file for persistence
            
        Returns:
            bool: Success status
        """
        try:
            # Check if content already exists in JSON
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    # Check if this exact text was already processed
                    if text in stored_data:
                        print(f"Loading existing content from {json_path}")
                        doc_id = stored_data[text]["doc_id"]
                        chunk_ids = stored_data[text]["chunk_ids"]
                        
                        # Add to index and vector store
                        text_bytes = text.encode('utf-8')
                        processed_text, _, _ = self.index.add_txt(text_bytes, doc_id, "text_input")
                        self.document_tracker.add_chunk_ids('files\\text_input.txt', chunk_ids)
                        return True

            # Process new text content
            doc_id = self.document_tracker.generate_doc("text_input", text)
            text_bytes = text.encode('utf-8')
            processed_text, chunk_ids, _ = self.index.add_txt(text_bytes, doc_id, "text_input")
            embeddings = self.create_embeddings(processed_text)
            self.vector_store.add_vectors(embeddings, doc_id, chunk_ids, "files")
            self.document_tracker.add_chunk_ids('files\\text_input.txt', chunk_ids)
            self.changed = True

            # Save to JSON
            content_data = {
                "doc_id": doc_id,
                "chunk_ids": list(chunk_ids),
                "timestamp": str(datetime.datetime.now())
            }
            
            stored_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    
            stored_data[text] = content_data
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(stored_data, f, indent=2)
                
            print(f"Saved new content to {json_path}")
            return True

        except Exception as e:
            print(f"Error processing text input: {e}")
            return False
    def read_from_url(self, url: str) -> bool:
        """Process content from URL and add it to the index and vector store"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            doc_id = self.document_tracker.generate_doc_id(url)
            text_bytes = text.encode('utf-8')
            processed_text, chunk_ids, _ = self.index.add_txt(text_bytes, doc_id, url)
            embeddings = self.create_embeddings(processed_text)
            self.vector_store.add_vectors(embeddings, doc_id, chunk_ids, "files")
            self.document_tracker.add_chunk_ids('files', chunk_ids)
            self.changed = True
            return True
            
        except ImportError:
            print("requests and beautifulsoup4 are required for URL support. Install with: pip install requests beautifulsoup4")
            return False
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return False
        
    
    def save_vector_store(self, directory= 'vector_store'):
        self.vector_store.save(directory)
            
            
    def create_embeddings(self, text):
        if isinstance(self.embedding_model, EmbeddingOpenAI):
            if type(text) == str:
                return asyncio.run(self.embedding_model.create_embedding(text))
            return asyncio.run(self.embedding_model.get_embeddings(text))
        else:
            return self.embedding_model.get_embeddings(text)