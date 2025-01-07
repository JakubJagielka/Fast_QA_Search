from abc import ABC, abstractmethod


class FileHandler(ABC):
    @abstractmethod
    def read_content(self, file_path: str) -> bytes:
        """Read file content and return as bytes"""
        pass

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Check if handler can process this file type"""
        pass

class TxtHandler(FileHandler):
    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith('.txt')
    
    def read_content(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()

class MarkdownHandler(FileHandler):
    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith('.md')
    
    def read_content(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()

class PdfHandler(FileHandler):
    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def read_content(self, file_path: str) -> bytes:
        try:
            import PyPDF2
            content = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content.append(page.extract_text())
            return '\n'.join(content).encode('utf-8')
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install it with: pip install PyPDF2")
        
class DocxHandler(FileHandler):
    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith('.docx')
    
    def read_content(self, file_path: str) -> bytes:
        try:
            from docx import Document
            doc = Document(file_path)
            content = []
            for paragraph in doc.paragraphs:
                content.append(paragraph.text)
            return '\n'.join(content).encode('utf-8')
        except ImportError:
            raise ImportError("python-docx is required for DOCX support. Install it with: pip install python-docx")

class HtmlHandler(FileHandler):
    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.html', '.htm'))
    
    def read_content(self, file_path: str) -> bytes:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Extract text while removing script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                return text.encode('utf-8')
        except ImportError:
            raise ImportError("BeautifulSoup4 is required for HTML support. Install it with: pip install beautifulsoup4")


class FileHandlerFactory:
    def __init__(self):
        self.handlers = [
            TxtHandler(),
            MarkdownHandler(),
            PdfHandler(),
            DocxHandler(),
            HtmlHandler()
        ]
    
    def get_handler(self, file_path: str) -> FileHandler:
        
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return handler.read_content(file_path)
        raise ValueError(f"Not supported file type")