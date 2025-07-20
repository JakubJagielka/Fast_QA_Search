from abc import ABC, abstractmethod
import asyncio
import ssl
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
from .utils._logger import get_logger
try:
    from robotsparser.parser import RobotsParser
except ImportError:
    RobotsParser = None # type: ignore # Handle optional dependency
    
    
logger = get_logger()

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
        
class WebPageHandler(FileHandler):
    def can_handle(self, url: str) -> bool:
        return url.lower().startswith(('http://', 'https://')) or url.lower().startswith(('www.')) or url.lower().endswith(('.com', '.org', '.net', '.edu', '.gov', '.io', '.co','.pl'))
    
    def read_content(self, url: str) -> bytes:
        try:
            import requests
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            return text.encode('utf-8')
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return b""
    
    
class HtmlDomainHandler:
    def __init__(
        self,
        max_pages: int = 10000,
        request_delay: float = 0.2,
        timeout: int = 10,
        max_concurrent_requests: int = 20,
        queue_maxsize: int = 10000,
        max_links_per_page: int = 100,
        allowed_domains: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
    ):
        self.max_pages = max_pages
        self.request_delay = request_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent_requests = max_concurrent_requests
        self.queue_maxsize = queue_maxsize
        self.max_links_per_page = max_links_per_page

        self.allowed_domains = allowed_domains
        self.allowed_paths = allowed_paths

        self.results: Dict[str, str] = {}
        self.results2: Dict[str, str] = {}
        self.visited_urls: Set[str] = set()
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_maxsize)
        self.base_domain: Optional[str] = None
        self.base_scheme: Optional[str] = None

    def _normalize_url(self, url: str) -> str:
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        query = "&".join(sorted(parsed.query.split("&"))) if parsed.query else ""
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized

    def _is_within_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            return False
        if self.allowed_paths:
            for prefix in self.allowed_paths:
                if parsed.path.startswith(prefix):
                    return True
            return False
        return parsed.netloc == self.base_domain

    def _should_enqueue(self, url: str) -> bool:
        banned_patterns = [
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".tar", ".rar",
            "/calendar", "/gallery", "/galeria", "/img", "/image", "/images",
            "?action=", "?session", "?sid=", "?sort=", "?order=", "?filter=",
            "&action=", "&session", "&sid=", "&sort=", "&order=", "&filter=",
            "logout", "login", "register", "signup"
        ]
        url_lc = url.lower()
        return not any(pattern in url_lc for pattern in banned_patterns)

    def _extract_text(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, "lxml")
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = ' '.join(soup.stripped_strings)
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    async def _worker(self, name: str, session: aiohttp.ClientSession):
        while True:
            try:
                url = await self.queue.get()
                logger.info(f"{name}: Processing {url} (Queue: {self.queue.qsize()}, Results: {len(self.results)})")

                if len(self.results2) >= self.max_pages:
                    logger.info(f"{name}: Max pages ({self.max_pages}) reached. Skipping {url}.")
                    self.queue.task_done()
                    continue

                headers = {'User-Agent': 'ProfessionalCrawler/1.0'}
                try:
                    async with session.get(url, timeout=self.timeout, headers=headers, ssl=False) as response:
                        status = response.status
                        content_type = response.headers.get('Content-Type', '').lower()
                        try:
                            content = await response.text(errors="replace")
                        except Exception:
                            content = ""
                        # Extract text if HTML, else store empty string
                        if status == 200 and "html" in content_type:
                            text = self._extract_text(content)
                            self.results[url] = text
                            self.results2[url] = text
                            soup = BeautifulSoup(content, "lxml")
                            links_found = 0
                            for link in soup.find_all("a", href=True):
                                if links_found >= self.max_links_per_page:
                                    break
                                raw_href = link["href"]
                                abs_url = urljoin(url, raw_href)
                                norm_url = self._normalize_url(abs_url)
                                if (
                                    norm_url not in self.visited_urls
                                    and self._is_within_scope(norm_url)
                                    and self._should_enqueue(norm_url)
                                    and self.queue.qsize() < self.queue_maxsize
                                ):
                                    await self.queue.put(norm_url)
                                    self.visited_urls.add(norm_url)
                                    links_found += 1
                            logger.debug(f"{name}: Enqueued {links_found} links from {url}")
                        else:
                            self.results[url] = ""  # Not HTML or not 200
                            logger.debug(f"{name}: Skipped link extraction for {url} (status={status}, content_type={content_type})")
                except Exception as e:
                    logger.warning(f"{name}: Error fetching {url}: {e}")
                    self.results[url] = ""
                finally:
                    self.queue.task_done()
                    await asyncio.sleep(self.request_delay)
            except asyncio.CancelledError:
                logger.info(f"{name}: Worker cancelled.")
                break
            except Exception as e:
                logger.error(f"{name}: Worker error: {e}", exc_info=True)
                if not self.queue.empty():
                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass
                break

    async def fetch_domain_content(self, start_url: str, max_pages: int) -> Dict[str, str]:
        self.max_pages = max_pages
        self.max_concurrent_requests: int = max(40, int(min(max_pages/80, 150)))
        parsed = urlparse(start_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid start URL")
        self.base_domain = parsed.netloc.lower()
        self.base_scheme = parsed.scheme.lower()
        if self.allowed_domains is None:
            self.allowed_domains = {self.base_domain}
        norm_start_url = self._normalize_url(start_url)
        await self.queue.put(norm_start_url)
        self.visited_urls.add(norm_start_url)

        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(self._worker(f"Worker-{i+1}", session))
                for i in range(self.max_concurrent_requests)
            ]
            await self.queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        logger.info(f"Crawling finished. Fetched {len(self.results)} pages.")
        return self.results2

class FileHandlerFactory:
    def __init__(self):
        self.handlers = [
            TxtHandler(),
            MarkdownHandler(),
            PdfHandler(),
            DocxHandler(),
            HtmlHandler(),
            WebPageHandler()
        ]
        self.domainHandler = HtmlDomainHandler()
    def get_handler(self, file_path: str) -> bytes:
            
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return handler.read_content(file_path)
        raise ValueError(f"Not supported file type")
    
    def is_supported(self, file_path: str) -> bool:
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return True
        return False
    
    def get_domain(self, url: str, max_pages: str) -> dict[str, bytes]:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                return asyncio.create_task(self.domainHandler.fetch_domain_content(url, max_pages))
            else:
                return asyncio.run(self.domainHandler.fetch_domain_content(url, max_pages))
