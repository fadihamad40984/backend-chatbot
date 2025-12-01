"""
Source Fetchers for Free and Open Data Sources
Retrieves information from Wikipedia, arXiv, PubMed, Stack Exchange, OpenLibrary, and OpenStreetMap
"""

import requests
import json
from typing import List, Dict, Optional
import time
from urllib.parse import quote


class WikipediaFetcher:
    """Fetch articles from Wikipedia"""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        # Custom User-Agent to avoid 403 errors
        self.session.headers.update({
            'User-Agent': 'AIBot/1.0 (Educational; fadih@example.com) Python/3.x'
        })
        self.last_request_time = 0
        self.min_delay = 1.0  # Minimum delay between requests in seconds
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Wikipedia and return article summaries"""
        try:
            # Rate limiting - ensure minimum delay between requests
            import time
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srprop": "snippet"
            }
            
            # Retry logic for transient failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(self.base_url, params=params, timeout=10)
                    self.last_request_time = time.time()
                    
                    # Check if response is valid
                    if response.status_code == 403:
                        print(f"Wikipedia 403 Forbidden - API may be blocking requests. Using fallback sources.")
                        return []
                    elif response.status_code != 200:
                        print(f"Wikipedia HTTP error: {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return []
                    
                    try:
                        data = response.json()
                        break  # Success, exit retry loop
                    except Exception as json_err:
                        print(f"Wikipedia JSON error: {json_err}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return []
                except requests.exceptions.RequestException as req_err:
                    print(f"Wikipedia request error (attempt {attempt + 1}/{max_retries}): {req_err}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []
            
            results = []
            for item in data.get("query", {}).get("search", []):
                page_id = item["pageid"]
                content = self.get_page_content(page_id)
                if content:
                    results.append({
                        "title": item["title"],
                        "text": content,
                        "source": f"Wikipedia: {item['title']}",
                        "url": f"https://en.wikipedia.org/?curid={page_id}"
                    })
            return results
        except Exception as e:
            print(f"Wikipedia fetch error: {e}")
            return []
    
    def get_page_content(self, page_id: int) -> Optional[str]:
        """Get full page content for a given page ID"""
        try:
            # Rate limiting
            import time
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            params = {
                "action": "query",
                "format": "json",
                "pageids": page_id,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code != 200:
                print(f"Wikipedia page content HTTP error: {response.status_code}")
                return None
                
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            return pages.get(str(page_id), {}).get("extract", "")
        except Exception as e:
            print(f"Wikipedia page content error: {e}")
            return None


class ArxivFetcher:
    """Fetch scientific papers from arXiv"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search arXiv for papers"""
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit
            }
            response = self.session.get(self.base_url, params=params, timeout=15)
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            results = []
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", namespace):
                title = entry.find("atom:title", namespace)
                summary = entry.find("atom:summary", namespace)
                arxiv_id = entry.find("atom:id", namespace)
                
                if title is not None and summary is not None:
                    results.append({
                        "title": title.text.strip().replace("\n", " "),
                        "text": summary.text.strip().replace("\n", " "),
                        "source": f"arXiv: {arxiv_id.text.split('/')[-1] if arxiv_id is not None else 'N/A'}",
                        "url": arxiv_id.text if arxiv_id is not None else ""
                    })
            
            return results
        except Exception as e:
            print(f"arXiv fetch error: {e}")
            return []


class PubMedFetcher:
    """Fetch medical papers from PubMed Central"""
    
    def __init__(self):
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.session = requests.Session()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search PubMed for medical articles"""
        try:
            # Search for article IDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json"
            }
            search_response = self.session.get(self.search_url, params=search_params, timeout=10)
            search_data = search_response.json()
            
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []
            
            # Fetch article summaries
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }
            fetch_response = self.session.get(self.fetch_url, params=fetch_params, timeout=15)
            
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(fetch_response.content)
            
            results = []
            for article in root.findall(".//PubmedArticle"):
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                pmid_elem = article.find(".//PMID")
                
                if title_elem is not None and abstract_elem is not None:
                    results.append({
                        "title": title_elem.text or "",
                        "text": abstract_elem.text or "",
                        "source": f"PubMed: {pmid_elem.text if pmid_elem is not None else 'N/A'}",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else ""
                    })
            
            return results
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []


class StackExchangeFetcher:
    """Fetch Q&A from Stack Exchange network"""
    
    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3/search"
        self.session = requests.Session()
    
    def search(self, query: str, site: str = "stackoverflow", limit: int = 5) -> List[Dict]:
        """Search Stack Exchange sites"""
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "intitle": query,
                "site": site,
                "pagesize": limit,
                "filter": "withbody"
            }
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                # Get top answer if available
                answer_text = ""
                if item.get("answer_count", 0) > 0 and item.get("accepted_answer_id"):
                    answer_text = self.get_answer(item["accepted_answer_id"], site)
                
                results.append({
                    "title": item.get("title", ""),
                    "text": f"Question: {self._clean_html(item.get('body', ''))}\n\nAnswer: {answer_text}",
                    "source": f"Stack Overflow: {item.get('title', 'Question')}",
                    "url": item.get("link", "")
                })
            
            return results
        except Exception as e:
            print(f"Stack Exchange fetch error: {e}")
            return []
    
    def get_answer(self, answer_id: int, site: str) -> str:
        """Fetch specific answer"""
        try:
            url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
            params = {"site": site, "filter": "withbody"}
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            items = data.get("items", [])
            if items:
                return self._clean_html(items[0].get("body", ""))
            return ""
        except Exception as e:
            return ""
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags"""
        import re
        clean = re.sub('<.*?>', '', html_text)
        return clean.strip()[:500]  # Limit length


class OpenLibraryFetcher:
    """Fetch book information from OpenLibrary"""
    
    def __init__(self):
        self.search_url = "https://openlibrary.org/search.json"
        self.session = requests.Session()
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search OpenLibrary for books"""
        try:
            params = {
                "q": query,
                "limit": limit
            }
            response = self.session.get(self.search_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for doc in data.get("docs", []):
                text = f"Author: {', '.join(doc.get('author_name', ['Unknown']))}\n"
                text += f"Published: {doc.get('first_publish_year', 'N/A')}\n"
                text += f"Description: {doc.get('first_sentence', [''])[0] if doc.get('first_sentence') else 'No description available'}"
                
                results.append({
                    "title": doc.get("title", "Unknown Title"),
                    "text": text,
                    "source": f"OpenLibrary: {doc.get('title', 'Book')}",
                    "url": f"https://openlibrary.org{doc.get('key', '')}"
                })
            
            return results
        except Exception as e:
            print(f"OpenLibrary fetch error: {e}")
            return []


class OpenStreetMapFetcher:
    """Fetch geographic data from OpenStreetMap"""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AI-Chatbot/1.0'})
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search OpenStreetMap for locations"""
        try:
            params = {
                "q": query,
                "format": "json",
                "limit": limit,
                "addressdetails": 1
            }
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for item in data:
                address = item.get("address", {})
                text = f"Location: {item.get('display_name', '')}\n"
                text += f"Type: {item.get('type', 'N/A')}\n"
                text += f"Coordinates: {item.get('lat', '')}, {item.get('lon', '')}"
                
                results.append({
                    "title": item.get("display_name", "Location"),
                    "text": text,
                    "source": "OpenStreetMap",
                    "url": f"https://www.openstreetmap.org/?mlat={item.get('lat')}&mlon={item.get('lon')}"
                })
            
            time.sleep(1)  # Rate limiting for OSM
            return results
        except Exception as e:
            print(f"OpenStreetMap fetch error: {e}")
            return []


class SourceAggregator:
    """Aggregate results from all sources"""
    
    def __init__(self):
        self.wikipedia = WikipediaFetcher()
        self.arxiv = ArxivFetcher()
        self.pubmed = PubMedFetcher()
        self.stack_exchange = StackExchangeFetcher()
        self.open_library = OpenLibraryFetcher()
        self.osm = OpenStreetMapFetcher()
    
    def search_all(self, query: str, sources: List[str] = None) -> List[Dict]:
        """
        Search across multiple sources
        sources: list of source names to search, or None for all
        Available: 'wikipedia', 'arxiv', 'pubmed', 'stackoverflow', 'openlibrary', 'osm'
        """
        if sources is None:
            sources = ['wikipedia', 'stackoverflow']  # Default to most reliable
        
        all_results = []
        
        if 'wikipedia' in sources:
            all_results.extend(self.wikipedia.search(query, limit=3))
        
        if 'arxiv' in sources:
            all_results.extend(self.arxiv.search(query, limit=2))
        
        if 'pubmed' in sources:
            all_results.extend(self.pubmed.search(query, limit=2))
        
        if 'stackoverflow' in sources:
            all_results.extend(self.stack_exchange.search(query, limit=3))
        
        if 'openlibrary' in sources:
            all_results.extend(self.open_library.search(query, limit=2))
        
        if 'osm' in sources:
            all_results.extend(self.osm.search(query, limit=2))
        
        return all_results
