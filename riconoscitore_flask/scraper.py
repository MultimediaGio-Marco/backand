import requests
import os
from bs4 import BeautifulSoup
import urllib.parse
import re

class ScraperWiki:
    def __init__(self):
        lang_env = os.getenv('LANG')
        self.sysLang = lang_env.split('_')[0] if lang_env else 'en'

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }

        # Stopwords di base in inglese
        self.stopwords = {
            'a', 'an', 'the', 'this', 'that', 'in', 'on', 'with', 'and', 'of', 'for', 'at', 'by',
            'to', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'it'
        }

    def extract_keywords(self, text, max_keywords=3):
        """Estrae parole chiave ignorando stopwords, ma mantiene input significativi"""
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in self.stopwords]
        
        # Fallback: se poche parole utili, usa il testo originale
        if len(keywords) < 2:
            return [text.strip().lower()]

        return keywords[:max_keywords]

    def search_google(self, topic):
        """Cerca su Google e estrae l'overview/snippet basato su keyword."""
        try:
            keywords = self.extract_keywords(topic)
            query_str = " ".join(keywords) if keywords else topic
            print(f"Searching Google for: {query_str}")
            query = urllib.parse.quote_plus(f"{query_str} overview")
            url = f"https://www.google.com/search?q={query}&hl={self.sysLang}"

            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            overview = self._extract_google_overview(soup)
            if overview:
                print(f"Google overview: {overview[:100]}...")
                return overview
            return None
        except requests.HTTPError as e:
            print(f"HTTP error during Google search: {e}")
            return None
        except Exception as e:
            print(f"Error in search_google: {e}")
            return None

    def _extract_google_overview(self, soup):
        # Knowledge Panel
        kp = soup.find('div', class_='kp-blk')
        if kp:
            descs = kp.find_all(['span', 'div'], class_=['kno-rdesc', 'Uo8X3b'])
            text = " ".join(d.get_text(strip=True) for d in descs)
            if len(text) > 50:
                return text

        # Featured snippets
        for cls in ['hgKElc', 'kno-rdesc', 'Uo8X3b', 'yp', 'osl']:
            snip = soup.find('div', class_=cls)
            if snip and len(snip.get_text(strip=True)) > 100:
                return snip.get_text(strip=True)

        # Risultati principali
        results = soup.find_all('div', class_=['VwiC3b', 'aCOpRe', 'yDYNvb', 'IsZvec'], limit=3)
        text = " ".join(r.get_text(strip=True) for r in results if len(r.get_text(strip=True)) > 50)
        if len(text) > 50:
            return text[:400]
        return None

    def search_wikipedia_fallback(self, topic, fallback_lang=None):
        """Fallback su Wikipedia se Google non funziona"""
        try:
            lang = fallback_lang or self.sysLang
            keywords = self.extract_keywords(topic)
            page = keywords[0] if keywords else topic
            title = page.replace(' ', '_')
            print(f"Searching Wikipedia for: {title} in {lang}")
            url = f"https://{lang}.wikipedia.org/wiki/{title}"

            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text, 'html.parser')
            p_tags = soup.select('div.mw-parser-output > p')
            for p in p_tags:
                t = p.get_text(strip=True)
                if len(t) > 50:
                    return t
            return None
        except Exception as e:
            print(f"Error in search_wikipedia_fallback: {e}")
            return None

    def search(self, topic, fallback_lang='en'):
        """Cerca informazioni prima su Google, poi su Wikipedia."""
        result = self.search_google(topic)
        if result:
            return result
        print("Google failed, trying Wikipedia...")
        result = self.search_wikipedia_fallback(topic, fallback_lang)
        if result:
            return result
        print("Both failed, retrying generic Google...")
        return self.search_google(f"what is {topic}") or f"No info found for: {topic}"
