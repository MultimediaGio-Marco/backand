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
            'Accept-Language': f'{self.sysLang},{self.sysLang}-q=0.8,en-US;q=0.6,en;q=0.4',
            'Connection': 'keep-alive'
        }

        self.stopwords = {
            'a', 'an', 'the', 'this', 'that', 'in', 'on', 'with', 'and', 'of', 'for', 'at', 'by',
            'to', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'it'
        }

    def extract_keywords(self, text, max_keywords=3):
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in self.stopwords]
        return keywords[:max_keywords] if len(keywords) >= 2 else [text.strip().lower()]

    def search_google(self, topic):
        try:
            keywords = self.extract_keywords(topic)
            query_str = " ".join(keywords) + " overview"
            print(f"🔎 Google search for: {query_str}")
            query = urllib.parse.quote_plus(query_str)
            url = f"https://www.google.com/search?q={query}&hl={self.sysLang}"

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._extract_google_overview(soup)
        except Exception as e:
            print(f"⚠️ Google search error: {e}")
            return None

    def _extract_google_overview(self, soup):
        # Priority: Knowledge Panel → Featured Snippets → Search Results
        selectors = [
            ('div', ['kno-rdesc', 'Uo8X3b']),            # Knowledge Panel
            ('div', ['hgKElc', 'yp', 'osl']),            # Featured Snippets
            ('div', ['VwiC3b', 'aCOpRe', 'yDYNvb']),      # Normal Results
        ]

        for tag, class_list in selectors:
            for cls in class_list:
                for el in soup.find_all(tag, class_=cls):
                    text = el.get_text(strip=True)
                    if len(text) > 80:
                        return text[:400]

        # As fallback, try all paragraphs
        all_p = soup.find_all('p')
        for p in all_p:
            text = p.get_text(strip=True)
            if len(text) > 80:
                return text[:400]
        return None

    def search_wikipedia_fallback(self, topic, fallback_lang=None):
        try:
            lang = fallback_lang or self.sysLang
            keywords = self.extract_keywords(topic)
            title = keywords[0].replace(' ', '_')
            print(f"📚 Wikipedia search for: {title} [{lang}]")
            url = f"https://{lang}.wikipedia.org/wiki/{title}"

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for p in soup.select('div.mw-parser-output > p'):
                text = p.get_text(strip=True)
                if len(text) > 80 and not text.lower().startswith('coordinates'):
                    return text
            return None
        except Exception as e:
            print(f"⚠️ Wikipedia fallback error: {e}")
            return None

    def search(self, topic, fallback_lang='en'):
        # 1. Primary Google search
        result = self.search_google(topic)
        if result:
            return result

        # 2. Try Wikipedia with original topic
        print("🧭 Google failed, trying Wikipedia...")
        result = self.search_wikipedia_fallback(topic, fallback_lang)
        if result:
            return result

        # 3. Try again on Google with more context
        print("🔁 Retrying Google with 'what is ...'")
        result = self.search_google(f"what is {topic}")
        if result:
            return result

        # 4. Final fallback: try Wikipedia with full query
        result = self.search_wikipedia_fallback(f"what is {topic}", fallback_lang)
        if result:
            return result

        # 5. Nothing worked
        return f"No info found for: {topic}"
