import requests
import os
import re
from bs4 import BeautifulSoup
import urllib.parse
import time

class ScraperWiki():

    def __init__(self):
        lang_env = os.getenv('LANG')
        self.sysLang = lang_env.split('_')[0] if lang_env else 'en'
        
        # Headers per sembrare un browser reale
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def clean_topic(self, phrase):
        """Pulisce il topic usando regex"""
        # Rimuove punteggiatura e divide in parole
        words = re.findall(r'\b[a-zA-Z]+\b', phrase.lower())
        
        # Lista di parole comuni da ignorare
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Trova la prima parola significativa
        for word in words:
            if word not in stop_words and len(word) > 2:
                return word
        
        return words[0] if words else phrase.lower()

    def search_google(self, topic):
        """Cerca su Google e estrae l'overview/snippet"""
        try:
            # Codifica la query per l'URL
            query = urllib.parse.quote_plus(f"{topic} overview")
            
            # URL di ricerca Google
            url = f"https://www.google.com/search?q={query}&hl={self.sysLang}"
            
            # Effettua la richiesta
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Cerca diversi tipi di snippet/overview di Google
            overview_text = self._extract_google_overview(soup)
            
            if overview_text:
                print(f"Google Overview trovata: {overview_text[:100]}...")
                return overview_text
            
            return None
            
        except Exception as e:
            print(f"Errore nella ricerca Google: {e}")
            return None

    def _extract_google_overview(self, soup):
        """Estrae l'overview da diversi elementi di Google con più testo"""
        
        # 1. Cerca nel Knowledge Panel (pannello laterale) - raccoglie più paragrafi
        knowledge_panel = soup.find('div', class_='kp-blk')
        if knowledge_panel:
            # Cerca descrizione principale
            desc_main = knowledge_panel.find('span', class_='kno-rdesc')
            # Cerca anche paragrafi aggiuntivi
            desc_additional = knowledge_panel.find_all('div', class_=['kno-rdesc', 'Uo8X3b'])
            
            combined_text = ""
            if desc_main:
                combined_text += desc_main.get_text(strip=True) + " "
            
            for desc in desc_additional:
                text = desc.get_text(strip=True)
                if text and text not in combined_text:
                    combined_text += text + " "
            
            if combined_text and len(combined_text) > 50:
                return combined_text.strip()
        
        # 2. Cerca negli snippet in evidenza (featured snippets) - più classi
        featured_snippets = soup.find_all('div', class_=['hgKElc', 'kno-rdesc', 'Uo8X3b', 'yp', 'osl'])
        for snippet in featured_snippets:
            text = snippet.get_text(strip=True)
            if text and len(text) > 100:  # Aumentato il minimo per avere più contenuto
                return text
        
        # 3. Combina più risultati di ricerca per avere più testo
        search_results = soup.find_all('div', class_=['VwiC3b', 'aCOpRe', 'yDYNvb', 'IsZvec'])
        combined_results = ""
        
        for result in search_results[:3]:  # Prendi i primi 3 risultati
            text = result.get_text(strip=True)
            if text and len(text) > 50:
                combined_results += text + " "
                if len(combined_results) > 300:  # Limita la lunghezza totale
                    break
        
        if combined_results:
            return combined_results.strip()
        
        # 4. Cerca nei risultati con classe generica - più testo
        generic_results = soup.find_all('span', class_=['st', 'aCOpRe'])
        combined_generic = ""
        
        for result in generic_results[:2]:  # Prendi i primi 2
            text = result.get_text(strip=True)
            if text and len(text) > 50:
                combined_generic += text + " "
                if len(combined_generic) > 400:
                    break
        
        if combined_generic:
            return combined_generic.strip()
        
        # 5. Fallback: cerca qualsiasi div con testo lungo - aumentato il limite
        all_divs = soup.find_all('div')
        for div in all_divs:
            if div.get_text(strip=True):
                text = div.get_text(strip=True)
                # Filtra testi troppi corti o che sembrano menu/navigazione
                if (100 < len(text) < 800 and  # Aumentato il range
                    not any(word in text.lower() for word in ['cookie', 'privacy', 'terms', 'menu', 'search', 'navigation'])):
                    return text
        
        return None

    def search_wikipedia_fallback(self, topic):
        """Fallback su Wikipedia se Google non funziona"""
        try:
            keyword = self.clean_topic(topic)
            keyword_url = keyword.replace(" ", "_")
            url = f"https://{self.sysLang}.wikipedia.org/wiki/{keyword_url}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', class_='mw-parser-output')
            if not content:
                return None

            paragraphs = content.find_all('p', recursive=False)
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    return text
            
            return None
            
        except Exception as e:
            print(f"Errore Wikipedia fallback: {e}")
            return None

    def search(self, topic, fallback_lang='en'):
        """Cerca informazioni prima su Google, poi su Wikipedia come fallback"""
        try:
            # Prova prima con Google
            google_result = self.search_google(topic)
            if google_result:
                return google_result
            
            # Se Google non funziona, prova Wikipedia
            print("Google non ha restituito risultati, provo Wikipedia...")
            wiki_result = self.search_wikipedia_fallback(topic)
            if wiki_result:
                return wiki_result
            
            # Se anche Wikipedia fallisce, prova con una ricerca più generica
            print("Provo ricerca più generica...")
            generic_result = self.search_google(f"what is {topic}")
            if generic_result:
                return generic_result
            
            return f"Nessuna informazione trovata per: {topic}"
            
        except Exception as e:
            return f"Errore durante la ricerca: {str(e)}"