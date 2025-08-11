import os
import re
import json
import chromadb
import pdfplumber
import PyPDF2
import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from dotenv import load_dotenv
class DocumentProcessor:
    def __init__(self, file_path: str = None):
        # Si aucun fichier n'est spécifié, demander à l'utilisateur
        if file_path is None:
            file_path = select_pdf_file()
            if not file_path:
                raise ValueError("Aucun fichier sélectionné. Veuillez sélectionner un fichier PDF valide.")
        
        # Vérifier que le fichier existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        
        # Vérifier que c'est bien un PDF
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Le fichier sélectionné n'est pas un PDF.")
        
        self.file_path = file_path
        print(f"Fichier sélectionné: {os.path.basename(file_path)}")
        
        try:
            self.document = self._load_document()
            self.metadata = self._extract_metadata()
            print(f"Document chargé avec succès ({self.metadata['pdf_pages']} pages)")
        except Exception as e:
            print(f"Erreur lors du traitement du fichier: {e}")
            raise

    def _load_document(self) -> str:
        """Extrait et nettoie le texte du PDF avec pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                print(f"Extraction du texte avec pdfplumber...")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if page_text:
                        text += page_text + "\n"
                    print(f"Page {i+1}/{len(pdf.pages)} traitée", end='\r')
                print()  # Nouvelle ligne après la progression
        except Exception as e:
            print(f"Erreur avec pdfplumber: {e}. Utilisation de PyPDF2...")
            text = self._fallback_pypdf2()
        
        # Nettoyage juridique avancé
        text = self._clean_text(text)
        return text
    
    def _fallback_pypdf2(self) -> str:
        """Alternative avec PyPDF2 si pdfplumber échoue"""
        text = ""
        try:
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                print(f"Extraction du texte avec PyPDF2...")
                for i, page in enumerate(reader.pages):
                    text += page.extract_text() + "\n"
                    print(f"Page {i+1}/{len(reader.pages)} traitée", end='\r')
                print()  # Nouvelle ligne après la progression
        except Exception as e:
            print(f"Erreur avec PyPDF2: {e}")
            raise
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Nettoyage spécifique aux documents juridiques"""
        # Suppression des en-têtes/pieds de page
        text = re.sub(r'Page \d+/\d+', '', text)
        text = re.sub(r'Code des Douanes - \d+', '', text)
        
        # Normalisation des références
        text = re.sub(r'Art\.\s*', 'Art. ', text)
        text = re.sub(r'(\d+)\s*°', r'\1°', text)
        text = re.sub(r'CHAPITRE\s+([IVX]+)', r'CHAPITRE \1', text)
        text = re.sub(r'TITRE\s+([IVX]+)', r'TITRE \1', text)
        
        # Correction des sauts de ligne intempestifs
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
        text = re.sub(r'([.,;:?])\n([A-Z])', r'\1 \2', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_metadata(self) -> Dict:
        """Extrait les métadonnées globales"""
        return {
            "document": "Code des Douanes Malgache",
            "version": "LFI 2025",
            "language": "fr",
            "total_articles": len(re.findall(r'Art\.\s*\d+', self.document)),
            "pdf_pages": self._get_pdf_page_count()
        }
    
    def _get_pdf_page_count(self) -> int:
        """Compte le nombre de pages du PDF"""
        try:
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except:
            return 0
    
    def extract_articles(self) -> List[Dict]:
        """Extrait les articles avec métadonnées enrichies"""
        pattern = r'(Art\.\s*(?:\d+[a-z]*(?:\s*bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies)?)(?:\s*\([a-z]\))?)\s*(.*?)(?=Art\.|\Z)'
        matches = re.finditer(pattern, self.document, re.DOTALL)
        
        articles = []
        for match in matches:
            article_num = match.group(1).strip()
            content = match.group(2).strip()
            
            section = self._extract_section(content)
            lfi_modif = self._detect_modification(content)
            keywords = self._extract_keywords(content)
            related = self._find_related_articles(article_num, content)
            
            articles.append({
                "article": article_num,
                "section": section,
                "content": content,
                "modification": lfi_modif,
                "keywords": keywords,
                "related_articles": related,
                "word_count": len(content.split())
            })
        
        print(f"{len(articles)} articles extraits")
        return articles
    
    def _extract_section(self, content: str) -> str:
        """Extrait la section/chapitre avec contexte"""
        section_match = re.search(
            r'(TITRE\s+[IVX]+.*?|CHAPITRE\s+[IVX]+.*?|SECTION\s+\d+.*?)\n',
            content, re.IGNORECASE
        )
        return section_match.group(1).strip() if section_match else "Non spécifié"
    
    def _detect_modification(self, content: str) -> Optional[str]:
        """Détecte les modifications récentes"""
        if "LFI 2025" in content:
            return "LFI 2025"
        elif "LFI 2024" in content:
            return "LFI 2024"
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraction améliorée des termes juridiques"""
        legal_terms = [
            "douane", "tarif", "droit", "taxe", "importation", "exportation", 
            "contentieux", "sanction", "régime économique", "procédure",
            "dédouanement", "transit", "entrepôt", "perfectionnement",
            "acquit-à-caution", "drawback", "OEA", "agrément",
            "valeur en douane", "origine", "contrôle", "infraction"
        ]
        
        found_terms = []
        for term in legal_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                found_terms.append(term)
        
        numbers = re.findall(r'\b\d+\s*%\b', text)
        found_terms.extend(numbers)
        
        return found_terms
    
    def _find_related_articles(self, current_article: str, content: str) -> List[str]:
        """Trouve les références à d'autres articles avec contexte"""
        references = re.findall(
            r'Art\.\s*(\d+[a-z]*(?:\s*bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies)?)',
            content, re.IGNORECASE
        )
        return list(set(ref for ref in references if ref != current_article))

        