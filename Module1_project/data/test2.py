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

warnings.filterwarnings('ignore')

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Import Groq
from groq import Groq

# Fonction pour sélectionner un fichier via une boîte de dialogue
def select_pdf_file():
    """Ouvre une boîte de dialogue pour sélectionner un fichier PDF"""
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale
    file_path = filedialog.askopenfilename(
        title="Sélectionner le PDF du Code des Douanes",
        filetypes=[("Fichiers PDF", "*.pdf"), ("Tous les fichiers", "*.*")]
    )
    root.destroy()
    return file_path

# Fonction pour nettoyer les métadonnées pour ChromaDB
def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Nettoie les métadonnées pour les rendre compatibles avec ChromaDB"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""  # Remplacer None par chaîne vide
        elif isinstance(value, list):
            # Convertir les listes en chaînes séparées par des virgules
            cleaned[key] = ", ".join(str(item) for item in value if item is not None)
        elif isinstance(value, (str, int, float, bool)):
            # Types déjà supportés par ChromaDB
            cleaned[key] = value
        else:
            # Convertir les autres types en chaîne
            cleaned[key] = str(value)
    return cleaned

# 1. PRÉTRAITEMENT AVANCÉ DU DOCUMENT PDF
class LegalDocumentProcessor:
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

# 2. CHUNKING HIÉRARCHIQUE ET THÉMATIQUE
class LegalChunker:
    def __init__(self, articles: List[Dict]):
        self.articles = articles
        self.chunks = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une'])
        
    def create_chunks(self) -> List[Dict]:
        """Crée des chunks optimisés avec contexte thématique"""
        print("Création des chunks...")
        for i, article in enumerate(self.articles):
            chunk = self._create_article_chunk(article)
            self.chunks.append(chunk)
            print(f"Article {i+1}/{len(self.articles)} traité", end='\r')
        print()  # Nouvelle ligne après la progression
        
        self._create_thematic_chunks()
        self._create_summary_chunks()
        
        print(f"{len(self.chunks)} chunks créés")
        return self.chunks
    
    def _create_article_chunk(self, article: Dict) -> Dict:
        """Crée un chunk pour un article individuel"""
        if article['word_count'] > 300:
            sub_chunks = self._split_long_article(article)
            return sub_chunks[0]
        
        return {
            "text": f"{article['article']} - {article['content']}",
            "metadata": clean_metadata({
                "article": article['article'],
                "section": article['section'],
                "modification": article['modification'],
                "keywords": article['keywords'],
                "related_articles": article['related_articles'],
                "theme": self._identify_theme(article['content']),
                "type": "article",
                "word_count": article['word_count']
            })
        }
    
    def _split_long_article(self, article: Dict) -> List[Dict]:
        """Divise les articles longs en sous-chunks cohérents"""
        content = article['content']
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            if current_words + para_words > 250 and current_chunk:
                chunks.append(self._create_sub_chunk(article, current_chunk, len(chunks)+1))
                current_chunk = para
                current_words = para_words
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_words += para_words
        
        if current_chunk:
            chunks.append(self._create_sub_chunk(article, current_chunk, len(chunks)+1))
        
        return chunks
    
    def _create_sub_chunk(self, article: Dict, content: str, part_num: int) -> Dict:
        """Crée un sous-chunk avec métadonnées appropriées"""
        return {
            "text": f"{article['article']} (Partie {part_num}) - {content}",
            "metadata": clean_metadata({
                "article": article['article'],
                "section": article['section'],
                "modification": article['modification'],
                "keywords": article['keywords'],
                "related_articles": article['related_articles'],
                "theme": self._identify_theme(content),
                "type": "sub_article",
                "part": part_num
            })
        }
    
    def _identify_theme(self, content: str) -> str:
        """Identifie le thème principal avec scoring"""
        theme_keywords = {
            "Principes généraux": ["principe", "général", "définition", "champ d'application", "objet"],
            "Tarifs douaniers": ["tarif", "droit de douane", "taxe", "valeur en douane", "taux"],
            "Régimes économiques": ["régime économique", "entrepôt", "perfectionnement", "admission temporaire", "exportation temporaire"],
            "Procédures douanières": ["procédure", "dédouanement", "déclaration", "contrôle", "vérification"],
            "Contentieux et sanctions": ["contentieux", "sanction", "infraction", "pénalités", "transaction"],
            "Taxes diverses": ["taxe", "fiscalité", "redevance", "contribution", "assiette"]
        }
        
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in content.lower())
            theme_scores[theme] = score
        
        best_theme = max(theme_scores, key=theme_scores.get)
        return best_theme if theme_scores[best_theme] > 0 else "Autre"
    
    def _create_thematic_chunks(self):
        """Crée des chunks thématiques en regroupant les articles connexes"""
        thematic_groups = {
            "Opérateur Économique Agréé": [13, 14, 15, 16],
            "Procédures contentieuses": [266, 267, 268, 269, 270],
            "Régimes de l'entrepôt": [132, 133, 134, 135, 136, 137, 138],
            "Droits et taxes": [8, 9, 10, 11, 14, 15, 257, 258, 259, 260, 261, 262, 263, 264, 265]
        }
        
        for theme, article_numbers in thematic_groups.items():
            related_articles = []
            for article in self.articles:
                num_match = re.search(r'(\d+)', article['article'])
                if num_match and int(num_match.group(1)) in article_numbers:
                    related_articles.append(article)
            
            if related_articles:
                combined_content = "\n\n".join([f"{a['article']} - {a['content']}" for a in related_articles])
                chunk = {
                    "text": f"THÈME: {theme}\n{combined_content}",
                    "metadata": clean_metadata({
                        "theme": theme,
                        "type": "thematic_block",
                        "articles": [a['article'] for a in related_articles],
                        "keywords": list(set([kw for a in related_articles for kw in a['keywords']]))
                    })
                }
                self.chunks.append(chunk)
    
    def _create_summary_chunks(self):
        """Crée des chunks de résumé pour les sections importantes"""
        sections = {}
        for article in self.articles:
            section = article['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(article)
        
        for section, articles in sections.items():
            if len(articles) > 5:
                summary = f"SECTION: {section}\n"
                summary += "Articles inclus: " + ", ".join([a['article'] for a in articles]) + "\n"
                summary += "Mots-clés principaux: " + ", ".join(list(set([kw for a in articles for kw in a['keywords'][:5]])))
                
                chunk = {
                    "text": summary,
                    "metadata": clean_metadata({
                        "section": section,
                        "type": "section_summary",
                        "articles": [a['article'] for a in articles],
                        "article_count": len(articles)
                    })
                }
                self.chunks.append(chunk)

# 3. GESTION DES EMBEDDINGS AVEC GESTION D'ERREURS
class EmbeddingManager:
    def __init__(self, model_name: str = None):
        # Liste de modèles compatibles à essayer
        self.models_to_try = [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/distiluse-base-multilingual-cased-v1',
            'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
        ]
        
        if model_name and model_name not in self.models_to_try:
            self.models_to_try.insert(0, model_name)
        
        self.model = None
        self._load_model()
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _load_model(self):
        """Essaye de charger un modèle d'embedding compatible"""
        for model_name in self.models_to_try:
            try:
                print(f"Tentative de chargement du modèle: {model_name}")
                self.model = SentenceTransformer(model_name)
                print(f"Modèle chargé avec succès: {model_name}")
                return
            except Exception as e:
                print(f"Échec du chargement du modèle {model_name}: {e}")
                continue
        
        # Si aucun modèle n'a pu être chargé, utiliser une approche alternative
        print("Aucun modèle n'a pu être chargé. Utilisation d'une alternative basée sur TF-IDF.")
        self.model = None
    
    def generate_embeddings(self, chunks: List[Dict]):
        """Génère et stocke les embeddings dans ChromaDB"""
        print("Génération des embeddings...")
        
        if self.model is not None:
            # Utiliser Sentence Transformers
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.model.encode(texts, convert_to_tensor=True).cpu().numpy()
        else:
            # Utiliser TF-IDF comme alternative
            texts = [chunk['text'] for chunk in chunks]
            vectorizer = TfidfVectorizer(max_features=384)  # Correspond à la taille des embeddings standards
            embeddings = vectorizer.fit_transform(texts).toarray()
            # Sauvegarder le vectorizer pour la recherche
            self.vectorizer = vectorizer
        
        # Préparation des métadonnées avec nettoyage
        metadatas = [clean_metadata(chunk['metadata']) for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Embeddings générés pour {len(chunks)} chunks")
    
    def search(self, query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Recherche hybride (sémantique + métadonnées)"""
        if self.model is not None:
            # Utiliser Sentence Transformers
            query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filters
            )
        else:
            # Utiliser TF-IDF comme alternative
            query_vector = self.vectorizer.transform([query]).toarray()
            all_embeddings = self.collection.get(include=["embeddings", "documents", "metadatas"])
            
            # Calculer la similarité cosinus
            similarities = cosine_similarity(query_vector, np.array(all_embeddings['embeddings']))[0]
            
            # Trier par similarité
            sorted_indices = np.argsort(similarities)[::-1][:n_results]
            
            # Formater les résultats
            results = {
                'documents': [[all_embeddings['documents'][i] for i in sorted_indices]],
                'metadatas': [[all_embeddings['metadatas'][i] for i in sorted_indices]],
                'distances': [[1 - similarities[i] for i in sorted_indices]]  # Convertir similarité en distance
            }
        
        # Formatage des résultats
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return formatted_results

# 4. SYSTÈME RAG COMPLET AVEC GROQ
class LegalRAGSystem:
    def __init__(self, document_path: str = None):
        # Initialisation de Groq avec la clé depuis les variables d'environnement
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Pipeline de traitement
        self.processor = LegalDocumentProcessor(document_path)
        self.articles = self.processor.extract_articles()
        
        self.chunker = LegalChunker(self.articles)
        self.chunks = self.chunker.create_chunks()
        
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.generate_embeddings(self.chunks)
        
        # Essayer de charger un cross-encoder pour le reranking
        self.cross_encoder = None
        self._load_cross_encoder()
        
        print(f"Système initialisé avec {len(self.chunks)} chunks optimisés")
        print(f"Document traité: {self.processor.metadata['document']} ({self.processor.metadata['pdf_pages']} pages)")
    
    def _load_cross_encoder(self):
        """Essaye de charger un cross-encoder pour le reranking"""
        try:
            print("Tentative de chargement du cross-encoder...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("Cross-encoder chargé avec succès")
        except Exception as e:
            print(f"Impossible de charger le cross-encoder: {e}")
            self.cross_encoder = None
    
    def query(self, question: str, filters: Dict = None, n_results: int = 5) -> Dict:
        """Interroge le système RAG avec génération via Groq"""
        # Recherche initiale
        results = self.embedding_manager.search(question, n_results, filters)
        
        # Expansion contextuelle
        expanded_results = self._context_expansion(results)
        
        # Reranking avec cross-encoder si disponible
        reranked_results = self._rerank_results(question, expanded_results)
        
        # Génération de réponse avec Groq
        response = self._generate_response_with_groq(question, reranked_results)
        
        return {
            "question": question,
            "context": reranked_results,
            "answer": response,
            "filters_used": filters
        }
    
    def _context_expansion(self, results: List[Dict]) -> List[Dict]:
        """Ajoute le contexte des articles liés"""
        expanded = []
        for result in results:
            expanded.append(result)
            
            # Récupérer les articles liés depuis les métadonnées
            related_articles_str = result['metadata'].get('related_articles', '')
            if related_articles_str:
                related_articles = [art.strip() for art in related_articles_str.split(',') if art.strip()]
                for related in related_articles[:2]:  # Limiter à 2 articles liés
                    related_results = self.embedding_manager.search(
                        f"Article {related}", 
                        n_results=1,
                        filters={"article": related}
                    )
                    if related_results:
                        expanded.append(related_results[0])
        
        return expanded
    
    def _rerank_results(self, question: str, results: List[Dict]) -> List[Dict]:
        """Réorganise les résultats avec un cross-encoder"""
        if self.cross_encoder is None:
            print("Cross-encoder non disponible, utilisation de l'ordre original")
            return results[:5]
        
        try:
            # Préparer les paires (question, document)
            pairs = [(question, result['text']) for result in results]
            
            # Utiliser le cross-encoder pour prédire les scores
            scores = self.cross_encoder.predict(pairs)
            
            # Trier les résultats par score décroissant
            ranked_results = [result for _, result in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
            
            return ranked_results[:5]
        except Exception as e:
            print(f"Erreur lors du reranking: {e}. Utilisation de l'ordre original.")
            return results[:5]
    
    def _generate_response_with_groq(self, question: str, context: List[Dict]) -> str:
        """Génère une réponse en utilisant Groq"""
        # Préparation du contexte pour Groq
        context_text = "\n\n".join([f"Document {i+1}:\n{result['text']}" for i, result in enumerate(context)])
        
        # Création du prompt
        prompt = f"""Tu es un expert en droit douanier malgache. Réponds à la question suivante en te basant exclusivement sur les documents fournis.

Documents de référence:
{context_text}

Question: {question}

Instructions:
1. Réponds de manière précise et complète
2. Cite les articles pertinents (ex: "Art. 13 quinquies")
3. Mentionne les sections et chapitres concernés
4. Si les documents ne contiennent pas l'information, indique-le clairement
5. Utilise un langage juridique clair et professionnel

Réponse:"""

        try:
            # Appel à l'API Groq
            completion = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",  # Modèle recommandé pour le français
                messages=[
                    {"role": "system", "content": "Tu es un expert en droit douanier malgache."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Faible température pour des réponses précises
                max_tokens=1024
            )
            
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Erreur lors de la génération avec Groq: {e}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse. Veuillez réessayer."

# 5. EXEMPLE D'UTILISATION
if __name__ == "__main__":
    try:
        # Initialisation du système sans spécifier de fichier (sélection via boîte de dialogue)
        rag_system = LegalRAGSystem()
        
        # Exemples de requêtes
        queries = [
            # "Quelles sont les conditions pour obtenir le statut d'OEA?",
            # "Comment la clause transitoire a-t-elle évolué avec la LFI 2025?",
            # "Quelles sont les procédures de contentieux douanier?",
            # "Expliquez le régime de l'entrepôt douanier",
            # "Quels sont les droits de douane applicables aux importations?"
            "Comment exporter des voitures"
        ]
        
        # Exécution des requêtes
        for query in queries:
            print(f"\n{'='*50}\nRequête: {query}\n{'='*50}")
            
            # Recherche avec filtres thématiques optionnels
            if "OEA" in query:
                filters = {"theme": "Opérateur Économique Agréé"}
            elif "contentieux" in query:
                filters = {"theme": "Contentieux et sanctions"}
            elif "entrepôt" in query:
                filters = {"theme": "Régimes économiques"}
            else:
                filters = None
            
            results = rag_system.query(query, filters)
            
            # Affichage des résultats
            print("\nRéponse générée:")
            print(results['answer'])
            
            print("\nContexte utilisé:")
            for i, context in enumerate(results['context'][:3], 1):
                print(f"\nContexte {i}:")
                print(f"Article: {context['metadata'].get('article', 'N/A')}")
                print(f"Thème: {context['metadata'].get('theme', 'N/A')}")
                print(f"Texte: {context['text'][:200]}...")
                
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        print("Vérifiez que le fichier PDF existe et est accessible.")
        print("Assurez-vous d'avoir configuré correctement votre clé API Groq dans le fichier .env.")