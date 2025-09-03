import spacy
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# La configurazione del logging ora verrà gestita dal tuo servizio principale,
# ma possiamo comunque definire un logger specifico per questo modulo.
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        """
        Il costruttore della classe. Carica i modelli NLP una sola volta
        quando un oggetto di questa classe viene creato.
        """
        self.nlp = None
        self.sbert = None
        
        # Caricamento del modello spaCy
        try:
            logger.info("Caricamento del modello spaCy it_core_news_sm...")
            self.nlp = spacy.load("it_core_news_sm")
            logger.info("Modello spaCy caricato con successo")
        except Exception as e:
            logger.error(f"Errore critico nel caricamento del modello spaCy: {e}")
            raise  # Se non si carica spaCy, è un problema serio

        # Caricamento del modello SBERT
        try:
            logger.info("Caricamento del modello SBERT all-MiniLM-L6-v2...")
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Modello SBERT caricato con successo")
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello SBERT: {e}")
            logger.warning("SBERT non disponibile. I vettori di testo useranno il fallback di spaCy.")
            self.sbert = None

    def parse_text(self, text: str) -> dict:
        """
        Esegue l'analisi completa del testo (token, entità, vettore).
        Questa è la logica che prima era nell'endpoint /parse.
        """
        if not self.nlp:
            raise RuntimeError("Modello spaCy non caricato. Impossibile processare il testo.")

        logger.info(f"Elaborazione testo: {text}")
        doc = self.nlp(text)
        
        # Estrazione dei token con lemmi
        tokens = [{"text": t.text, "lemma": t.lemma_} for t in doc]
        
        # Estrazione delle entità
        entities = [(e.text, e.label_) for e in doc.ents]
        
        # Estrazione del vettore del documento
        # Se SBERT è disponibile, usa quello per un vettore migliore, altrimenti fallback su spaCy
        if self.sbert:
            vector = self.sbert.encode(text, normalize_embeddings=True).tolist()
        else:
            vector = doc.vector.tolist()
        
        logger.info(f"Elaborazione completata: {len(tokens)} token, {len(entities)} entità")
        return {
            "tokens": tokens,
            "entities": entities,
            "vector": vector
        }

    def cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        """
        Calcola la similarità coseno. Logica dall'endpoint /cosine_similarity.
        """
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        # Il dot product di due vettori normalizzati è la loro similarità coseno
        similarity = np.dot(vec1, vec2)
        return float(similarity)

    def normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Normalizza un vettore. Logica dall'endpoint /normalize_vector.
        """
        vec = np.array(vector)
        norm = np.linalg.norm(vec)
        if norm > 0:
            normalized = vec / norm
        else:
            normalized = vec # Restituisce il vettore nullo se la norma è 0
        return normalized.tolist()