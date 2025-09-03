"""Topic Detection utilities
=========================
Un unico modulo che raggruppa **tutto** ciò che serve alla nuova pipeline:

* arricchimento metadati in fase *offline* (già implementato nella sezione
  `QuestionImporter`);
* soglie letta da **variabili d’ambiente** (`TH_FUZZY`, `TH_COS`,
  `COVERAGE_THRESHOLD_PERCENT`) così l’utente può ritoccarle via `.env` senza
  toccare il codice;
* funzione `detect_covered_topics()` usata a *runtime* da
  `InterviewState.missing_topics`;
* piccola helper `topic_objects_from_meta()` che prende i campi salvati nello
  YAML e costruisce una lista di oggetti `Topic` pronta per la cascata.

L’idea è che **interview_state.py** debba solo:
```python
from topic_detection import topic_objects_from_meta, detect_covered_topics
```
poi, dentro `missing_topics()`:
```python
# 0. init
topics = topic_objects_from_meta(
            self.current_subtopics,
            self.current_keywords,   # lista‑di‑liste, già allineata
            self.current_lemma_sets, # nuovo campo nel DOMANDE
            self.current_fuzzy_norms,
            self.current_vectors,
        )
covered, coverage = detect_covered_topics(last_user_text, topics)
missing  = [t.name for t in topics if t.name not in covered]
return missing, coverage*100
```
Così non cambiamo nomi delle variabili già presenti – aggiungiamo solo i
nuovi array paralleli `lemma_sets`, `fuzzy_norms`, `vectors` nella variabile
`DOMANDE`.
"""

from __future__ import annotations

import json
import os
import re
import logging
import time
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any, Tuple, Set

import numpy as np
import unidecode  # Modifica: importiamo il modulo invece della funzione

# ---------------------------------------------------------------------------
# Caricamento del processore NLP
# ---------------------------------------------------------------------------
# Importiamo la classe dal file che abbiamo creato in precedenza
from Main.services.nlp_services import NLPProcessor

# Creiamo un'istanza globale che caricherà i modelli spaCy e SBERT
processor = NLPProcessor()

from Main.core import config
from openai import OpenAI, AsyncOpenAI
import openai

# ---------------------------------------------------------------------------
# Third‑party libs (import lazy per evitare crash se mancano SBERT / RapidFuzz)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from rapidfuzz.fuzz import token_sort_ratio  # type: ignore
except ImportError:  # pragma: no cover
    token_sort_ratio = None  # type: ignore



SBERT = None
if SentenceTransformer:
    try:
        #SBERT = SentenceTransformer("paraphrase-multilingual-MiniLM-L6-v2")
        SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover
        logging.warning(f"SBERT non caricato, fallback al servizio NLP: {exc}")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soglie da .env (con fallback a default robusti)
# ---------------------------------------------------------------------------
TH_FUZZY: int = int(os.getenv("TH_FUZZY", "90"))  # 0‑100
TH_COS: float = float(os.getenv("TH_COS", "0.75"))
COVERAGE_THRESHOLD_PERCENT: float = float(os.getenv("COVERAGE_THRESHOLD_PERCENT", "80"))

# ---------------------------------------------------------------------------
# Dataclass Topic (runtime)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Topic:
    name: str
    keywords: List[str]
    lemma_set: Set[str]
    fuzzy_norm: str
    vector: np.ndarray  # Ritorniamo a usare np.ndarray

# ---------------------------------------------------------------------------
# Helper build per metadati offline (usato già da QuestionImporter) ---------
# ---------------------------------------------------------------------------

class TopicMetaBuilder:
    """Costruisce *lemma_set*, *fuzzy_norm* e *vector* per un sub‑topic."""

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", unidecode.unidecode(text.lower().strip()))

    @classmethod
    def build(cls, keywords: List[str]) -> Tuple[List[str], str, List[float]]:
        """Restituisce (lemma_set, fuzzy_norm, vector_norm)."""
        lemmas = cls._get_lemmas(keywords)
        norm_string = cls._normalise(" ".join(keywords))
        # Sempre uso il servizio NLP per il vettore (richiede sentence-transformers)
        doc_vec = cls._get_vector(norm_string)
        # Normalizzazione locale con numpy
        vec = cls._normalize_vector(doc_vec)
        return list(lemmas), norm_string, vec.tolist()  # Converto in lista per salvare in JSON
        
    @staticmethod
    @staticmethod
    def _get_lemmas(keywords: List[str]) -> Set[str]:
        """Ottiene i lemmi usando il NLPProcessor locale."""
        try:
            # Usiamo il nostro processore locale
            analysis = processor.parse_text(" ".join(keywords))
            lemmas = {token["lemma"] for token in analysis["tokens"]}
            return lemmas
        except Exception as e:
            logging.error(f"Errore nell'uso di NLPProcessor per lemmi: {e}")
            return set()
    
    @staticmethod
    @staticmethod
    def _get_vector(text: str) -> np.ndarray:
        """Ottiene il vettore usando il NLPProcessor locale."""
        try:
            # Usiamo il nostro processore locale
            analysis = processor.parse_text(text)
            vector = np.array(analysis["vector"])
            return vector
        except Exception as e:
            logging.error(f"Errore nell'uso di NLPProcessor per vettore: {e}")
            # La dimensione del vettore di SBERT è 384
            return np.zeros(384)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalizza un vettore usando numpy localmente."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector  # Ritorna il vettore non normalizzato se la norma è 0

# ---------------------------------------------------------------------------
# Runtime utilities ---------------------------------------------------------
# Alias per retro‑compatibilità con interview_state (topic_from_meta) ---------
# ---------------------------------------------------------------------------

def topic_from_meta(
    subtopics: List[str],
    lemma_sets: List[List[str]],
    fuzzy_norms: List[str],
    vectors: List[List[float]],
    keywords: Optional[List[List[str]]] = None,
) -> List[Topic]:
    """Wrapper retro‑compatibile; forwards to topic_objects_from_meta.

    Se *keywords* è None si assume che queste non servano per la logica di
    coverage ma manteniamo la stessa firma del nuovo helper.
    """
    if keywords is None:
        keywords = [[] for _ in subtopics]
    return topic_objects_from_meta(subtopics, keywords, lemma_sets, fuzzy_norms, vectors)

# ---------------------------------------------------------------------------
# Runtime utilities ---------------------------------------------------------
# ---------------------------------------------------------------------------

def topic_objects_from_meta(
    subtopics: List[str],
    keywords: List[List[str]],
    lemma_sets: List[List[str]],
    fuzzy_norms: List[str],
    vectors: List[List[float]],
) -> List[Topic]:
    """Trasforma liste parallele (come in DOMANDE) in oggetti Topic."""
    topics: List[Topic] = []
    for name, kw, lem, fn, vec in zip(subtopics, keywords, lemma_sets, fuzzy_norms, vectors):
        topics.append(
            Topic(
                name=name,
                keywords=kw,
                lemma_set=set(lem),
                fuzzy_norm=fn,
                vector=np.asarray(vec, dtype=np.float32),  # Converti la lista in np.ndarray
            )
        )
    return topics


def _norm_user(text: str) -> Tuple[str, Set[str]]:
    """Ritorna (text_normalised, lemmi_set)."""
    text_norm = re.sub(r"\s+", " ", unidecode.unidecode(text.lower()))
    lemmi = TopicMetaBuilder._get_lemmas([text_norm])
    return text_norm, lemmi


def detect_covered_topics(user_text: str, topics: List[Topic]) -> Tuple[Set[str], float]:
    """Ritorna (set subtopic coperti, coverage_fraction 0‑1)."""
    if not user_text.strip():  # nessuna risposta
        return set(), 0.0

    txt_norm, user_lemmi = _norm_user(user_text)
    if token_sort_ratio is None:
        raise ImportError("rapidfuzz non installato – richiesto per fuzzy matching")

    # vettore utente solo se servirà il livello 3
    user_vec = None

    remaining = set(t.name for t in topics)
    covered: Set[str] = set()

    # ---- Livello 1: exact lemma --------------------------------------
    for t in topics:
        if t.name in remaining and t.lemma_set.intersection(user_lemmi):
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 2: fuzzy -------------------------------------------
    for t in topics:
        if t.name not in remaining:
            continue
        score = token_sort_ratio(txt_norm, t.fuzzy_norm)
        if score >= TH_FUZZY:
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 3: cosine ------------------------------------------
    if remaining:
        if user_vec is None:
            doc_vec = TopicMetaBuilder._get_vector(txt_norm)
            # Normalizzazione locale con numpy
            user_vec = TopicMetaBuilder._normalize_vector(doc_vec)
        for t in topics:
            if t.name not in remaining:
                continue
            # Calcolo locale della similarità coseno con numpy
            cos = _calculate_cosine_similarity(user_vec, t.vector)
            if cos >= TH_COS:
                covered.add(t.name)
                remaining.remove(t.name)

    coverage = 1 - len(remaining) / len(topics) if topics else 0.0
    return covered, coverage

def adaptive_topic_detection(user_text: str, topics: List[Topic]) -> Tuple[Set[str], float]:
    """Detection con soglie adattive basate su caratteristiche del testo."""
    if not user_text.strip():
        return set(), 0.0

    # Calcola statistiche del testo
    text_length = len(user_text.split())
    topic_count = len(topics)
    
    # Adatta soglie basandosi su lunghezza testo
    if text_length < 10:  # Testo corto - più permissivo
        fuzzy_threshold = 80
        cos_threshold = 0.6
    elif text_length < 30:  # Testo medio
        fuzzy_threshold = 85
        cos_threshold = 0.7
    else:  # Testo lungo - più rigoroso
        fuzzy_threshold = 90
        cos_threshold = 0.75
    
    # Adatta per numero di topic
    if topic_count > 6:  # Molti topic = più specifici
        fuzzy_threshold += 5
        cos_threshold += 0.05
    
    logger.debug(f"Soglie adattive: fuzzy={fuzzy_threshold}, cosine={cos_threshold:.2f} "
                f"(testo: {text_length} parole, topic: {topic_count})")

    # ✅ IMPLEMENTAZIONE DETECTION CON SOGLIE ADATTIVE
    txt_norm, user_lemmi = _norm_user(user_text)
    if token_sort_ratio is None:
        raise ImportError("rapidfuzz non installato – richiesto per fuzzy matching")

    user_vec = None
    remaining = set(t.name for t in topics)
    covered: Set[str] = set()

    # ---- Livello 1: exact lemma (invariato) -------------------------
    for t in topics:
        if t.name in remaining and t.lemma_set.intersection(user_lemmi):
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 2: fuzzy con soglia adattiva -----------------------
    for t in topics:
        if t.name not in remaining:
            continue
        score = token_sort_ratio(txt_norm, t.fuzzy_norm)
        if score >= fuzzy_threshold:  # ✅ Soglia adattiva
            covered.add(t.name)
            remaining.remove(t.name)

    # ---- Livello 3: cosine con soglia adattiva ----------------------
    if remaining:
        if user_vec is None:
            doc_vec = TopicMetaBuilder._get_vector(txt_norm)
            user_vec = TopicMetaBuilder._normalize_vector(doc_vec)
        for t in topics:
            if t.name not in remaining:
                continue
            cos = _calculate_cosine_similarity(user_vec, t.vector)
            if cos >= cos_threshold:  # ✅ Soglia adattiva
                covered.add(t.name)
                remaining.remove(t.name)

    coverage = 1 - len(remaining) / len(topics) if topics else 0.0
    return covered, coverage

def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcola la similarità coseno usando numpy localmente."""
    return float(np.dot(vec1, vec2))

# ---------------------------------------------------------------------------
# Fine modulo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aggiunte di Gabriele
# ---------------------------------------------------------------------------


#def detect_covered_topics_with_gpt(user_text: str, topics: List[Topic], subtopic) -> Tuple[Set[str], float]:
def detect_covered_topics_with_gpt(user_text: str, topics: List[str], subtopic) -> Tuple[[str], float]:

    if not user_text.strip():
        return set(), 0.0

    # Calcola statistiche del testo
    text_length = len(user_text.split())
    topic_count = len(topics)
    
    # Adatta soglie basandosi su lunghezza testo
    if text_length < 10:  # Testo corto - più permissivo
        fuzzy_threshold = 80
        cos_threshold = 0.6
    elif text_length < 30:  # Testo medio
        fuzzy_threshold = 85
        cos_threshold = 0.7
    else:  # Testo lungo - più rigoroso
        fuzzy_threshold = 90
        cos_threshold = 0.75
    
    # Adatta per numero di topic
    if topic_count > 6:  # Molti topic = più specifici
        fuzzy_threshold += 5
        cos_threshold += 0.05
    
    logger.debug(f"Soglie adattive: fuzzy={fuzzy_threshold}, cosine={cos_threshold:.2f} "
                f"(testo: {text_length} parole, topic: {topic_count})")

    # ✅ IMPLEMENTAZIONE DETECTION CON SOGLIE ADATTIVE
    txt_norm, user_lemmi = _norm_user(user_text)
    if token_sort_ratio is None:
        raise ImportError("rapidfuzz non installato – richiesto per fuzzy matching")

    user_vec = None
    remaining = [t.name for t in topics]
    covered = []

    if checkUnknowAnswer(user_text.lower()) or repeatedQuestions(user_text.lower()):
        logger.debug(f"subtopics prima:{remaining}")
        for t in topics:
            if t.name == subtopic:
                covered.append(t.name)
                remaining.remove(t.name)
                break
        logger.debug(f"subtopics dopo:{remaining}")
    elif text_length < 4:
        # ---- Livello 1: exact lemma (invariato) -------------------------
        for t in topics:
            if t.name in remaining: # and t.lemma_set.intersection(user_lemmi):
                covered.append(t.name)
                remaining.remove(t.name)
    else:
        # ---- Livello 2: fuzzy con soglia adattiva -----------------------
        openai.api_key = config.OPENAI_API_KEY
        # Costruzione del prompt
        prompt = f"""
        Dato il seguente testo:

        "{user_text}"

        Dimmi se questo testo riguarda ciascuno dei seguenti topic {", ".join(t.name for t in topics)}. Rispondi solo con "T" o "F" separati da una virgola, nello stesso ordine dei topic.
        
        Non aggiungere nient’altro nella risposta.
        """
        # Non aggiungere nient’altro nella risposta. Nessuna spiegazione. 

        logger.debug("LANCIO DEL PROMPT")
        # Invio del prompt a GPT-4
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # oppure "gpt-3.5-turbo" per un'alternativa più economica
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Imposta a 0 per massima coerenza e zero creatività
        )
        # Estrai la risposta
        output = response.choices[0].message.content
        logger.debug(f"Risposta output: {output}")
        bools = [value.strip() for value in output.split(",")]
        logger.debug(f"Risposta bools: {bools}")
        # Ora abbina ogni valore al corrispondente topic
        for t, boolean in zip(topics, bools):
            if boolean == "T" and t.name == subtopic: #and t in remaining:
                covered.append(t.name)
                remaining.remove(t.name)

    coverage = 1 - len(remaining) / len(topics) if topics else 0.0
    return covered, coverage

def covered_topics_with_gpt(user_text: str, topics: List[str], subtopic) -> Tuple[[str], float]:

    if not user_text.strip():
        return set(), 0.0

    # Calcola statistiche del testo
    text_length = len(user_text.split())

    # ✅ IMPLEMENTAZIONE DETECTION CON SOGLIE ADATTIVE
    txt_norm, user_lemmi = _norm_user(user_text)
    if token_sort_ratio is None:
        raise ImportError("rapidfuzz non installato – richiesto per fuzzy matching")

    remaining = [t for t in topics]
    covered = []

    if checkUnknowAnswer(user_text.lower()) or repeatedQuestions(user_text.lower()):
        logger.debug(f"subtopics prima:{remaining}")
        for t in topics:
            if t == subtopic:
                covered.append(t)
                remaining.remove(t)
                break
        logger.debug(f"subtopics dopo:{remaining}")
    elif text_length < 4:
        # ---- Livello 1: exact lemma (invariato) -------------------------
        for t in topics:
            if t in remaining: # and t.lemma_set.intersection(user_lemmi):
                covered.append(t)
                remaining.remove(t)
    else:
        # ---- Livello 2: fuzzy con soglia adattiva -----------------------
        openai.api_key = config.OPENAI_API_KEY
        # Costruzione del prompt
        prompt = f"""
        Dato il seguente testo:

        "{user_text}"

        Dimmi se questo testo riguarda ciascuno dei seguenti topic {", ".join(t for t in topics)}. Rispondi solo con "T" o "F" separati da una virgola, nello stesso ordine dei topic.
        
        Non aggiungere nient’altro nella risposta.
        """
        # Non aggiungere nient’altro nella risposta. Nessuna spiegazione. 

        logger.debug("LANCIO DEL PROMPT")
        # Invio del prompt a GPT-4
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # oppure "gpt-3.5-turbo" per un'alternativa più economica
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Imposta a 0 per massima coerenza e zero creatività
        )
        # Estrai la risposta
        output = response.choices[0].message.content
        logger.debug(f"Risposta output: {output}")
        bools = [value.strip() for value in output.split(",")]
        logger.debug(f"Risposta bools: {bools}")
        # Ora abbina ogni valore al corrispondente topic
        for t, boolean in zip(topics, bools):
            if boolean == "T" and t == subtopic: #and t in remaining:
                covered.append(t)
                remaining.remove(t)

    coverage = 1 - len(remaining) / len(topics) if topics else 0.0
    return covered, coverage


def checkUnknowAnswer(user_text):
    risposteNonSo = [
        # Neutre
        "non lo so",
        "non so",
        "non ne ho idea",
        "non ho idea",
        "non saprei",
        "non so rispondere",
        "non so che dire",
        "non so la risposta",
        "non conosco la risposta",
        "non ho certezze in merito",
        "non mi risulta",
        "non ho abbastanza dati per rispondere",
        
        # Informali / colloquiali
        "boh",
        "bho",
        "ma che ne so",
        "eh, chi lo sa",
        "non ne ho la più pallida idea",
        "passo",
        "mystery",
        "mi hai beccato in castagna",
        "fosse per me",
        "mi sfugge, sinceramente",
        "mai sentito, davvero",
        
        # Ironiche / divertenti
        "se lo scopro te lo dico",
        "nemmeno nostradamus lo saprebbe",
        "avrei voluto saperlo anch'io",
        "potrei inventare qualcosa, ma non sarebbe giusto",
        "non lo so, ma suona importante",
        "se mi dessero i soldi volentieri",
        "anche google avrebbe difficoltà",
        "un giorno forse lo sapremo",
        "torneremo su questo punto dopo la pubblicità",
        
        # Formali / professionali
        "attualmente non dispongo di queste informazioni",
        "mi riservo di verificare",
        "non sono in grado di fornire una risposta precisa",
        "è fuori dalla mia area di competenza",
        "mi informerò al riguardo",
        "al momento non posso confermare",
        
        # Filosofiche / poetiche
        "il sapere è un mare infinito, e io sono ancora sulla riva",
        "la conoscenza è un viaggio, non una destinazione",
        "a volte non sapere è già una risposta",
        "il dubbio è l’inizio della saggezza",

        "non lo conosco",
        "chi può dirlo",
        "mi cogli impreparato",
        "mi cogli impreparata",
        "è un mistero anche per me",
        "preferisco non sbilanciarmi",
        "dovrei controllare",
        "devo controllare",
        "non sono sicuro",
        "non sono sicura",
        "non ho abbastanza informazioni",
        "bella domanda",
        "me lo stavo chiedendo anch'io",
        "non è il mio campo",
        "forse qualcuno più esperto lo sa",
        "ci devo pensare su",
        "mai sentito prima",
        "potrei sbagliarmi, ma non credo di saperlo",
        "non mi viene in mente",
        "mi sfugge in questo momento",
        "mi sfugge",
        "ma che domande fai? non lo so",
    ]

    logger.debug("controllo checkUnknowAnswer")

    for non_so_anwer in risposteNonSo:
        #logger.debug(f"{non_so_anwer}, {user_text}, {non_so_anwer in user_text}")
        if non_so_anwer in user_text:
            logger.debug("ANSWER UNKNOWN")
            return True
    return False

def repeatedQuestions(user_text):
    frasi_domanda_ripetuta = [
        # Neutri / Cortesi
        "questa domanda l'hai già fatta",
        "questa domanda l'ha già fatta",
        "me lo hai già chiesto",
        "ne abbiamo già parlato",
        "mi pare che tu l'abbia già chiesto",
        "se non sbaglio, l'hai già chiesto",
        "è una domanda ripetuta",
        "l'abbiamo già affrontata",
        "abbiamo già toccato questo punto",
        "me lo ha già chiesto",

        # Gentili / Soft
        "mi sembra di aver risposto a questa",
        "mi sa che ce l'eravamo già chiesti",
        "penso di averti già risposto a riguardo",
        "non vorrei ripetermi, ma l'hai già chiesto",
        "potrebbe essere un déjà vu, ma suona familiare",
        "forse l'hai già chiesto senza volerlo",

        # Ironiche / Passive-aggressive
        "questa mi suona molto familiare",
        "stai facendo copia e incolla per caso",
        "hai problemi di memoria o stai testando la mia",
        "questa domanda mi sembra... riciclata",
        "c'è un'eco qui o l'hai già detta",
        "ci risiamo",

        # Seccate / Dirette
        "l'hai già chiesta, ascolta meglio",
        "quante volte devo rispondere",
        "non è la prima volta che me lo chiedi",
        "già risposto, non insistere",
        "è la stessa domanda di prima",
        "sei ripetitivo",
        "sei ripetitiva",

        "lo hai già detto",
        "lo hai già chiesto",
        "me l'hai già chiesto",
        "sei ripetitivo",
        "ti stai ripetendo",
        "l'hai già detto",
        "perchè me lo chiedi di nuovo",
        "penso che me lo hai già chiesto"
    ]

    logger.debug("controllo repeatedQuestions")

    for repeated_answer in frasi_domanda_ripetuta:
        #logger.debug(f"{repeated_answer}, {user_text}, {repeated_answer in user_text}")
        if repeated_answer in user_text:
            logger.debug("QUESTION REPEATED")
            return True
    return False


# ---------------------------------------------------------------------------
# Fine aggiunte di Gabriele
# ---------------------------------------------------------------------------
