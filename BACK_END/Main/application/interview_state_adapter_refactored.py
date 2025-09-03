# interview_state_adapter_refactored.py
"""
Adapter puro: delega tutte le chiamate a InterviewState senza logica di business, fallback o manipolazione diretta dello stato.
Implementa il pattern Adapter correttamente, limitandosi a delegare le chiamate all'oggetto adattato.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from interviewer_reflection import InterviewerReflection
import os
from topic_detection import (
    COVERAGE_THRESHOLD_PERCENT as TD_COVERAGE_THRESHOLD_PERCENT,
    detect_covered_topics,
    topic_objects_from_meta,
    detect_covered_topics_with_gpt,
    covered_topics_with_gpt
)
from Main.core.logger import logger

from datetime import datetime, timezone

#from interview_state import InterviewState
# Costante importata direttamente dal modulo adattato
try:
    from interview_state import COVERAGE_THRESHOLD_PERCENT
except ImportError:
    COVERAGE_THRESHOLD_PERCENT = 60.0  # Valore di default

# Dizionario globale delle sessioni - Mapping user_id -> InterviewStateAdapter

class InterviewStateAdapter:
    """
    Adapter puro per InterviewState che delega le chiamate senza aggiungere logica di business.
    
    Questo adapter:
    1. Non contiene logica di business, solo delegazione
    2. Non manipola direttamente gli attributi dell'oggetto adattato
    3. Non fa monkey-patching dell'oggetto adattato
    4. Non implementa fallback nel proprio codice
    5. Espone un'interfaccia coerente al chiamante
    """
    
    
    """def __init__(self, user_id: str, script: List[Dict[str, Any]] = None):
        
        #Inizializza un nuovo adapter per InterviewState.
        #
        #Args:
        #    user_id: ID dell'utente
        #    script: Script dell'intervista (domande)
        
        
        # Deleghiamo la creazione all'oggetto adattato
        self._interview_state = InterviewState(user_id, script or [])"""

    def __init__(self, user_id: str, script: List[Dict[str, Union[str, List[str]]]]) -> None:
        self.user_id: str = user_id
        self.session_id: str = str(uuid.uuid4())  # mantiene compatibilità con DB
        self.script: List[Dict[str, Union[str, List[str]]]] = script
        self.idx: int = 0  # domanda corrente
        self.rm: InterviewerReflection = InterviewerReflection()  # gestore riflessioni
        self.questions: List[Dict[str, Any]] = []

        # Nuovi campi per la struttura della domanda corrente generata dall'LLM
        self.current_topic: Optional[str] = None
        self.current_subtopics: List[str] = []
        self.current_keywords: List[List[str]] = []  # keywords per ogni subtopic
        self.current_question_is_follow_up_for_subtopic: Optional[str] = None
        self.missing_topics: List[str] = []

        # Lista per memorizzare le risposte dell'utente e i relativi metadati
        self.user_responses: List[Dict[str, Any]] = []

        # Campi provenienti da InterviewState presente in user_session_service
        self.interview_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.current_question_id = None
        self.questions_asked = []  # ID delle domande già poste
        self.answers = {}  # Risposte fornite (question_id -> answer_text)
        self.completed = False
        self.score = None

    def to_string(self) -> str:
        return (
            f"user_id: {self.user_id}\n"
            f"session_id: {self.session_id}\n"
            f"script: {self.script}\n"
            f"idx: {self.idx}\n"
            f"questions: {self.questions}\n"
            f"current_topic: {self.current_topic}\n"
            f"current_subtopics: {self.current_subtopics}\n"
            f"current_keywords: {self.current_keywords}\n"
            f"current_question_is_follow_up_for_subtopic: {self.current_question_is_follow_up_for_subtopic}\n"
            f"missing_topics: {self.missing_topics}\n"
            f"user_responses: {self.user_responses}\n"
            f"interview_id: {self.interview_id}\n"
            f"start_time: {self.start_time}\n"
            f"current_question_id: {self.current_question_id}\n"
            f"questions_asked: {self.questions_asked}\n"
            f"answers: {self.answers}\n"
            f"completed: {self.completed}\n"
            f"score: {self.score}"
        )

    ########### DA INTERVIEW STATE #############
    # ---------------------------------------------------------------------
    # Proprietà/alias per retro‑compatibilità con parti di codice legacy
    # ---------------------------------------------------------------------
    @property
    def question(self) -> str:  # alias usato da altri componenti che fanno obj.question
        """Restituisce la domanda corrente mantenendo compatibilità legacy."""
        return self.domanda_corrente()

    def get_notes(self) -> str:
        """Delegato che restituisce eventuali note dal riflettore.

        Alcuni moduli chiamano ``InterviewState.get_notes()`` per contestualizzare
        il prompt del LLM. Se ``InterviewerReflection`` espone già `get_notes`,
        lo usiamo; altrimenti costruiamo un fallback concatenando i turni
        dell'assistente.
        """
        if hasattr(self.rm, "get_notes") and callable(getattr(self.rm, "get_notes")):
            return self.rm.get_notes()
        # Fallback minimale
        return "\n".join(
            t.get("text", "") for t in self.rm.transcript if t.get("speaker") == "assistant"
        )

    # --------- API chiamate dal main.py ---------
    def domanda_corrente(self) -> str:
        """Restituisce il testo della domanda corrente."""
        try:
            # Ottieni la domanda da DOMANDE
            #from Main.application.user_session_service import DOMANDE
            from Main.api.routes_interview import DOMANDE
            if DOMANDE and self.idx < len(DOMANDE):
                domanda = DOMANDE[self.idx]
                if isinstance(domanda, dict):
                    for key in ['question', 'text', 'domanda', 'testo']:
                        if key in domanda and domanda[key]:
                            return str(domanda[key])
        
        # Se arriviamo qui, non abbiamo trovato una domanda valida
            logger.warning(f"Domanda non trovata in DOMANDE all'indice {self.idx}")
            return "Domanda non disponibile"
        except Exception as e:
            logger.error(f"Errore nel recuperare la domanda corrente: {e}", exc_info=True)
            return "Domanda non disponibile"

    def save_user_response_and_reflect(self, text: str) -> None:
        """Salva risposta utente + riflettore."""
        self.rm.add_turn("user", text)

    def find_missing_topics(self, user_response = "") -> Tuple[List[str], float]:
        """Restituisce (subtopic mancanti, coverage_percent).

        Implementa la cascata exact‑lemma → fuzzy → cosine delegando al
        modulo ``topic_detection``.
        """
        # Importa DOMANDE
        #from Main.application.user_session_service import DOMANDE
        from Main.api.routes_interview import DOMANDE

        # --------------- 0. Validità indice/script --------------------
        if not DOMANDE or self.idx >= len(DOMANDE):
            return [], 0.0

        q: Dict[str, Any] = DOMANDE[self.idx]
        expected_subtopics: List[str] = q.get("subtopics", [])
        if not expected_subtopics:
            return [], 100.0  # nessun topic definito

        # --------------- 1. Ricava meta per i sub-topic ---------------
        try:
            topics = topic_objects_from_meta(
                subtopics=q.get("subtopics", []),
                keywords=q.get("keywords", []),
                lemma_sets=q.get("lemma_sets", []),
                fuzzy_norms=q.get("fuzzy_norms", []),
                vectors=q.get("vectors", []),
            )

        except (KeyError, TypeError, ValueError) as e:
            # Fallback se lo YAML non è stato ancora arricchito o ha struttura errata
            logger.warning(
                "Metadata derivati mancanti o errati: %s – fallback keyword-only", e
            )
            subtopics = q.get("subtopics", [])
            keywords = q.get("keywords", [])

            # Verifica che keywords sia una lista di liste
            if not isinstance(keywords, list):
                keywords = []
            elif keywords and not isinstance(keywords[0], list):
                # Se keywords è una lista semplice, la trasformiamo in lista di liste
                keywords = [keywords]

            # Assicurati che keywords abbia la stessa lunghezza di subtopics
            while len(keywords) < len(subtopics):
                keywords.append([])

            topics = topic_objects_from_meta(
                subtopics=subtopics,
                keywords=keywords,
                lemma_sets=[[] for _ in subtopics],
                fuzzy_norms=["" for _ in subtopics],
                vectors=[[0.0] * 300 for _ in subtopics],  # spaCy dim.
            )

        # --------------- 2. Recupera l'ultima risposta utente ---------
        """last_user_text: str = next(
            (
                t.get("text", "")
                for t in reversed(self.rm.transcript)
                if t.get("speaker") == "user"
            ),
            "",
        )"""
        last_user_text = user_response 
        if not last_user_text.strip():
            return expected_subtopics, 0.0  # nessuna risposta

        # --------------- 3. Applica cascata ---------------------------
        try:
            #covered, coverage_frac = detect_covered_topics(last_user_text, topics)
            logger.debug( "----------------DEBUG TOPIC DETECTION----------------")
            logger.debug(f"last_user_text: {last_user_text}")
            logger.debug(f"user text: {user_response}")
            logger.debug("GPT CALLED: detect_covered_topics_with_gpt")
            covered, coverage_frac = detect_covered_topics_with_gpt(user_response.lower(), topics, expected_subtopics[0])
            #covered, coverage_frac = detect_covered_topics_with_gpt(user_response.lower(), expected_subtopics, expected_subtopics[0])
            logger.debug("GPT EXIT: detect_covered_topics_with_gpt")
            logger.debug(f"covered topics: {covered}")
            logger.debug(f"expected_subtopics: {expected_subtopics}")
            missing = [t for t in expected_subtopics if t not in covered]
            logger.debug(f"missing topics: {missing}")
            coverage_percent = round(coverage_frac * 100, 1)
            logger.debug(f"coverage %: {coverage_percent}")
        except Exception as e:
            logger.error("Errore nell'analisi dei topic: %s", e)
            return expected_subtopics, 0.0  # fallback completo in caso di errore

        logger.debug( "----------------------------------------------")
        logger.debug(
            "Analisi risposta – coperti %s / %s (%.1f%%)",
            len(covered),
            len(topics),
            coverage_percent,
        )
        
        # --------------- 4. Decide follow‑up -------------------------
        if coverage_percent >= COVERAGE_THRESHOLD_PERCENT:
            return [], coverage_percent
        return missing, coverage_percent

    def advance_main(self) -> None:
        """Passa alla prossima domanda del copione."""
        self.idx = min(self.idx + 1, len(self.script) - 1)

    ########### DA INTERVIEW STATE ADAPTER REFACTORED #############
    
    # ---- Metodi di accesso alla sessione (statici) ----
    
    # ---- Metodi delegati a InterviewState ----
    
    def get_current_question(self) -> Dict[str, Any]:
        """
        Ottiene la domanda corrente dalla sessione.
        
        Returns:
            Dizionario con i dati della domanda corrente
        """
        try:
            question_text = self.domanda_corrente()
            current_idx = self.idx
            
            # Costruisce un dizionario con i dati della domanda
            result = {
                "id": f"q{current_idx + 1}",
                "text": question_text,
                "category": "main",
                "difficulty": "medium",
                "type": "main"
            }
            
            # Verifica se è una domanda di follow-up
            if self.current_question_is_follow_up_for_subtopic:
                result["id"] = f"{result['id']}_followup"
                result["type"] = "follow_up"
                result["follow_up_for"] = self.current_question_is_follow_up_for_subtopic
                
            return result
        except Exception as e:
            logger.error(f"Errore nell'ottenere la domanda corrente: {e}", exc_info=True)
            return {
                "id": "error",
                "text": "Errore nel recupero della domanda",
                "category": "error",
                "difficulty": "medium"
            }
    
    def save_answer(self, user_response: str) -> Tuple[bool, float, List[str]]:
        
        #Salva la risposta dell'utente e verifica se è necessario un follow-up.
        
        #Args:
        #    user_response: Risposta testuale dell'utente
            
        #Returns:
        #    Tuple[bool, float, List[str]]: 
        #        - Flag che indica se è necessario un follow-up
        #        - Percentuale di copertura dei sottotopici
        #        - Lista dei topic mancanti
        
        try:
            # Salva la risposta nell'oggetto InterviewStateAdapter
            self.save_user_response_and_reflect(user_response)

            # QUESTO LO SPOSTEREI POI O LO REPLICHEREI DA UN ALTRA PARTE
            # RECUPERA I VARI DATI DELLA SESSIONE
            current_id = self.current_question_id
            if current_id is None:
                self.current_question_id= self.questions[0].get('id')
                current_id = self.current_question_id
            if self.answers is None:
                self.answers = {}
            if self.answers.get(current_id,None) is None:
                self.answers[current_id]=[]
            from Main.api.routes_interview import get_question_metadata_status
            try:
                if current_id is not None:                
                    """metadata = get_question_metadata_status(current_id)
                    if metadata:
                        metadata_content = metadata['metadata']
                        primary_topic = metadata_content.get('primary_topic')
                        self.current_topic = primary_topic
                        subtopics = metadata_content.get('subtopics', [])
                        self.current_subtopics = subtopics
                        keywords = metadata_content.get('keywords', [])
                        self.current_keywords = keywords
                        current_question = None
                        for q in self.questions:
                            if q.get('id') == current_id:
                                current_question = q
                                break
                        if current_question:
                            #question_idx = self.questions.index(current_question)
                            #question_text = current_question.get('Domanda', '')
                            #if question_idx:
                            #    self.answers[question_idx].append(user_response)
                            self.answers[current_id].append(user_response)"""

                    primary_topic =  self.questions[0]["topics"][0]
                    self.current_topic = primary_topic
                    subtopics = self.questions[0]["topics"]
                    self.current_subtopics = subtopics
                    keywords = self.questions[0]["keywords"]
                    self.current_keywords = keywords
                    current_question = None
                    for q in self.questions:
                        if q.get('id') == current_id:
                            current_question = q
                            break
                    if current_question:
                        #question_idx = self.questions.index(current_question)
                        #question_text = current_question.get('Domanda', '')
                        #if question_idx:
                        #    self.answers[question_idx].append(user_response)
                        self.questions_asked.append(current_question.get('Domanda', ''))
                        self.answers[current_id].append(user_response)
            except Exception as e:
                print(f"Error: {e}") 

            print("SAVE ANSWER\n",self.to_string())
            
            # Ottiene i topic mancanti e la percentuale di copertura
            #missing_topics, coverage_percent = self.find_missing_topics(user_response)
            covered_topics, coverage_frac = covered_topics_with_gpt(user_response, subtopics, primary_topic)

            missing_topics = [t for t in subtopics if t not in covered_topics]
            coverage_percent = round(coverage_frac * 100, 1)
            
            self.missing_topics = missing_topics
            self.score = coverage_percent

            if self.current_topic in self.missing_topics:
                self.current_question_is_follow_up_for_subtopic = self.current_topic
            else:
                self.current_question_is_follow_up_for_subtopic = None

            
            # Decide se è necessario un follow-up
            needs_followup = coverage_percent < COVERAGE_THRESHOLD_PERCENT and missing_topics
            
            # {id, domanda, topics, keywords}
            self.questions[0]["topics"] = missing_topics
            if coverage_percent > COVERAGE_THRESHOLD_PERCENT:
                self.questions.pop(0)
            
            # Se non è necessario un follow-up, avanza alla prossima domanda
            if not needs_followup:
                #self.advance_main()
                self.idx = self.idx+1
            
            return needs_followup, coverage_percent, missing_topics
        except Exception as e:
            logger.error(f"Errore nel salvataggio della risposta: {e}", exc_info=True)
            return False, 0.0, []

    def get_context(self):
        # QUESTO LO SPOSTEREI POI O LO REPLICHEREI DA UN ALTRA PARTE
        # RECUPERA I VARI DATI DELLA SESSIONE
        print("DEBUG get_context 1")
        current_id = self.current_question_id
        if current_id is None:
            self.current_question_id= self.questions[0].get('id')
        print("DEBUG get_context 2")
        if self.answers is None:
            self.answers = {}
        print("DEBUG get_context 3")
        self.answers[current_id] = user_response
        print("DEBUG get_context 4")
        from Main.api.routes_interview import get_question_metadata_status
        try:
            if current_id:                
                metadata = get_question_metadata_status(current_id)
                if metadata:
                    print("DEBUG get_context 5")
                    metadata_content = metadata['metadata']
                    primary_topic = metadata_content.get('primary_topic')
                    self.current_topic = primary_topic
                    subtopics = metadata_content.get('subtopics', [])
                    print("DEBUG get_context 6")
                    self.current_subtopics = subtopics
                    keywords = metadata_content.get('keywords', [])
                    self.current_keywords = keywords
                    print("DEBUG get_context 7")
                    current_question = None
                    for q in self.questions:
                        if q.get('id') == current_id:
                            current_question = q
                            break
                    print("DEBUG get_context 8")
                    if current_question:
                        question_idx = self.questions.index(current_question)
                        question_text = current_question.get('Domanda', '')
                        if question_idx:
                            self.answers[question_idx].append(user_response)
        except Exception as e:
            print(f"Error: {e}") 
    
    def advance_to_next_question(self) -> bool:
        """
        Avanza alla prossima domanda dell'intervista.
        
        Returns:
            True se l'avanzamento è avvenuto con successo
        """
        try:
            # Avanza alla prossima domanda
            self.advance_main()
            return True
        except Exception as e:
            logger.error(f"Errore nell'avanzamento alla prossima domanda: {e}", exc_info=True)
            return False
    
    # ---- Metodi di accesso agli attributi di InterviewState ----
    
    def get_user_id(self) -> str:
        """Restituisce l'ID dell'utente."""
        return self.user_id
    
    def get_session_id(self) -> str:
        """Restituisce l'ID della sessione."""
        return self.session_id
    
    def get_current_topic(self) -> Optional[str]:
        """Restituisce il topic corrente."""
        return self.current_topic
    
    def get_current_subtopics(self) -> List[str]:
        """Restituisce i subtopic correnti."""
        return self.current_subtopics
    
    def get_reflection_notes(self) -> str:
        """Restituisce le note di riflessione. Alias di get_notes per chiarezza API."""
        return self.get_notes()
    
    def set_follow_up_question(self, question_text: str, subtopic: str) -> None:
        """
        Imposta una domanda di follow-up.
        
        Args:
            question_text: Testo della domanda di follow-up
            subtopic: Subtopic per cui è necessario il follow-up
        """
        # Questi attributi devono essere presenti in InterviewState
        # In una vera implementazione, InterviewState dovrebbe avere un metodo dedicato
        # per questa operazione, invece di esporre gli attributi direttamente
        self.follow_up_question = question_text
        self.current_question_is_follow_up_for_subtopic = subtopic

    # PRESI DA userr_session_service.py

    def get_next_question(self) -> Optional[Dict[str, Any]]:
        """Ottiene la prossima domanda da porre"""
        # Filtra le domande non ancora poste
        available_questions = [q for q in self.questions 
                             if q.get("id") not in self.questions_asked]
        logger.debug("--------------CHECK QUESTIONS--------------")
        logger.debug(f"available_questions: {available_questions}")
        
        if not available_questions or len(available_questions)==0:
            self.completed = True
            return None
            
        # Seleziona la prima domanda disponibile
        next_question = available_questions[0]
        question_id = next_question.get("id", str(len(self.questions_asked)))

        logger.debug(f"next_question: {next_question}")
        logger.debug(f"question_id: {question_id}")
        
        # Aggiorna lo stato dell'intervista
        self.current_question_id = question_id
        self.questions_asked.append(question_id)
        
        return next_question
    
    def add_answer(self, question_id: str, answer_text: str) -> bool:
        """Registra una risposta data dall'utente"""
        if question_id != self.current_question_id:
            logger.warning(f"Tentativo di rispondere a domanda non corrente: {question_id}")
            return False
            
        self.answers[question_id] = answer_text
        logger.info(f"Risposta registrata per domanda {question_id}")
        return True
    
    def complete_interview(self) -> int:
        """Termina l'intervista e calcola un punteggio"""
        self.completed = True
        
        # In una versione reale, calcoleremo un punteggio basato sulle risposte
        # Per ora, assegniamo un punteggio casuale
        import random
        self.score = random.randint(60, 95)
        
        logger.info(f"Intervista {self.interview_id} completata con punteggio: {self.score}")
        return self.score

    