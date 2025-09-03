# interview_state_adapter_refactored.py
"""
Adapter puro: delega tutte le chiamate a InterviewState senza logica di business, fallback o manipolazione diretta dello stato.
Implementa il pattern Adapter correttamente, limitandosi a delegare le chiamate all'oggetto adattato.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from interview_state import InterviewState
from Main.core.logger import logger

# Costante importata direttamente dal modulo adattato
try:
    from interview_state import COVERAGE_THRESHOLD_PERCENT
except ImportError:
    COVERAGE_THRESHOLD_PERCENT = 60.0  # Valore di default

# Dizionario globale delle sessioni - Mapping user_id -> InterviewState
SESSIONS: Dict[str, InterviewState] = {}

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
    
    def __init__(self, user_id: str, script: List[Dict[str, Any]] = None):
        """
        Inizializza un nuovo adapter per InterviewState.
        
        Args:
            user_id: ID dell'utente
            script: Script dell'intervista (domande)
        """
        
        # Deleghiamo la creazione all'oggetto adattato
        self._interview_state = InterviewState(user_id, script or [])
    
    # ---- Metodi di accesso alla sessione (statici) ----
    
    @staticmethod
    def get_state(uid: str) -> InterviewState:
        """
        Recupera lo stato dell'intervista per un utente o ne crea uno nuovo.
        
        Args:
            uid: ID dell'utente
            
        Returns:
            Istanza di InterviewState
        """
        if uid not in SESSIONS:
            # Creiamo una nuova sessione con l'implementazione esterna
            SESSIONS[uid] = InterviewState(uid, [])
            logger.info(f"Creata nuova sessione per l'utente {uid}")
        
        return SESSIONS[uid]
    
    @staticmethod
    def has_active_session(uid: str) -> bool:
        """Verifica se esiste una sessione attiva per l'utente."""
        return uid in SESSIONS
    
    @staticmethod
    def reset_session(uid: str) -> bool:
        """Elimina una sessione utente se esiste."""
        if uid in SESSIONS:
            del SESSIONS[uid]
            logger.info(f"Sessione eliminata per l'utente {uid}")
            return True
        return False
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """
        Restituisce informazioni sulle sessioni attive.
        
        Returns:
            Dizionario con statistiche sulle sessioni
        """
        return {
            "active_sessions": len(SESSIONS),
            "session_ids": list(SESSIONS.keys())
        }
    
    # ---- Metodi delegati a InterviewState ----
    
    def get_current_question(self) -> Dict[str, Any]:
        """
        Ottiene la domanda corrente dalla sessione.
        
        Returns:
            Dizionario con i dati della domanda corrente
        """
        try:
            question_text = self._interview_state.domanda_corrente()
            current_idx = self._interview_state.idx
            
            # Costruisce un dizionario con i dati della domanda
            result = {
                "id": f"q{current_idx + 1}",
                "text": question_text,
                "category": "main",
                "difficulty": "medium",
                "type": "main"
            }
            
            # Verifica se è una domanda di follow-up
            if self._interview_state.current_question_is_follow_up_for_subtopic:
                result["id"] = f"{result['id']}_followup"
                result["type"] = "follow_up"
                result["follow_up_for"] = self._interview_state.current_question_is_follow_up_for_subtopic
                
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
        """
        Salva la risposta dell'utente e verifica se è necessario un follow-up.
        
        Args:
            user_response: Risposta testuale dell'utente
            
        Returns:
            Tuple[bool, float, List[str]]: 
                - Flag che indica se è necessario un follow-up
                - Percentuale di copertura dei sottotopici
                - Lista dei topic mancanti
        """
        try:
            # Salva la risposta nell'oggetto InterviewState
            self._interview_state.save_user_response_and_reflect(user_response)
            
            # Ottiene i topic mancanti e la percentuale di copertura
            missing_topics, coverage_percent = self._interview_state.missing_topics(user_response)
            
            # Decide se è necessario un follow-up
            needs_followup = coverage_percent < COVERAGE_THRESHOLD_PERCENT and missing_topics
            
            # Se non è necessario un follow-up, avanza alla prossima domanda
            if not needs_followup:
                self._interview_state.advance_main()
            
            return needs_followup, coverage_percent, missing_topics
        except Exception as e:
            logger.error(f"Errore nel salvataggio della risposta: {e}", exc_info=True)
            return False, 0.0, []
    
    def advance_to_next_question(self) -> bool:
        """
        Avanza alla prossima domanda dell'intervista.
        
        Returns:
            True se l'avanzamento è avvenuto con successo
        """
        try:
            # Avanza alla prossima domanda
            self._interview_state.advance_main()
            return True
        except Exception as e:
            logger.error(f"Errore nell'avanzamento alla prossima domanda: {e}", exc_info=True)
            return False
    
    # ---- Metodi di accesso agli attributi di InterviewState ----
    
    def get_user_id(self) -> str:
        """Restituisce l'ID dell'utente."""
        return self._interview_state.user_id
    
    def get_session_id(self) -> str:
        """Restituisce l'ID della sessione."""
        return self._interview_state.session_id
    
    def get_current_topic(self) -> Optional[str]:
        """Restituisce il topic corrente."""
        return self._interview_state.current_topic
    
    def get_current_subtopics(self) -> List[str]:
        """Restituisce i subtopic correnti."""
        return self._interview_state.current_subtopics
    
    def get_reflection_notes(self) -> str:
        """Restituisce le note di riflessione. Alias di get_notes per chiarezza API."""
        return self._interview_state.get_notes()
    
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
        self._interview_state.follow_up_question = question_text
        self._interview_state.current_question_is_follow_up_for_subtopic = subtopic
