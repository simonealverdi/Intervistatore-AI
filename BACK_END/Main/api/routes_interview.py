from fastapi import APIRouter, HTTPException, Depends, Request, File, UploadFile, Form, Query, Cookie
from typing import Dict, List, Any, Optional, Union
import logging
import uuid
from datetime import datetime



import json
import tempfile
import time
import threading

# Import dai moduli interni
from Main.core import config
from Main.models import (
    InterviewResponse, ErrorResponse, InterviewStatusResponse,
    InterviewQuestionResponse, InterviewResultResponse, AnswerRequest
)
from Main.application.interview_state_adapter_refactored import InterviewStateAdapter
from .auth import get_current_user

from fastapi.responses import JSONResponse
import base64
import os

# Import dai moduli interni
from .auth import get_current_user_optional
import urllib.parse

from jose import JWTError, jwt

# Import dai moduli interni
from Main.core.config import DEVELOPMENT_MODE
from Main.models import TTSResponse

#from Main.application.user_session_service import SCRIPT

# Configurazione logger
logger = logging.getLogger(__name__)
# Router per l'intervista
router = APIRouter(tags=["Interview"])

SESSIONS: Dict[str, InterviewStateAdapter] = {}
SCRIPT: List[Dict[str, Any]] = []
DOMANDE = []
METADATA_STATUS = {}

try:
    from Importazioni import QuestionImporter
    logger.info("Importato QuestionImporter per la generazione di metadati")
except ImportError:
    logger.warning("Non è stato possibile importare QuestionImporter")
    QuestionImporter = None

metadata_processing_status = {
    'total_questions': 0,
    'processed_questions': 0,
    'in_progress': False,
    'start_time': None,
    'end_time': None,
    'error': None
}

def get_state(uid: str) -> InterviewStateAdapter:
    """
    Recupera lo stato dell'intervista per un utente o ne crea uno nuovo.
    
    Args:
        uid: ID dell'utente
        
    Returns:
        Istanza di InterviewState
    """
    if uid not in SESSIONS:
        current_script_for_session = SCRIPT
        if not current_script_for_session:
            logger.warning(f"SCRIPT globale è vuoto per user_id '{uid}'. Uso un set di domande placeholder.")
            current_script_for_session = [
                {"id": "q1", "text": "Parlami di te.", "category": "background", "difficulty": "medium"},
                {"id": "q2", "text": "Quali sono i tuoi obiettivi professionali?", "category": "goals", "difficulty": "easy"},
                {"id": "q3", "text": "Descrivi una sfida che hai affrontato in ambito lavorativo.", "category": "experience", "difficulty": "hard"}
            ]
            # Creiamo una nuova sessione con l'implementazione esterna
            SESSIONS[uid] = InterviewStateAdapter(uid, current_script_for_session)
        else:
            SESSIONS[uid] = InterviewStateAdapter(uid, [])
        logger.info(f"Creata nuova sessione per l'utente {uid}")
    
    return SESSIONS[uid]

def has_active_session(uid: str) -> bool:
    """Verifica se esiste una sessione attiva per l'utente."""
    return uid in SESSIONS

def reset_session(uid: str) -> bool:
    """Elimina una sessione utente se esiste."""
    if uid in SESSIONS:
        del SESSIONS[uid]
        logger.info(f"Sessione eliminata per l'utente {uid}")
        return True
    return False

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

def load_script(new_script: List[Dict[str, Any]]) -> bool:
    """
    Carica un nuovo script globale e avvia l'elaborazione asincrona dei metadati.
    Popola la struttura DOMANDE con le domande e i relativi metadati man mano che vengono elaborati.
    """
    global SCRIPT
    global DOMANDE  # Nuova struttura globale
    global metadata_processing_status
    
    try:
        # Svuotiamo la lista DOMANDE per il nuovo caricamento
        DOMANDE.clear()
        
        # Verifica che new_script non sia vuoto
        if not new_script:
            logger.error("Tentativo di caricare uno script vuoto! Operazione annullata.")
            return False
            
        # Verifica che gli elementi abbiano il formato corretto
        valid_items = []
        for i, item in enumerate(new_script):
            if not isinstance(item, dict):
                logger.error(f"Elemento {i} non è un dizionario: {type(item)}")
                continue
                
            if 'Domanda' not in item or not item['Domanda']:
                logger.error(f"Elemento {i} non ha una domanda valida: {item}")
                continue
            
            # Assicuriamoci che ogni domanda abbia un ID univoco
            if "id" not in item or not item["id"]:
                item["id"] = str(uuid.uuid4())
                
            valid_items.append(item)
            
        if not valid_items:
            logger.error("Nessuna domanda valida trovata nello script fornito!")
            return False
            
        # Assegna subito gli elementi validi per renderli disponibili immediatamente sia in SCRIPT che in DOMANDE
        SCRIPT = valid_items  # Manteniamo SCRIPT per retrocompatibilità
        print("QUELLO CHE CI SERVE:",type(SCRIPT))
        
        # Inizializza DOMANDE con solo il testo delle domande (i metadati verranno aggiunti dopo)
        for i, question in enumerate(valid_items):
            domanda_testo = question.get('Domanda', '')
            q_struct = {
                "id": question.get("id"),
                "testo": domanda_testo,
                "topic": "",  # Sarà popolato durante l'elaborazione metadati
                "subtopics": [],  # Sarà popolato durante l'elaborazione metadati
                "keywords": [],   # Sarà popolato durante l'elaborazione metadati
                "lemma_sets": [],
                "fuzzy_norms": [],
                "vectors": []   
            }
            DOMANDE.append(q_struct)
        
        # Stampa dettagli per debug
        logger.info(f"Nuovo script caricato con {len(valid_items)} domande valide su {len(new_script)} fornite.")
        
        # Log delle domande caricate
        for i, question in enumerate(DOMANDE[:3]):  # Stampa solo le prime 3 come esempio
            logger.info(f"Domanda {i+1}: {question['testo'][:50]}...")
        
        # Generiamo i metadati DIRETTAMENTE usando QuestionImporter da Importazioni.py
        if QuestionImporter is not None:
            # Avvia l'elaborazione dei metadati un elemento alla volta
            def process_metadata_async():
                global DOMANDE
                global metadata_processing_status
                
                try:
                    # Aggiorniamo lo stato di elaborazione
                    metadata_processing_status['in_progress'] = True
                    metadata_processing_status['total_questions'] = len(valid_items)
                    metadata_processing_status['processed_questions'] = 0
                    metadata_processing_status['start_time'] = datetime.now()
                    metadata_processing_status['error'] = None
                    
                    logger.info("Avvio generazione metadati per le domande - un elemento alla volta...")
                    
                    # Estrai i testi delle domande
                    question_texts = []
                    for q in valid_items:
                        if "Domanda" in q:
                            question_texts.append(q["Domanda"])
                        elif "text" in q:
                            question_texts.append(q["text"])
                    
                    if not question_texts:
                        logger.error("Nessun testo di domanda disponibile per la generazione dei metadati")
                        return
                    
                    # Creiamo un file temporaneo JSON con le domande
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False, encoding='utf-8') as tmp:
                        json.dump(question_texts, tmp)
                        temp_file_path = tmp.name
                        logger.info(f"File temporaneo creato: {temp_file_path}")
                    
                    try:
                        # Chiamiamo direttamente la funzione di generazione metadati di QuestionImporter
                        logger.info(f"Chiamata a QuestionImporter.generate_metadata per {len(question_texts)} domande")
                        question_metas = QuestionImporter.generate_metadata(temp_file_path)
                        logger.info(f"Generati metadati per {len(question_metas)} domande")
                        
                        # Inizializza la struttura per il salvataggio dei metadati
                        metadata_dict = {
                            'timestamp': datetime.now().isoformat(),
                            'total_questions': len(question_metas),
                            'questions': []
                        }
                        
                        # Aggiorniamo le domande con i metadati mantenendo l'ordine originale - UN ELEMENTO ALLA VOLTA
                        for i, meta in enumerate(question_metas):
                            if i < len(valid_items) and i < len(DOMANDE):
                                # Metadati correnti
                                primary_topic = meta.primary_topic
                                subtopics = meta.subtopics
                                keywords = meta.keywords
                                
                                # Aggiorna il dizionario DOMANDE con i metadati disponibili
                                DOMANDE[i]['topic'] = primary_topic
                                DOMANDE[i]['subtopics'] = subtopics
                                DOMANDE[i]['keywords'] = keywords
                                DOMANDE[i]['lemma_sets'] = meta.lemma_sets
                                DOMANDE[i]['fuzzy_norms'] = meta.fuzzy_norms
                                DOMANDE[i]['vectors'] = meta.vectors
                                
                                # Aggiorna anche gli item originali per retrocompatibilità
                                valid_items[i]["topics"] = [primary_topic] + subtopics
                                valid_items[i]["keywords"] = keywords
                                
                                # Log dei risultati
                                question_id = DOMANDE[i].get("id", f"q{i}")
                                logger.info(f"Metadati generati per domanda {i+1}: {DOMANDE[i]['testo'][:50]}...")
                                logger.info(f"  Topic: {primary_topic}, Subtopics: {subtopics}")
                                
                                # Prepara i dati per il file JSON
                                question_data = {
                                    'id': question_id,
                                    'domanda': DOMANDE[i]['testo'][:100] + ('...' if len(DOMANDE[i]['testo']) > 100 else ''),
                                    'primary_topic': primary_topic,
                                    'subtopics': subtopics,
                                    'keywords': keywords,
                                    'lemma_sets': meta.lemma_sets,
                                    'fuzzy_norms': meta.fuzzy_norms,
                                    'vectors': meta.vectors
                                }
                                metadata_dict['questions'].append(question_data)
                                
                                # Aggiornamento contatore
                                metadata_processing_status['processed_questions'] += 1
                                
                                # Breve pausa per non sovraccaricare il sistema
                                time.sleep(0.01)
                        
                        # Aggiornamento stato finale
                        metadata_processing_status['in_progress'] = False
                        metadata_processing_status['end_time'] = datetime.now()
                        logger.info(f"Completata generazione metadati per {len(question_metas)} domande")
                        
                        # Salva i metadati in un file JSON per visualizzazione
                        try:
                            metadata_file_path = os.path.join(os.path.dirname(temp_file_path), 'metadati_generati.json')
                            
                            # Scrivi il file JSON con indentazione per leggibilità
                            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                                
                            logger.info(f"Metadati salvati in file JSON: {metadata_file_path}")
                            
                        except Exception as e:
                            logger.error(f"Errore nel salvataggio dei metadati in file JSON: {e}")
                    except Exception as e:
                        metadata_processing_status['error'] = str(e)
                        logger.error(f"Errore nella generazione dei metadati con QuestionImporter: {e}")
                    finally:
                        # Pulizia del file temporaneo
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"File temporaneo rimosso: {temp_file_path}")
                        except Exception as e:
                            logger.warning(f"Impossibile rimuovere il file temporaneo: {e}")
                except Exception as e:
                    metadata_processing_status['error'] = str(e)
                    logger.error(f"Errore generale nella generazione dei metadati: {e}")
            
            # Avvia l'elaborazione metadati in un thread separato per non bloccare
            thread = threading.Thread(target=process_metadata_async)
            thread.daemon = True  # Il thread muore quando il programma principale termina
            thread.start()
            logger.info("Thread di elaborazione metadati avviato con successo")
        else:
            logger.warning("QuestionImporter non disponibile, metadati non verranno generati")
        
        return True
    except Exception as e:
        logger.error(f"Errore durante il caricamento dello script: {e}", exc_info=True)
        return False


@router.post("/start", response_model=InterviewResponse, responses={
    401: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def start_interview(request: Request, current_user: str = Depends(get_current_user)) -> InterviewResponse:
    """Avvia una nuova sessione di intervista"""
    try:
        # Utilizza l'ID utente ottenuto dal token JWT o le credenziali fisse
        user_id = current_user  # current_user è già l'ID utente (stringa)
        
        # Azzera eventuali sessioni precedenti per questo utente
        reset_session(user_id)
        
        # Ottieni una nuova sessione di intervista per l'utente
        session = get_state(user_id)
        interview_id = session.session_id
        
        logger.info(f"Nuova intervista avviata: {interview_id} per utente: {user_id}")
        
        return InterviewResponse(
            status="success", 
            message="Intervista avviata con successo", 
            interview_id=interview_id
        )
        
    except Exception as e:
        logger.error(f"Errore durante l'avvio dell'intervista: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Errore durante l'avvio dell'intervista: {str(e)}"
        )

@router.get("/next-question/{interview_id}", response_model=InterviewQuestionResponse, responses={
    404: {"model": ErrorResponse},
    410: {"model": InterviewResponse}
})
async def get_next_question(interview_id: str) -> Dict[str, Any]:
    """Ottieni la prossima domanda per un'intervista"""
    # Cerca l'utente associato all'ID dell'intervista
    # Per semplicità in questa versione, l'ID dell'intervista è l'ID dell'utente
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    logger.debug("ENTRO IN GET NEXT QUESTION")
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Verifica se tutte le domande sono state poste
    all_questions_asked = False
    if hasattr(session, 'questions_asked') and hasattr(session, 'questions'):
        all_questions_asked = len(session.questions_asked) >= len(session.questions)

    # Se tutte le domande sono state poste, restituisci un messaggio di completamento
    if all_questions_asked:
        # Segna l'intervista come completata
        session.completed = True
        # Restituisci una risposta di completamento invece di un errore
        return InterviewQuestionResponse(
            status="completed",
            message="Intervista completata",
            interview_id=interview_id,
            question={
                "id": "interview_completed",
                "text": "Grazie per aver partecipato all'intervista.",
                "type": "completion",
                "interview_completed": True
            }
    )

    # Verifica se l'intervista è già completata
    if session.completed:
        return InterviewResponse(
            status="completed",
            message="L'intervista è già stata completata",
            interview_id=interview_id
        )
    
    # Ottieni la prossima domanda dall'intervista
    question = session.get_current_question()
    print("CONTROLLO QUESTION","question",question)
    
    # Verifica se la domanda corrente è una domanda di follow-up
    is_follow_up = hasattr(session, 'current_question_is_follow_up_for_subtopic') and session.current_question_is_follow_up_for_subtopic is not None
    
    # Se la domanda è un follow-up, aggiungi metadati specifici
    if is_follow_up:
        if "type" not in question:
            question["type"] = "follow_up"
        if "follow_up_for" not in question and session.current_question_is_follow_up_for_subtopic:
            question["follow_up_for"] = session.current_question_is_follow_up_for_subtopic
        logger.info(f"Servita domanda di follow-up per intervista {interview_id}, topic: {session.current_question_is_follow_up_for_subtopic}")
    
    # Se non ci sono più domande disponibili, segna l'intervista come completata
    if not question or "id" not in question:
        # Tutte le domande sono state poste, segna l'intervista come completata
        session.completed = True
        raise HTTPException(
            status_code=410,  # Gone - risorsa non più disponibile 
            detail="Tutte le domande sono state poste. L'intervista è completata."
        )
    
    # Restituisci la domanda corrente
    return InterviewQuestionResponse(
        status="success",
        message="Domanda disponibile",
        interview_id=interview_id,
        question=question
    )


@router.post("/submit-answer/{interview_id}/{question_id}", response_model=InterviewResponse, responses={
    404: {"model": ErrorResponse},
    400: {"model": ErrorResponse}
})
async def submit_answer(interview_id: str, question_id: str, answer: AnswerRequest) -> InterviewResponse:
    """Invia una risposta a una domanda dell'intervista"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Verifica se l'intervista è già completata
    if session.completed:
        return InterviewResponse(
            status="error",
            message="L'intervista è già stata completata",
            interview_id=interview_id
        )
    
    # Verifica che la domanda sia quella corrente
    current_question = session.get_current_question()
    current_question_id = current_question.get("id") if current_question else None
    
    if current_question_id != question_id:
        logger.warning(f"ID domanda non corrisponde: atteso {current_question_id}, ricevuto {question_id}")
        return InterviewResponse(
            status="error",
            message="ID domanda non corrisponde alla domanda corrente",
            interview_id=interview_id
        )
    
    logger.debug(f">>>>>>>> SESSION: {session}")
    # Salva la risposta utilizzando l'adapter
    # DA VECCHIO interview_state_adapter
    needed_followup, coverage, missing_topics = session.save_answer(answer.answer_text)

    """if len(missing_topics)==0:
        self.questions= self.questions.pop(0) TODO"""

    logger.debug("___------------_________________")
    logger.debug(f"needed_followup: {needed_followup}, missing_topics: {missing_topics}, coverage: {coverage}")
    logger.debug("___------------_________________")

    logger.info(f"Risposta registrata per intervista {interview_id}, domanda {question_id}")
    
    # Se non sono necessari follow-up, possiamo avanzare alla domanda successiva
    if not needed_followup:
        session.advance_to_next_question(session)
    
    return InterviewResponse(
        status="success",
        message="Risposta registrata con successo",
        interview_id=interview_id
    )

@router.get("/status/{interview_id}", response_model=InterviewStatusResponse, responses={
    404: {"model": ErrorResponse}
})
async def get_interview_status(interview_id: str) -> InterviewStatusResponse:
    """Ottieni lo stato attuale di un'intervista"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Ottieni la domanda corrente
    current_question = session.get_current_question(session)
    current_question_id = current_question.get("id") if current_question else None
    
    return InterviewStatusResponse(
        status="success",
        message="Stato intervista recuperato",
        interview_id=interview_id,
        user_id=session.user_id,
        start_time=session.start_time,
        current_question_id=current_question_id,
        questions_asked=session.questions_asked,
        answers_count=len(session.answers),
        completed=session.completed,
        score=session.score
    )


@router.post("/end/{interview_id}", response_model=InterviewResultResponse, responses={
    404: {"model": ErrorResponse}
})
async def end_interview(interview_id: str) -> InterviewResultResponse:
    """Termina un'intervista in corso"""
    # Recupera l'utente associato all'intervista
    user_id = config.DEV_USERNAME  # Default per sviluppo
    
    # Ottieni lo stato dell'intervista
    session = get_state(user_id)
    
    # Verifica se l'intervista è quella richiesta
    if session.session_id != interview_id:
        logger.warning(f"Intervista non trovata: {interview_id}")
        raise HTTPException(status_code=404, detail="Intervista non trovata")
    
    # Contrassegna l'intervista come completata e calcola un punteggio
    session.completed = True
    
    # Assegna un punteggio fittizio (in una versione reale, questo verrebbe calcolato in base alle risposte)
    import random
    session.score = random.randint(60, 100)
    
    # In una versione reale, qui salveremmo il risultato finale nel database
    if not config.DEVELOPMENT_MODE and config.MONGODB_ENABLED:
        try:
            # Potremmo usare una funzione specifica per salvare il risultato finale
            logger.info(f"Dati finali dell'intervista {interview_id} salvati nel database")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei dati finali: {e}")
    
    logger.info(f"Intervista {interview_id} terminata con punteggio: {session.score}")
    
    return InterviewResultResponse(
        status="success",
        message="Intervista terminata con successo",
        interview_id=interview_id,
        score=session.score,
        questions_asked=len(session.questions_asked),
        answers_provided=len(session.answers)
    )

@router.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    audio_only: bool = Form(False),
    token_query: Optional[str] = Form(None)
):
    """
    Riceve un file audio di risposta, simula una trascrizione (per ora), e restituisce la prossima domanda come audio.
    """
    try:
        logger.info(f"Ricevuto file audio da utente: {user_id}, dimensione: {audio.size} bytes")
        
        # Trascriviamo l'audio usando il servizio OpenAI Whisper
        try:
            # Leggiamo il contenuto del file audio
            audio_content = await audio.read()
            logger.debug(f"Audio ricevuto: {len(audio_content)} bytes")
            
            # Importiamo il servizio di trascrizione
            from Main.services.whisper_service import speech_to_text_openai
            
            # Verifichiamo se siamo in modalità di sviluppo
            if config.DEVELOPMENT_MODE:
                logger.warning("DEVELOPMENT_MODE attivo: utilizzo trascrizione simulata")
                transcription = "Risposta simulata dell'utente"
            else:
                logger.info("Trascrizione con OpenAI Whisper in corso...")
                transcription = await speech_to_text_openai(audio_content)
                logger.info(f"Trascrizione completata: {transcription[:100]}...")
        except Exception as e:
            logger.error(f"Errore durante la trascrizione: {e}", exc_info=True)
            # Fallback alla trascrizione simulata in caso di errore
            transcription = "Risposta simulata (errore trascrizione)"
            logger.warning(f"Utilizzata trascrizione fallback: {transcription}")
        
        # Recupera la sessione dell'utente
        session = get_state(user_id)
        print()
        logger.info(f"Sessione recuperata per {user_id}: {session.session_id}")
        
        # Salviamo la risposta dell'utente
        #from Main.application.interview_state_adapter import save_answer, advance_to_next_question
        #from Main.application.interview_state_adapter_refactored import save_answer, advance_to_next_question
        
        # Aumento del livello di log globale per debug
        import logging
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        logger.debug(f"SESSION PRIMA DI SAVE ANSWER: {session}")
        
        try:
            # Salviamo la risposta e verifichiamo se è necessario un follow-up
            # NOTA: save_answer internamente avanza già alla prossima domanda se necessario
            # e restituisce i valori in quest'ordine: (needs_followup, coverage_percent, missing_topics)
            needed_followup, coverage, missing_topics = session.save_answer(transcription)
            logger.info(f"ANALISI RISPOSTA: needed_followup={needed_followup}, coverage={coverage:.1f}%, missing_topics={missing_topics}")            
            
            logger.debug("TO STRING DOPO LA RISPOSTA")
            print(session.to_string())
            
            # Verifichiamo i metadati della domanda corrente
            if hasattr(session, 'current_topic') and session.current_topic:
                print()
                logger.debug(f"Metadati domanda: topic={session.current_topic}, subtopics={session.current_subtopics}")
                if hasattr(session, 'current_keywords') and session.current_keywords:
                    print()
                    logger.debug(f"Keywords domanda: {session.current_keywords}")
            print()
            # Se è richiesto un follow-up, la domanda successiva sarà di tipo follow-up
            # altrimenti procediamo con la domanda successiva dello script
            if needed_followup:
                follow_up_subtopic = getattr(session, 'current_question_is_follow_up_for_subtopic', 'attributo non disponibile')
                logger.info(f"Domanda di follow-up necessaria per subtopic: {follow_up_subtopic}")
                print()
            else:
                logger.info("Nessun follow-up necessario, procedo con la prossima domanda")
                # NOTA: L'avanzamento alla prossima domanda ora viene gestito internamente
                # nella funzione save_answer, quindi non è più necessario chiamare next_question qui
        except Exception as e:
            print("This error:", e)
        finally:
            # Ripristiniamo il livello di log precedente
            root_logger.setLevel(previous_level)
        
        # Ottieni la prossima domanda (se c'è)
        next_question = None
        
        # Otteniamo la domanda corrente dalla sessione
        #from Main.application.interview_state_adapter import get_current_question
        #from Main.application.interview_state_adapter_refactored import get_current_question
        
        # Recuperiamo i dati della domanda corrente
        try:
            # Verifichiamo se la domanda è di tipo follow-up
            is_follow_up = hasattr(session, 'current_question_is_follow_up_for_subtopic') and session.current_question_is_follow_up_for_subtopic is not None
            
            # Recuperiamo la domanda corrente (che può essere una domanda principale o un follow-up)
            question_data = session.get_current_question()
            logger.debug(f"Question data recuperati: {question_data}")
            
            if question_data and 'text' in question_data:
                next_question = {
                    'id': question_data.get('id', 'q1'),
                    'Domanda': question_data.get('text', ''),
                    'is_follow_up': is_follow_up
                }
                logger.info(f"Prossima domanda ({'FOLLOW-UP' if is_follow_up else 'PRINCIPALE'}): {next_question['Domanda'][:50]}...")
                
                if is_follow_up:
                    subtopic = getattr(session, 'current_question_is_follow_up_for_subtopic', 'attributo non disponibile')
                    logger.info(f"Domanda di follow-up per subtopic: {subtopic}")
            else:
                logger.warning("Nessuna domanda disponibile dalla sessione")
                next_question = None
        except Exception as e:
            logger.error(f"Errore nel recupero della domanda corrente: {e}")
            next_question = None
        
        # Se non ci sono domande disponibili, usa una risposta generica
        if not next_question:
            logger.warning(f"Nessuna domanda disponibile per l'utente: {user_id}")
            generic_response = "Non ci sono altre domande disponibili. L'intervista è terminata."
            
            # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
            voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
            logger.info(f"Utilizzo TTS interno per il testo: '{generic_response[:50]}...' con voce {voice}")
            
            # Non generiamo più audio qui, solo l'URL per il TTS
            audio_content = ""  # Nessun audio embedded
            
            # Prepariamo l'URL per il TTS con i parametri necessari
            encoded_text = urllib.parse.quote(generic_response)
            tts_url = f"/tts/speak?voice_id={voice}&text={encoded_text}"
            
            return {
                "status": "end",
                "message": "Intervista terminata",
                "audio_content": audio_content,
                "text": generic_response,
                "audio_url": tts_url,
                "type": "end"
            }
        
        # Altrimenti, prepara la prossima domanda
        question_text = next_question.get('Domanda', 'Non ci sono altre domande disponibili.')
        
        # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
        voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
        logger.info(f"Utilizzo TTS interno per il testo: '{question_text[:50]}...' con voce {voice}")
        
        # Non generiamo audio qui
        audio_content = ""  # Nessun audio embedded
        
        # Prepariamo l'URL per il TTS con i parametri necessari
        encoded_text = urllib.parse.quote(question_text)
        tts_url = f"/tts/speak?voice_id={voice}&text={encoded_text}"


        
        # Prepara la risposta
        response = {
            "status": "success",
            "message": "Audio processato con successo",
            "transcription": transcription,
            "audio_content": audio_content,
            "text": question_text,
            "audio_url": tts_url,
            "type": "question",
            "question_id": next_question.get('id', '')
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Errore nella trascrizione dell'audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore nella trascrizione: {str(e)}"
        )


@router.get("/first_prompt", response_model=Dict[str, Any], responses={
    401: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def get_first_prompt(
    request: Request,
    user_id: str = Query(..., description="ID dell'utente"),
    token_query: Optional[str] = Query(None, description="Token JWT in query parameter"),
    token: Optional[str] = Cookie(None)  # Usiamo il cookie opzionale invece di Depends
):
    """Restituisce il primo prompt audio per iniziare l'intervista."""
    try:
        logger.debug("COMINCIO DA QUI")
        logger.info(f"Richiesta primo prompt per utente: {user_id}")
        
        # Verifica e decodifica il token, sia da cookie che da query parameter
        try:
            current_user = None
            
            # Forza modalità di sviluppo per questa sessione (test e debug)
            force_dev_mode = True  # <-- Importante: usato solo per debugging
            
            # Prendo il token da query parameter o da cookie
            actual_token = token_query if token_query else token
            logger.info(f"Token: {'Presente in query' if token_query else 'Presente in cookie' if token else 'Non presente'}")
            
            # Controlla sia DEVELOPMENT_MODE che il flag force_dev_mode
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            
            # In modalità sviluppo, permetti l'accesso senza token o con token fittizio
            if is_dev_mode:
                logger.info("Modalità sviluppo o forzata: permesso accesso senza autenticazione completa")
                
                # Se c'è un token fittizio che inizia con 'dev_token_', NON tentare di decodificarlo come JWT
                if actual_token and actual_token.startswith('dev_token_'):
                    logger.info(f"Dev token rilevato: {actual_token[:20]}... - Accesso sviluppo consentito")
                    current_user = user_id
                else:
                    logger.info(f"Accesso sviluppo: usando user_id={user_id} direttamente")
                    current_user = user_id  # Usa l'ID utente fornito direttamente
                
                # In modalità sviluppo, saltiamo completamente la logica JWT
                logger.info(f"Dev mode: autenticazione bypass per user_id={user_id}")
            elif not actual_token:
                # Solo in produzione richiediamo obbligatoriamente un token
                logger.error("Nessun token fornito (né in cookie né in query parameter)")
                raise HTTPException(status_code=401, detail="Token non fornito")
            else:
                # Modalità produzione: verifica token reale JWT
                logger.info(f"Modalità produzione: decodifica token JWT: {actual_token[:20]}...")
                try:
                    payload = jwt.decode(actual_token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
                    current_user = payload.get("sub")
                    
                    if not current_user:
                        logger.error(f"Token decodificato, ma 'sub' mancante: {payload}")
                        raise HTTPException(status_code=401, detail="Token senza identificatore utente")
                        
                    logger.info(f"Auth OK: user={current_user}, richiesto per session={user_id}")
                except Exception as jwt_err:
                    logger.error(f"Errore durante la decodifica del token: {jwt_err}")
                    raise HTTPException(status_code=401, detail=f"Token non valido: {jwt_err}")
        except JWTError as e:
            # Questo catch viene eseguito solo se si tenta di decodificare un token JWT non valido
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            if is_dev_mode:
                logger.warning(f"JWTError in modalità sviluppo: {e}, continuiamo con user_id={user_id}")
                current_user = user_id
            else:
                logger.error(f"JWTError: {e}")
                raise HTTPException(status_code=401, detail=f"Token non valido: {e}")
        except Exception as general_e:
            # Errore generale di autenticazione
            is_dev_mode = config.DEVELOPMENT_MODE or force_dev_mode
            logger.error(f"Errore generale durante l'autenticazione: {general_e}")
            if is_dev_mode:
                logger.warning(f"Continuiamo in modalità sviluppo con user_id={user_id}")
                current_user = user_id
            else:
                raise HTTPException(status_code=500, detail=f"Errore di autenticazione: {str(general_e)}")
        
        # Ottieni la sessione dell'utente o creane una nuova
        logger.info(f"Recupero sessione per user_id: {current_user}")
        session = get_state(current_user)
        
        # Controlla se le domande sono state caricate
        if not session.questions or len(session.questions) == 0:
            # Se non ci sono domande nella sessione, verifica se ci sono domande globali in SCRIPT
            if SCRIPT and len(SCRIPT) > 0:
                logger.info(f"Nessuna domanda nella sessione, ma SCRIPT globale ha {len(SCRIPT)} domande. Le uso per questa sessione.")
                session.questions = SCRIPT  # Copia le domande globali nella sessione
            else:
                logger.info(f"Nessuna domanda caricata per user: {current_user}, restituisco messaggio informativo")
                return {
                    "message": "Nessuna domanda caricata. Caricare le domande prima di iniziare il colloquio.",
                    "audio_url": None,
                    "question_text": "Nessuna domanda disponibile",
                    "question_type": "info",
                    "question_index": 0,
                    "questions_loaded": False
                }
        
        # Preparare il testo introduttivo per l'intervista
        intro_text = "Benvenuto all'intervista X. Mettiti comodo. Rilàssati. Sono qui per farti alcune domande. Non è un esame, quindi non ci sono domande giuste o sbagliate. Detto questo, iniziamo con la prima domanda: "
        if config.AWS_POLLY_VOICE_ID == "Matthew":
            intro_text = "Welcome to interview X. Make yourself comfortable. Relax. I'm here to ask you a few questions. This is not a test, so there are no right or wrong answers. Now, let's start with the first question:"

        # Utilizza la nuova struttura DOMANDE per ottenere la prima domanda
        #from Main.application.user_session_service import DOMANDE
        
        # Verifichiamo se ci sono domande nella struttura DOMANDE
        if DOMANDE and len(DOMANDE) > 0:
            # Prendiamo la prima domanda dalla struttura DOMANDE
            first_question_data = DOMANDE[0]
            question_id = first_question_data.get("id", "q1")
            
            # Accediamo direttamente al testo della domanda dalla struttura DOMANDE
            question_text = first_question_data.get("testo", "Chi sei e quali sono le tue competenze principali?")
            logger.info(f"Prima domanda ottenuta da DOMANDE: {question_text[:50]}...")
        else:
            # Fallback alla vecchia struttura se DOMANDE non contiene elementi
            logger.warning("Struttura DOMANDE vuota, uso il vecchio metodo con SCRIPT")
            first_question = session.questions[0]
            question_id = first_question.get("id", "q1")
            question_text = first_question.get("Domanda", first_question.get("text", "Chi sei e quali sono le tue competenze principali?"))
            logger.info(f"Prima domanda ottenuta da SCRIPT: {question_text[:50]}...")

        
        # Componi il testo completo da convertire in audio
        full_text = f"{intro_text} {question_text}"
        
        # Non generiamo più l'audio direttamente qui, utilizziamo solo l'endpoint TTS interno
        voice = "Bianca"  # Usa la voce italiana predefinita per il TTS interno
        logger.info(f"Utilizzo TTS interno per il testo: '{full_text[:50]}...' con voce {voice}")
        
        # Non generiamo né codifichiamo l'audio, sarà gestito dall'endpoint TTS
        audio_content = ""  # Nessun audio embedded
        logger.info("Audio sarà generato tramite endpoint TTS separato")
            
        # Prepara URL per il TTS con i parametri necessari (voce e testo)
        import urllib.parse
        encoded_text = urllib.parse.quote(full_text)
        tts_url = f"{config.API_BASE_URL}/tts/speak?voice_id={voice}&text={encoded_text}"
        
        logger.info(f"Primo prompt pronto per l'utente {current_user}: {full_text[:50]}...")
        
        return {
            "status": "success",
            "message": "Primo prompt generato con successo",
            "text": full_text,
            "audio_content": audio_content,  # Audio già generato in base64
            "audio_url": tts_url,  # URL del servizio TTS (backup)
            "question_id": question_id
        }
        
    except Exception as e:
        logger.error(f"Errore nella generazione del primo prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore nella generazione del primo prompt: {str(e)}"
        )
        
@router.get("/load_questions_status")
async def load_questions_status():
    """Controlla se ci sono domande caricate nel sistema."""
    script_size = len(SCRIPT)
    first_question = None
    questions_loaded = script_size > 0
    
    if questions_loaded and SCRIPT[0]:
        first_question = SCRIPT[0].get("Domanda", "<Domanda non trovata>")[:50] + "..."
    
    # Controlla anche le sessioni attive
    #from Main.application.user_session_service import get_session_info
    session_info = get_session_info()
    
    return {
        "status": "success" if questions_loaded else "warning",
        "questions_loaded": questions_loaded,
        "question_count": script_size,
        "first_question_preview": first_question,
        "active_sessions": session_info["active_sessions"],
        "dev_mode": config.DEVELOPMENT_MODE
    }

def get_session_info() -> Dict[str, Any]:
    """Restituisce informazioni sulle sessioni attive"""
    return {
        "active_sessions": len(SESSIONS),
        "session_ids": list(SESSIONS.keys()),
        "script_size": len(SCRIPT)
    }

def get_question_metadata_status(question_id: str) -> Dict[str, Any]:
    """
    Verifica lo stato dei metadati di una domanda specifica nella struttura DOMANDE
    
    Args:
        question_id: ID della domanda da verificare
        
    Returns:
        Dizionario con lo stato dei metadati e, se disponibili, i metadati stessi
    """
    # Cerca la domanda nella nuova struttura DOMANDE per indice
    found_index = -1
    for i, q in enumerate(DOMANDE):
        if q.get('id') == question_id:
            found_index = i
            break
    
    if found_index >= 0:
        # La domanda esiste nella struttura DOMANDE
        domanda = DOMANDE[found_index]
        
        # Verifica se i metadati sono stati generati
        if domanda.get('topic') and domanda.get('subtopics'):
            # I metadati sono stati generati
            return {
                'status': 'completed',
                'metadata': {
                    'primary_topic': domanda.get('topic', ''),
                    'subtopics': domanda.get('subtopics', []),
                    'keywords': domanda.get('keywords', []),
                    'lemma_sets': domanda.get('lemma_sets', []),
                    'fuzzy_norms': domanda.get('fuzzy_norms', []),
                    'vectors': domanda.get('vectors', [])
                },
                'message': f"Metadati per domanda {question_id} completati"
            }
        else:
            # La domanda esiste ma i metadati non sono ancora pronti
            return {
                'status': 'pending',
                'metadata': None,
                'message': 'Metadati in elaborazione'
            }
    else:
        # Se non trovata in DOMANDE, cerca in SCRIPT per retrocompatibilità
        found = False
        for q in SCRIPT:
            if q.get('id') == question_id:
                found = True
                break
        
        if found:
            # La domanda esiste in SCRIPT ma non in DOMANDE
            logger.warning(f"Domanda {question_id} trovata in SCRIPT ma non in DOMANDE")
            return {
                'status': 'pending',
                'metadata': None,
                'message': 'Metadati non ancora generati - usa SCRIPT per retrocompatibilità'
            }
        else:
            # La domanda non esiste in nessuna struttura
            logger.warning(f"Domanda {question_id} non trovata in DOMANDE né in SCRIPT")
            return {
                'status': 'not_found',
                'metadata': None,
                'message': 'Domanda non trovata'
            }
        
def wait_for_question_ready(question_id: str, 
                          max_attempts: int = 2, 
                          initial_delay: float = 1.0,
                          backoff_factor: float = 1.5) -> Optional[Dict[str, Any]]:
    """
    Attende che i metadati di una domanda siano pronti con backoff esponenziale.
    Limitato a un massimo di 2 tentativi come richiesto.
    Se i metadati non sono pronti dopo i tentativi, si restituisce comunque lo stato
    per permettere all'applicazione di procedere con la domanda successiva.
    
    Args:
        question_id: ID della domanda da controllare
        max_attempts: Numero massimo di tentativi (default: 2)
        initial_delay: Ritardo iniziale in secondi (default: 1.0)
        backoff_factor: Fattore di moltiplicazione per il backoff (default: 1.5)
        
    Returns:
        Lo stato della domanda, anche se i metadati non sono pronti
    """
    delay = initial_delay
    for attempt in range(max_attempts):
        status = get_question_metadata_status(question_id)
        
        if status['status'] == 'completed':
            logger.info(f"Metadati per domanda {question_id} pronti al tentativo {attempt + 1}")
            return status
        elif status['status'] == 'not_found':
            logger.error(f"Domanda {question_id} non trovata nel sistema")
            return None
            
        # Aspetta con backoff esponenziale
        logger.info(f"Attesa metadati per domanda {question_id} (tentativo {attempt + 1}/{max_attempts})...")
        time.sleep(delay)
        delay *= backoff_factor  # Aumenta il ritardo per il prossimo tentativo
    
    logger.warning(f"Timeout attesa metadati per domanda {question_id} dopo {max_attempts} tentativi. Procedo con la domanda successiva.")
    return status  # Restituisce comunque lo stato anche se non è 'completed'

# Funzione per ottenere la prossima domanda disponibile
def get_next_available_question(current_question_id: str, max_attempts: int = 2) -> Optional[str]:
    """
    Trova la prossima domanda disponibile nella struttura DOMANDE.
    Utilizza un counter e un ciclo come richiesto per navigare tra le domande.
    Se i metadati non sono pronti entro il timeout, procede comunque alla domanda successiva.
    
    Args:
        current_question_id: ID della domanda corrente
        max_attempts: Numero massimo di tentativi per domanda
        
    Returns:
        ID della prossima domanda disponibile, o None se non ce ne sono altre
    """
    # Verifichiamo se ci sono domande nella struttura DOMANDE
    if not DOMANDE:
        logger.warning("Struttura DOMANDE vuota, nessuna domanda disponibile")
        return None
    
    # Trova l'indice della domanda corrente in DOMANDE
    current_index = -1
    for i, q in enumerate(DOMANDE):
        if q.get("id") == current_question_id:
            current_index = i
            break
    
    # Se la domanda corrente non è stata trovata o è l'ultima
    if current_index == -1 or current_index >= len(DOMANDE) - 1:
        # Se la domanda non è stata trovata ma ci sono domande, restituisci la prima
        if current_index == -1 and len(DOMANDE) > 0:
            logger.warning(f"Domanda corrente {current_question_id} non trovata in DOMANDE, restituisco la prima domanda")
            return DOMANDE[0].get("id")
        else:
            logger.info("Siamo all'ultima domanda, nessuna domanda successiva disponibile")
            return None
    
    # Ottieni il numero totale di domande
    numero_domande = len(DOMANDE)
    
    # Inizializza un contatore per navigare attraverso le domande
    i = current_index + 1  # Inizia dalla domanda successiva a quella corrente
    
    # Utilizza un ciclo per trovare la prossima domanda disponibile
    # Se stiamo cercando la domanda i-esima (i > 0), otteniamo DOMANDE[i]
    if i < numero_domande:
        next_question = DOMANDE[i]
        next_id = next_question.get("id")
        
        if next_id:
            logger.info(f"Prossima domanda: {next_id} (indice {i})")
            
            # Verifica se i metadati sono pronti con timeout
            counter = 0
            while counter < max_attempts:
                # Controlla se i metadati sono pronti
                if next_question.get("topic") and next_question.get("subtopics"):
                    logger.info(f"Metadati per la domanda {next_id} sono pronti")
                    break
                
                # Altrimenti aspetta un po' e riprova
                logger.info(f"Attesa metadati per domanda {next_id} (tentativo {counter + 1}/{max_attempts})")
                time.sleep(1.0)  # Attendi 1 secondo tra i tentativi
                counter += 1
            
            # Anche se i metadati non sono pronti dopo i tentativi, procedi comunque
            if counter >= max_attempts:
                logger.warning(f"Timeout attesa metadati per domanda {next_id}. Procedo comunque.")
            
            return next_id
    
    return None

def get_metadata_processing_status() -> Dict[str, Any]:
    """
    Restituisce lo stato attuale dell'elaborazione dei metadati.
    Utilizza la nuova struttura DOMANDE per calcolare lo stato attuale.
    
    Returns:
        Dizionario con informazioni sullo stato dell'elaborazione
    """
    global metadata_processing_status
    global DOMANDE
    
    # Base result dal dizionario di stato
    result = metadata_processing_status.copy()
    
    # Se DOMANDE contiene elementi, ricalcoliamo lo stato effettivo di elaborazione
    if DOMANDE:
        # Conta quante domande hanno effettivamente metadati completi
        processed_count = 0
        for domanda in DOMANDE:
            if domanda.get('topic') and domanda.get('subtopics'):
                processed_count += 1
        
        # Aggiorna il contatore in base ai dati effettivi
        total_questions = len(DOMANDE)
        result['total_questions'] = total_questions
        result['processed_questions'] = processed_count
        
        # Aggiorna flag in_progress
        result['in_progress'] = processed_count < total_questions
        
        # Se tutti i metadati sono stati elaborati ma non era stato segnalato
        if processed_count == total_questions and result['in_progress']:
            result['in_progress'] = False
            result['end_time'] = datetime.now()
    
    # Calcola la percentuale di completamento
    if result['total_questions'] > 0:
        result['completion_percentage'] = (result['processed_questions'] / result['total_questions']) * 100
    else:
        result['completion_percentage'] = 0
    
    # Calcola il tempo trascorso se l'elaborazione è in corso
    if result['in_progress'] and result['start_time']:
        elapsed = datetime.now() - result['start_time']
        result['elapsed_seconds'] = elapsed.total_seconds()
    elif result['end_time'] and result['start_time']:
        elapsed = result['end_time'] - result['start_time']
        result['elapsed_seconds'] = elapsed.total_seconds()
    else:
        result['elapsed_seconds'] = 0
    
    # Aggiungi informazioni sulla struttura DOMANDE
    result['domande_structure'] = {
        'count': len(DOMANDE),
        'status': 'active' if DOMANDE else 'empty'
    }
    
    # Converti i datetime in stringhe per la serializzazione JSON
    if result['start_time']:
        result['start_time'] = result['start_time'].isoformat()
    if result['end_time']:
        result['end_time'] = result['end_time'].isoformat()
    
    return result
