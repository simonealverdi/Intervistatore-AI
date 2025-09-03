from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
import uvicorn
import atexit

# Configurazione centralizzata
from Main.core import config
from Main.services.persistence_service import dump_dev_storage_to_file

# Configurazione di logging centralizzata
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Imposta variabili di ambiente e path
os.environ['PYTHONPATH'] = os.path.join(os.getcwd())

# Salvataggio dei dati di sviluppo alla chiusura dell'applicazione
if config.DEVELOPMENT_MODE:
    logger.info("Modalità sviluppo attiva - i dati saranno salvati alla chiusura")
    atexit.register(dump_dev_storage_to_file)

# Importa i router dalla struttura a livelli
try:
    # Importa i router principali
    #from Main.api import  tts_router, questions_router, first_prompt_router, transcribe_router
    from Main.api import  tts_router, questions_router #, first_prompt_router, transcribe_router
    from Main.api.auth import router as auth_router
    # Importa interview_router dal nuovo modulo di compatibilità
    from Main.api.interview_router import router as interview_router
except ImportError as e:
    import sys
    print(f"Errore di importazione nei router API: {e}", file=sys.stderr)
    # Crea stub per evitare errori di avvio
    from fastapi import APIRouter
    auth_router = APIRouter()
    tts_router = APIRouter()
    questions_router = APIRouter()
    interview_router = APIRouter()
    #first_prompt_router = APIRouter()
    #transcribe_router = APIRouter()
# Verifica i router importati
print("\nVerifica router importati:")
print(f"auth_router endpoints: {[r.path for r in auth_router.routes]}")
print(f"questions_router endpoints: {[r.path for r in questions_router.routes]}")
# Inizializzazione dell'app FastAPI
app = FastAPI(
    title="AI Interview API",
    description="API per l'intervista condotta con intelligenza artificiale",
    version="1.0.0"
)

# Configurazione CORS - permissiva in modalità sviluppo, più restrittiva altrimenti
origins = ["*"] if config.DEVELOPMENT_MODE else [
    "http://localhost:3000",    # React app principale
    "http://localhost:8080",
    "http://localhost:5173",    # Eventuale alternativa
    "https://aiint.app"         # Dominio di produzione
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Log delle impostazioni principali
logger.info(f"Modalità sviluppo: {config.DEVELOPMENT_MODE}")
logger.info(f"MongoDB abilitato: {config.MONGODB_ENABLED}")
logger.info(f"CORS origins: {origins}")

# Registra i router
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(tts_router, prefix="/api/tts", tags=["tts"])
app.include_router(questions_router, prefix="/api/questions", tags=["questions"])
app.include_router(interview_router, prefix="/api/interview", tags=["interview"])
#app.include_router(interview_router, prefix="/api", tags=["first_prompt"])
#app.include_router(interview_router, prefix="/api/transcribe", tags=["transcribe"])

# Endpoint informativi
@app.get("/")
async def root():
    return {
        "name": "AI Interview API",
        "version": "1.0.0",
        "status": "online - in fase di sviluppo",
        "endpoints": [
            "/api/token - Autenticazione (username: admin, password: admin)",
            "/api/check-token - Verifica validità token",
            "/api/tts/speak - Sintesi vocale (versione semplificata)",
            "/api/tts/status - Stato del servizio TTS",
            "/api/questions/load - Carica domande (mock)",
            "/api/questions/list - Elenco domande disponibili",
            "/api/questions/random - Ottieni domanda casuale",
            "/api/questions/count - Numero di domande disponibili",
            "/api/interview/start - Avvia una nuova intervista",
            "/api/interview/next-question/{interview_id} - Ottieni prossima domanda",
            "/api/interview/submit-answer/{interview_id}/{question_id} - Invia risposta",
            "/api/interview/status/{interview_id} - Stato intervista",
            "/api/interview/end/{interview_id} - Termina intervista"
        ],
        "nota": "Implementazione incrementale - funzionalità in espansione"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Avvio del server
if __name__ == "__main__":
    try:
        # Assicurati che le directory necessarie esistano
        config.ensure_dir_exists(config.TTS_CACHE_DIR)
        config.ensure_dir_exists(config.AUDIO_CACHE_DIR)
        
        logger.info("🚀 Avvio AI Interview API...")
        logger.info(f"📍 URL: http://localhost:8000")
        logger.info(f"📖 Docs: http://localhost:8000/docs")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=config.DEVELOPMENT_MODE,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Arresto richiesto dall'utente")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Errore durante l'avvio: {e}")
        sys.exit(1)