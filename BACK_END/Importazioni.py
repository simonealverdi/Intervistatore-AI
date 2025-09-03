"""

Importa domande da vari formati (docx, csv, excel, json) **e** genera la
struttura di metadati `QuestionMeta` per ciascuna domanda, secondo la
road‑map definita nella chat (sub‑topic, keyword seed, soglie ecc.).



Esempio d'uso
-------------
>>> from question_importer import QuestionImporter
>>> metas = QuestionImporter.generate_metadata("domande.xlsx")
>>> QuestionImporter.save_yaml(metas, "qmeta.yaml")
"""


from __future__ import annotations

import json
import os
import re
import logging
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any

import numpy as np
import openai
import pandas as pd
import yaml
from docx import Document
import unidecode

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caricamento del processore NLP
# ---------------------------------------------------------------------------
# Importiamo la classe dal file che abbiamo creato in precedenza
from Main.services.nlp_services import NLPProcessor

logger.info("Inizializzazione del processore NLP...")
# Creiamo un'istanza globale che caricherà i modelli spaCy e SBERT
processor = NLPProcessor()
logger.info("Processore NLP pronto.")

# ---------------------------------------------------------------------------
# OpenAI config
# ---------------------------------------------------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06") # "gpt-4o-2024-08-06")
TEMPERATURE = float(os.getenv("OPENAI_TEMP", 0.4))
MAX_TOKENS = 450
MAX_RETRIES = 3

openai_client = openai.OpenAI()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class QuestionMeta(NamedTuple):
    prompt: str
    primary_topic: Optional[str]
    subtopics: List[str]
    keywords: List[List[str]]  # elenco parallelo ai subtopics
    lemma_sets: List[List[str]]  # set di lemmi per keyword
    fuzzy_norms: List[str]      # stringa normalizzata "kw1 kw2 …"
    vectors: List[List[float]]  # vettori (unit‑norm)
    difficulty: Optional[int] = None
    expected_answer_format: Optional[str] = None


class TopicMetaBuilder:
    """Costruisce campi derivati per un (subtopic, keywords)."""

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", unidecode.unidecode(text.lower().strip()))

    @classmethod
    @classmethod
    def build(cls, keywords: List[str]) -> tuple[list[str], str, list[float]]:
        """
        Usa NLPProcessor per derivare lemmi, stringa normalizzata e vettore.
        """
        text_to_process = " ".join(keywords)
        
        try:
            # 1. Usa il nostro processore per analizzare il testo
            analysis_result = processor.parse_text(text_to_process)
            
            # 2. Estrai i dati necessari dal risultato
            lemmas = list(set(tok["lemma"] for tok in analysis_result["tokens"]))
            vec = analysis_result["vector"]
            
            # 3. La normalizzazione fuzzy rimane una logica locale di questa classe
            norm_string = cls._normalise(text_to_process)

            logger.info(f"NLPProcessor utilizzato con successo per {len(keywords)} keywords")
            return lemmas, norm_string, vec

        except Exception as e:
            # Se il processore fallisce, abbiamo un problema serio. Logghiamo l'errore.
            logger.error(f"Errore irreversibile durante l'uso di NLPProcessor: {e}")
            # Fallback di emergenza: restituisce valori vuoti per non bloccare tutto
            norm_string = cls._normalise(text_to_process)
            return [], norm_string, [0.0] * 384 # Dimensione di SBERT, o altra dimensione standard

# ---------------------------------------------------------------------------
# JSON schema per la risposta LLM (immutato)
# ---------------------------------------------------------------------------
SCHEMA_BODY: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_topic": {"type": "string"},
        "subtopics": {"type": "array", "items": {"type": "string"}},
        "keywords": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        },
    },
    "required": ["primary_topic", "subtopics", "keywords"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "metadata", "schema": SCHEMA_BODY, "strict": True},
}

SYSTEM_MESSAGE = (
    "Sei un assistente che restituisce esclusivamente JSON valido, "
    "esclusivamente in italiano, "
    "senza testo aggiuntivo, conforme allo schema."
)

# ---------------------------------------------------------------------------
# Helper: business‑rules validation (immutato)
# ---------------------------------------------------------------------------

def _check_business_rules(data: Dict[str, Any]) -> bool:
    try:
        subs: List[str] = data["subtopics"]
        kw_lists: List[List[str]] = data["keywords"]
        #if not (3 <= len(subs) <= 8) or len(set(subs)) != len(subs):
        if not (2 <= len(subs) <= 8) or len(set(subs)) != len(subs):
            return False
        if len(kw_lists) != len(subs):
            return False
        seen: set[str] = set()
        for kws in kw_lists:
            # if len(set(kws)) != len(kws): # or not (len(kws)>=7):
            if (len(kws)>=7):
                return False
            if seen.intersection(kws):
                return False
            seen.update(kws)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Helper: LLM call with retry (immutato)
# ---------------------------------------------------------------------------

def _ask_llm(messages: List[Dict[str, str]]) -> str:
    resp = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format=RESPONSE_FORMAT,
    )
    return resp.choices[0].message.content.strip()


def _json_from_llm(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _ask_llm(messages)
            m = re.search(r"{.*}", raw, re.S)
            raw_json = m.group(0) if m else raw
            data = json.loads(raw_json)
            if _check_business_rules(data):
                return data
            raise ValueError("Violazione business rules")
            if attempt>=3:
                return data
        except Exception as exc:
            logger.warning(f"Tentativo {attempt}/{MAX_RETRIES} fallito: {exc}")
            if attempt == MAX_RETRIES:
                raise RuntimeError("LLM non ha prodotto output valido dopo i retry") from exc
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Output non valido: " f"{str(exc)}. Riformatta seguendo ESATTAMENTE lo schema."
                    ),
                }
            )
            time.sleep(0.5) #1.0)

# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class QuestionImporter:
    """Importa domande e genera metadati arricchiti."""

    # ----- Import file -------------------------------------------------
    @staticmethod
    def import_questions(file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        ext = Path(file_path).suffix.lower()
        if ext == ".docx":
            return QuestionImporter._from_docx(file_path)
        if ext == ".csv":
            return QuestionImporter._from_csv(file_path)
        if ext in (".xls", ".xlsx"):
            return QuestionImporter._from_excel(file_path)
        if ext == ".json":
            return QuestionImporter._from_json(file_path)
        raise ValueError(f"Unsupported file: {ext}")

    @staticmethod
    def _from_docx(path: str) -> List[str]:
        doc = Document(path)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    @staticmethod
    def _from_csv(path: str) -> List[str]:
        df = pd.read_csv(path)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    @staticmethod
    def _from_excel(path: str) -> List[str]:
        df = pd.read_excel(path)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    @staticmethod
    def _from_json(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            return [str(v) for v in data.values()]
        raise ValueError("JSON must contain array or object of strings")

    # ----- Metadata ----------------------------------------------------
    @staticmethod
    def generate_metadata(file_path: str) -> List[QuestionMeta]:
        questions = QuestionImporter.import_questions(file_path)
        metas: List[QuestionMeta] = []

        for q in questions:
            # 1️⃣ LLM: topic, subtopic, keywords ------------------------
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": (
                        f'Analizza questa domanda per un\'intervista: "{q}". '
                        "Identifica 1) primary_topic; 2) 2-8 subtopics; 3) più di 2 keyword uniche per subtopic. "
                        "Le keyword di un subtopic non devono sovrapporsi con quelle di altri."
                    ),
                },
            ]
            try:
                result = _json_from_llm(messages)
                primary = result["primary_topic"]
                subs = result["subtopics"]
                kw_lists = result["keywords"]
            except Exception as exc:
                logger.error(f"Fallback per '{q[:40]}...': {exc}")
                primary, subs, kw_lists = None, [], []

            # 2️⃣ Derivazione campi ------------------------------------
            lemma_sets, fuzzy_norms, vectors = [], [], []
            for kw in kw_lists:
                lemmas, norm_str, vec = TopicMetaBuilder.build(kw)
                lemma_sets.append(lemmas)
                fuzzy_norms.append(norm_str)
                vectors.append(vec)

            metas.append(
                QuestionMeta(
                    prompt=q,
                    primary_topic=primary,
                    subtopics=subs,
                    keywords=kw_lists,
                    lemma_sets=lemma_sets,
                    fuzzy_norms=fuzzy_norms,
                    vectors=vectors,
                )
            )
        return metas

    # ----- Save --------------------------------------------------------
    @staticmethod
    def save_yaml(metas: List[QuestionMeta], out_path: str | os.PathLike) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump([m._asdict() for m in metas], f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def save_json(metas: List[QuestionMeta], out_path: str | os.PathLike, indent: int = 2) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([m._asdict() for m in metas], f, ensure_ascii=False, indent=indent)
