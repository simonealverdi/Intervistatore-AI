import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from openai import OpenAI


def extract_interview_data(file_path):
    """
    Estrae dati specifici dal file user.txt delle interviste
    """
    # Leggi il file CSV
    df = pd.read_csv(file_path)

    # Funzione per pulire i campi con liste
    def clean_list_field(field):
        if not isinstance(field, str):
            return []
        # Rimuove parentesi quadre e divide per pipe
        clean = field.replace('[', '').replace('|', '').replace(']', '').replace("'", "").split(',')
        # Rimuove elementi vuoti
        return [item.strip() for item in clean if item.strip()]

    # Pulisce i campi lista
    df['subtopics_clean'] = df['subtopics'].apply(clean_list_field)
    df['covered_subtopics_clean'] = df['covered_subtopics'].apply(clean_list_field)
    df['non_covered_subtopics_clean'] = df['non_covered_subtopics'].apply(clean_list_field)

    # Estrae i dati richiesti
    extracted_data = {
        'question_id': df['question_id'].tolist(),
        'question_text': df['question_text'].tolist(),
        'responses': df['response_text'].tolist(),
        'topic': df['topic'].tolist(),
        'subtopics': df['subtopics_clean'].tolist(),
        'covered_subtopics': df['covered_subtopics_clean'].tolist(),
        'coverage_percent': df['coverage_percent'].tolist()
    }

    # Crea un riassunto dei dati
    summary = {
        'num_responses': len(df),
        'unique_questions': df['question_id'].nunique(),
        'topics': df['topic'].unique().tolist(),
        'avg_coverage': df['coverage_percent'].mean(),
        'common_responses': df['response_text'].value_counts().head(3).to_dict()
    }

    return extracted_data, summary

def responses_resume(data):
    """
    Crea un riassunto delle risposte. Prima fa una frase di riepilogo e poi fa un riassunto più dettagliato.
    """
    total_responses = " ".join(data['responses'])

    # Usa OpenAI per generare un riassunto
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Sei un assistente esperto in analisi delle conversazioni. "
                "Il tuo compito è riassumere in modo chiaro e oggettivo le risposte dell’utente, "
                "evidenziando i temi principali, eventuali opinioni e problemi menzionati."
            )},
            {"role": "user", "content": f"Elabora un riassunto breve e informativo delle seguenti risposte. Usa frasi complete e dividi il testo in massimo 3 punti chiave:\n\n{total_responses}"}

        ],
        max_tokens=150,
        temperature=0.3
    )

    return response

# Esempio di utilizzo
if _name_ == "_main_":
    data, summary = extract_interview_data("user.txt")

    # Stampa analisi
    print("===== ANALISI DATI INTERVISTA =====")
    print(f"Riassunto dell'intervista: {responses_resume(data)}")
    print(f"Numero totale risposte: {summary['num_responses']}")
    print(f"Domande uniche: {summary['unique_questions']}")
    print(f"Topic: {', '.join(summary['topics'])}")
    print(f"Copertura media: {summary['avg_coverage']:.1f}%")

    print("\nRisposte più comuni:")
    for resp, count in summary['common_responses'].items():
        print(f"- '{resp}' ({count} volte)")

    # Visualizza distribuzione risposte
    plt.figure(figsize=(10, 6))
    plt.hist(data['coverage_percent'], bins=5, color='skyblue', edgecolor='black')
    plt.title('Distribuzione Percentuale di Copertura')
    plt.xlabel('Percentuale di Copertura')
    plt.ylabel('Numero di Risposte')
    plt.tight_layout()
    plt.savefig('coverage_analysis.png')

    # Stampa esempio di dati estratti
    print("\n===== ESEMPIO DATI ESTRATTI =====")
    for i in range(min(3, len(data['responses']))):
        print(f"\nRisposta {i+1}:")
        print(f"ID Domanda: {data['question_id'][i]}")
        print(f"Risposta: {data['responses'][i]}")
        print(f"Topic: {data['topic'][i]}")
        print(f"Subtopics: {', '.join(data['subtopics'][i])}")
        print(f"Subtopics Coperti: {', '.join(data['covered_subtopics'][i])}")
        print(f"Percentuale Copertura: {data['coverage_percent'][i]}%")
