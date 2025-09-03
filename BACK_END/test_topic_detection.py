import numpy as np
from topic_detection import topic_objects_from_meta, detect_covered_topics

def main():
    # Simulazione di metadati estratti dal YAML/QuestionImporter
    subtopics = ["saluto", "meteo", "orario"]
    keywords = [["ciao", "salve", "buongiorno"],
                ["meteo", "tempo", "previsioni"],
                ["orario", "che ore", "tempo"]]
    lemma_sets = [
        ["ciao", "salutare", "buongiorno"],
        ["meteo", "tempo", "previsione"],
        ["orario", "ora", "tempo"]
    ]
    fuzzy_norms = [
        "ciao salve buongiorno",
        "meteo tempo previsioni",
        "orario che ore tempo"
    ]
    # Generazione di vettori dummy normalizzati (300 dim)
    vectors = [
        np.random.rand(300).tolist(),
        np.random.rand(300).tolist(),
        np.random.rand(300).tolist()
    ]
    
    # Costruzione oggetti Topic
    topics = topic_objects_from_meta(subtopics, keywords, lemma_sets, fuzzy_norms, vectors)

    # Input utente
    user_input = input("Inserisci una frase: ")

    # Detection dei topic coperti
    covered, coverage = detect_covered_topics(user_input, topics)

    # Output risultato
    print("\n--- RISULTATI ---")
    print(f"Frase analizzata: {user_input}")
    print(f"Topics coperti: {covered}")
    print(f"Coverage: {coverage * 100:.2f}%")

if __name__ == "__main__":
    main()
