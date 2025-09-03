#!/usr/bin/env python3
"""
Test suite per topic_detection.py
Test progressivi del sistema NLP locale (spaCy + SBERT)

Istruzioni:
1. Decommenta i test gradualmente
2. Esegui con: python test_topic_det.py
3. Verifica che i modelli NLP siano installati
"""
import unittest
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np


# Aggiungi il percorso per importare i moduli
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from topic_detection import (
        TopicMetaBuilder, 
        detect_covered_topics, 
        topic_objects_from_meta,
        Topic,
        topic_from_meta
    )
    from Main.services.nlp_services import NLPProcessor
except ImportError as e:
    print(f"❌ Errore import: {e}")
    print("Assicurati di essere nella cartella BACK_END")
    sys.exit(1)

class TestTopicDetection(unittest.TestCase):
    """Test suite per la topic detection"""
    
    @classmethod
    def setUpClass(cls):
        """Setup una volta per tutta la classe di test"""
        print("\n🚀 Inizializzazione test Topic Detection...")
        
        # Testi di esempio in italiano
        cls.sample_texts = {
            'famiglia': "La mia famiglia è composta da quattro persone: mio padre, mia madre, mio fratello e io. Viviamo in una casa accogliente.",
            'lavoro': "Lavoro come sviluppatore software in un'azienda di tecnologia. Mi occupo principalmente di Python e JavaScript.",
            'hobby': "Nel tempo libero mi piace leggere libri, guardare film e fare escursioni in montagna con gli amici.",
            'vuoto': "",
            'lungo': "La tecnologia moderna ha rivoluzionato il modo in cui lavoriamo e comunichiamo. " * 50,
            'caratteri_speciali': "Ciao! Come stai? Tutto bene... spero di sì 😊 @#$%"
        }
        
        # Metadati di esempio per test
        cls.test_metadata = {
            'subtopics': ['famiglia', 'lavoro', 'hobby'],
            'keywords': [
                ['famiglia', 'genitori', 'casa', 'fratello'],
                ['lavoro', 'sviluppatore', 'programmazione', 'tecnologia'], 
                ['hobby', 'tempo libero', 'lettura', 'film']
            ],
            'lemma_sets': [
                ['famiglia', 'genitore', 'casa', 'fratello'],
                ['lavoro', 'sviluppatore', 'programmazione', 'tecnologia'],
                ['hobby', 'tempo', 'libero', 'lettura', 'film']
            ],
            'fuzzy_norms': [
                'famiglia genitori casa fratello',
                'lavoro sviluppatore programmazione tecnologia', 
                'hobby tempo libero lettura film'
            ],
            'vectors': []  # Sarà popolato nei test
        }

    def setUp(self):
        """Setup per ogni singolo test"""
        self.start_time = time.time()
    
    def tearDown(self):
        """Cleanup dopo ogni test"""
        elapsed = time.time() - self.start_time
        print(f"⏱️ Test completato in {elapsed:.2f}s")

    # ========================================================================
    # TEST LEVEL 1: UNIT TEST - Decommenta questi per iniziare
    # ========================================================================
    
#     def test_01_nlp_processor_initialization(self):
#         """Test 1: Verifica inizializzazione NLPProcessor"""
#         print("\n🧪 Test 1: Inizializzazione NLPProcessor")
        
#         try:
#             processor = NLPProcessor()
#             self.assertIsNotNone(processor, "NLPProcessor non inizializzato")
            
#             # Verifica disponibilità dei modelli
#             print(f"✅ spaCy disponibile: {processor.nlp is not None}")
#             print(f"✅ SBERT disponibile: {processor.sbert is not None}")
            
#             # Almeno uno dei due modelli deve essere disponibile
#             self.assertTrue(
#                 processor.nlp is not None or processor.sbert is not None,
#                 "Nessun modello NLP disponibile"
#             )
            
#         except Exception as e:
#             self.fail(f"Errore inizializzazione NLPProcessor: {e}")
    
#     def test_02_basic_text_parsing(self):
#         """Test 2: Parsing base del testo"""
#         print("\n🧪 Test 2: Parsing base del testo")
        
#         try:
#             processor = NLPProcessor()
#             text = self.sample_texts['famiglia']
            
#             result = processor.parse_text(text)
            
#             # Verifica struttura risultato
#             self.assertIn('tokens', result)
#             self.assertIn('entities', result) 
#             self.assertIn('vector', result)
            
#             # Verifica contenuto
#             self.assertIsInstance(result['tokens'], list)
#             self.assertIsInstance(result['entities'], list)
#             self.assertIsInstance(result['vector'], list)
            
#             print(f"✅ Token estratti: {len(result['tokens'])}")
#             print(f"✅ Entità trovate: {len(result['entities'])}")
#             print(f"✅ Dimensione vettore: {len(result['vector'])}")
            
#         except Exception as e:
#             self.fail(f"Errore parsing testo: {e}")
    
#     def test_03_lemma_extraction(self):
#         """Test 3: Estrazione lemmi"""
#         print("\n🧪 Test 3: Estrazione lemmi")
        
#         for name, text in self.sample_texts.items():
#             if text:  # Salta testi vuoti per questo test
#                 with self.subTest(text_type=name):
#                     try:
#                         processor = NLPProcessor()
#                         result = processor.parse_text(text)
                        
#                         lemmas = [token['lemma'] for token in result['tokens']]
#                         self.assertIsInstance(lemmas, list)
                        
#                         print(f"✅ {name}: {len(lemmas)} lemmi estratti")
#                         if lemmas:
#                             print(f"   Esempio lemmi: {lemmas[:5]}")
                        
#                     except Exception as e:
#                         self.fail(f"Errore estrazione lemmi per {name}: {e}")

#     def test_04_vector_generation(self):
#         """Test 4: Generazione vettori"""
#         print("\n🧪 Test 4: Generazione vettori")
        
#         try:
#             processor = NLPProcessor()
#             text = self.sample_texts['lavoro']
            
#             result = processor.parse_text(text)
#             vector = result['vector']
            
#             # Verifica vettore
#             self.assertIsInstance(vector, list)
#             self.assertGreater(len(vector), 0, "Vettore vuoto")
            
#             # Verifica che non sia tutto zero
#             non_zero = sum(1 for v in vector if abs(v) > 1e-6)
#             self.assertGreater(non_zero, 0, "Vettore tutto zero")
            
#             print(f"✅ Dimensione vettore: {len(vector)}")
#             print(f"✅ Elementi non-zero: {non_zero}/{len(vector)}")
#             print(f"✅ Range valori: [{min(vector):.3f}, {max(vector):.3f}]")
            
#         except Exception as e:
#             self.fail(f"Errore generazione vettore: {e}")

#     # ========================================================================
#     # TEST LEVEL 2: INTEGRATION TEST - Decommenta dopo il Level 1
#     # ========================================================================
    
#     def test_05_topic_meta_builder(self):
#         """Test 5: TopicMetaBuilder con NLP reale"""
#         print("\n🧪 Test 5: TopicMetaBuilder")
        
#         try:
#             keywords = ['famiglia', 'genitori', 'casa', 'fratello']
            
#             lemmas, norm_string, vector = TopicMetaBuilder.build(keywords)
            
#             # Verifica output
#             self.assertIsInstance(lemmas, list)
#             self.assertIsInstance(norm_string, str)
#             self.assertIsInstance(vector, list)
            
#             print(f"✅ Lemmi estratti: {lemmas}")
#             print(f"✅ Stringa normalizzata: '{norm_string}'")
#             print(f"✅ Dimensione vettore: {len(vector)}")
            
#         except Exception as e:
#             self.fail(f"Errore TopicMetaBuilder: {e}")
    
#     def test_06_topic_objects_from_meta(self):
#         """Test 6: Creazione oggetti Topic dai metadati"""
#         print("\n🧪 Test 6: Creazione oggetti Topic")
        
#         try:
#             # Prima genera alcuni vettori di test
#             processor = NLPProcessor()
#             test_vectors = []
#             for keywords in self.test_metadata['keywords']:
#                 result = processor.parse_text(' '.join(keywords))
#                 # Converti vettore a lista
#                 if isinstance(result['vector'], np.ndarray):
#                     vector_as_list = result['vector'].tolist()
#                 else:
#                     vector_as_list = list(result['vector'])
#                 test_vectors.append(vector_as_list)
            
#             # Aggiorna metadati con vettori reali
#             metadata = self.test_metadata.copy()
#             metadata['vectors'] = test_vectors
            
#             # Crea oggetti Topic
#             topics = topic_objects_from_meta(
#                 subtopics=metadata['subtopics'],
#                 keywords=metadata['keywords'],
#                 lemma_sets=metadata['lemma_sets'],
#                 fuzzy_norms=metadata['fuzzy_norms'],
#                 vectors=metadata['vectors']
#             )
            
#             # Verifica base
#             self.assertIsInstance(topics, list)
#             self.assertEqual(len(topics), len(metadata['subtopics']))
            
#             # Verifica ogni Topic
#             for i, topic in enumerate(topics):
#                 self.assertEqual(topic.name, metadata['subtopics'][i])
#                 self.assertIsInstance(topic.keywords, list)
                
#                 # Converti topic.vector a lista
#                 if isinstance(topic.vector, np.ndarray):
#                     topic_vector_as_list = topic.vector.tolist()
#                 else:
#                     topic_vector_as_list = list(topic.vector) if topic.vector else []
                
#                 # Verifica finale
#                 self.assertIsInstance(topic_vector_as_list, list)
#                 self.assertGreater(len(topic_vector_as_list), 0)
                
#             print(f"✅ {len(topics)} oggetti Topic creati")
            
#         except Exception as e:
#             self.fail(f"Errore creazione oggetti Topic: {e}")
    
#     # ========================================================================
#     # TEST LEVEL 3: PERFORMANCE TEST - Decommenta per ultimo
#     # ========================================================================
    
#     def test_07_performance_long_text(self):
#         """Test 7: Performance su testo lungo"""
#         print("\n🧪 Test 7: Performance testo lungo")
        
#         try:
#             processor = NLPProcessor()
#             text = self.sample_texts["long"]
            
#             start_time = time.time()
#             result = processor.parse_text(text)
#             elapsed = time.time() - start_time
            
#             # Verifica che non sia troppo lento (soglia: 5 secondi)
#             self.assertLess(elapsed, 5.0, f"Processing troppo lento: {elapsed:.2f}s")
            
#             print(f"✅ Testo di {len(text)} caratteri processato in {elapsed:.2f}s")
#             print(f"✅ Token estratti: {len(result['tokens'])}")
            
#         except Exception as e:
#             self.fail(f"Errore test performance: {e}")
    
#     def test_08_edge_cases(self):
#         """Test 8: Casi limite"""
#         print("\n🧪 Test 8: Casi limite")
        
#         edge_cases = {
#             'vuoto': "",
#             'solo_spazi': "   \n\t   ",
#             'solo_punteggiatura': "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./",
#             'numeri': "123 456 789 2024",
#             'molto_corto': "Ciao",
#             'caratteri_speciali': self.sample_texts['caratteri_speciali']
#         }
        
#         processor = NLPProcessor()
        
#         for case_name, text in edge_cases.items():
#             with self.subTest(case=case_name):
#                 try:
#                     result = processor.parse_text(text)
                    
#                     # Non deve crashare, ma può restituire risultati vuoti
#                     self.assertIsInstance(result, dict)
#                     self.assertIn('tokens', result)
#                     self.assertIn('vector', result)
                    
#                     print(f"✅ {case_name}: OK ({len(result['tokens'])} token)")
                    
#                 except Exception as e:
#                     print(f"⚠️ {case_name}: {e}")
#                     # Non fail per casi limite, solo warning

#     def test_09_detect_covered_topics_integration(self):
#         """Test 9: Test completo detect_covered_topics"""
#         print("\n🧪 Test 9: Topic detection completa")
        
#         try:
#             # Prepara metadati con vettori reali
#             processor = NLPProcessor()
#             metadata = self.test_metadata.copy()
#             metadata['vectors'] = []
            
#             for keywords in metadata['keywords']:
#                 result = processor.parse_text(' '.join(keywords))
#                 metadata['vectors'].append(result['vector'])
            
#             # Crea oggetti Topic
#             topics = topic_objects_from_meta(
#                 subtopics=metadata['subtopics'],
#                 keywords=metadata['keywords'],
#                 lemma_sets=metadata['lemma_sets'],
#                 fuzzy_norms=metadata['fuzzy_norms'],
#                 vectors=metadata['vectors']
#             )
            
#             # Test detection con testo sulla famiglia
#             text = self.sample_texts['famiglia']
#             covered, coverage = detect_covered_topics(text, topics)
            
#             # Verifica risultati
#             self.assertIsInstance(covered, set)
#             self.assertIsInstance(coverage, float)
#             self.assertGreaterEqual(coverage, 0.0)
#             self.assertLessEqual(coverage, 1.0)
            
#             print(f"✅ Topic coperti: {covered}")
#             print(f"✅ Coverage: {coverage:.2%}")
            
#             # Il testo sulla famiglia dovrebbe coprire il topic 'famiglia'
#             if 'famiglia' in [t.name for t in topics]:
#                 self.assertIn('famiglia', covered, "Topic 'famiglia' non rilevato")
            
#         except Exception as e:
#             self.fail(f"Errore test integrazione completa: {e}")

# # ...existing code...

#     def test_10_topic_matching_long_sentence(self):
#         """Test 7: Identificazione topic in frase lunga e complessa"""
#         print("\n🧪 Test 7: Topic Matching - Frase Lunga")
        
#         try:
#             # Lista di topic predefiniti
#             predefined_topics = [
#                 "infanzia",
#                 "matrimonio", 
#                 "lavoro",
#                 "famiglia",
#                 "educazione",
#                 "salute",
#                 "viaggi",
#                 "hobby",
#                 "casa",
#                 "tecnologia"
#             ]
            
#             # Frase lunga che contiene multiple indicazioni di topic
#             long_sentence = """
#             Dopo aver completato gli studi universitari in ingegneria informatica, 
#             ho iniziato a lavorare come sviluppatore software in una grande azienda tecnologica. 
#             Nel 2018 mi sono sposato con mia moglie Sarah e abbiamo comprato una casa 
#             in periferia dove viviamo felicemente con i nostri due figli. 
#             Durante il tempo libero mi piace viaggiare e praticare fotografia, 
#             mentre da bambino ero appassionato di videogiochi e robotica. 
#             Recentemente ho iniziato a seguire un corso di specializzazione 
#             per migliorare le mie competenze professionali e prendermi cura 
#             della salute mentale attraverso la meditazione.
#             """
            
#             # Topic attesi da trovare nella frase (in ordine di rilevanza)
#             expected_topics = [
#                 "lavoro",      # sviluppatore, azienda, competenze professionali
#                 "educazione",  # studi universitari, corso specializzazione  
#                 "matrimonio",  # sposato, moglie
#                 "famiglia",    # figli, casa
#                 "tecnologia"   # informatica, software, videogiochi
#             ]
            
#             processor = NLPProcessor()
            
#             # Genera vettori per i topic predefiniti
#             print("🔄 Generazione vettori topic...")
#             topic_vectors = {}
#             for topic in predefined_topics:
#                 result = processor.parse_text(topic)
#                 # Converti a lista
#                 if isinstance(result['vector'], np.ndarray):
#                     vector = result['vector'].tolist()
#                 else:
#                     vector = list(result['vector'])
#                 topic_vectors[topic] = vector
            
#             # Ottieni vettore della frase lunga
#             print("🔄 Analisi frase lunga...")
#             sentence_result = processor.parse_text(long_sentence)
#             if isinstance(sentence_result['vector'], np.ndarray):
#                 sentence_vector = sentence_result['vector'].tolist()
#             else:
#                 sentence_vector = list(sentence_result['vector'])
            
#             # Calcola similarità con ogni topic
#             similarities = {}
#             for topic, topic_vector in topic_vectors.items():
#                 similarity = self._cosine_similarity(sentence_vector, topic_vector)
#                 similarities[topic] = similarity
            
#             # Ordina topic per similarità (dal più alto al più basso)
#             sorted_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
#             print(f"\n📊 Risultati Topic Matching (Top 5):")
#             top_5_topics = []
#             for i, (topic, score) in enumerate(sorted_topics[:5]):
#                 print(f"{i+1}. {topic}: {score:.3f}")
#                 top_5_topics.append(topic)
            
#             # Verifica che almeno 3 dei topic attesi siano nei top 5
#             found_expected = sum(1 for topic in expected_topics if topic in top_5_topics)
            
#             print(f"\n✅ Topic attesi trovati nei top 5: {found_expected}/{len(expected_topics)}")
#             print(f"📝 Topic attesi: {expected_topics}")
#             print(f"🎯 Topic trovati: {top_5_topics}")
            
#             # Verifica che il miglior match abbia una similarità ragionevole
#             best_topic, best_score = sorted_topics[0]
#             self.assertGreater(best_score, 0.2, 
#                              f"Similarità del miglior match troppo bassa: {best_score:.3f}")
            
#             # Verifica che almeno la metà dei topic attesi sia nei top 5
#             min_expected = len(expected_topics) // 2
#             self.assertGreaterEqual(found_expected, min_expected,
#                                   f"Trovati solo {found_expected} topic attesi su {len(expected_topics)}")
            
#             # Analisi aggiuntiva: verifica distribuzione delle similarità
#             scores = [score for _, score in sorted_topics]
#             avg_score = sum(scores) / len(scores)
#             max_score = max(scores)
#             min_score = min(scores)
            
#             print(f"\n📈 Statistiche similarità:")
#             print(f"   Max: {max_score:.3f} | Avg: {avg_score:.3f} | Min: {min_score:.3f}")
#             print(f"   Range: {max_score - min_score:.3f}")
            
#             # Verifica che ci sia una buona differenziazione tra topic
#             self.assertGreater(max_score - min_score, 0.1,
#                              "Differenziazione tra topic insufficiente")
            
#             print(f"✅ Test completato: frase di {len(long_sentence.split())} parole analizzata")
            
#         except Exception as e:
#             self.fail(f"Errore nel topic matching frase lunga: {e}")

#     def _cosine_similarity(self, vec1, vec2):
#         """Calcola la similarità coseno tra due vettori"""
#         import math
        
#         # Calcola il prodotto scalare
#         dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
#         # Calcola le norme
#         norm1 = math.sqrt(sum(a * a for a in vec1))
#         norm2 = math.sqrt(sum(b * b for b in vec2))
        
#         # Evita divisione per zero
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
        
#         return dot_product / (norm1 * norm2)

    # ...existing code...

    # def test_13_adaptive_topic_detection(self):
    #     """Test 13: Verifica adaptive_topic_detection con soglie dinamiche"""
    #     print("\n🧪 Test 13: Adaptive Topic Detection")
        
    #     try:
    #         # Importa la nuova funzione
    #         from topic_detection import adaptive_topic_detection
            
    #         # Setup dati di test
    #         test_data = {
    #             'subtopics': ['programmazione', 'matrimonio', 'educazione', 'viaggi'],
    #             'keywords': [
    #                 ['python', 'codice', 'sviluppo', 'software'],
    #                 ['sposare', 'moglie', 'marito', 'cerimonia'],
    #                 ['università', 'studio', 'laurea', 'corso'],
    #                 ['vacanza', 'aereo', 'hotel', 'turismo']
    #             ]
    #         }
            
    #         # Genera metadati
    #         print("🔄 Generazione metadati...")
    #         lemma_sets, fuzzy_norms, vectors = [], [], []
    #         for keywords in test_data['keywords']:
    #             lemmas, fuzzy_norm, vector = TopicMetaBuilder.build(keywords)
    #             lemma_sets.append(lemmas)
    #             fuzzy_norms.append(fuzzy_norm)
    #             vectors.append(vector)
            
    #         # Crea oggetti Topic
    #         topics = topic_objects_from_meta(
    #             subtopics=test_data['subtopics'],
    #             keywords=test_data['keywords'],
    #             lemma_sets=lemma_sets,
    #             fuzzy_norms=fuzzy_norms,
    #             vectors=vectors
    #         )
            
    #         print(f"✅ Creati {len(topics)} topic per il test")
            
    #         # Test cases con diversi tipi di testo
    #         test_cases = [
    #             {
    #                 'text': 'Studio università',  # Testo CORTO (2 parole)
    #                 'expected_in': {'educazione'},
    #                 'description': 'Testo corto - soglie più permissive'
    #             },
    #             {
    #                 'text': 'Mi sono sposato e ho studiato programmazione Python per il lavoro',  # Testo MEDIO (12 parole)
    #                 'expected_in': {'matrimonio', 'educazione', 'programmazione'},
    #                 'description': 'Testo medio - soglie standard'
    #             },
    #             {
    #                 'text': '''Dopo aver completato gli studi universitari in informatica, 
    #                         ho iniziato a lavorare come sviluppatore Python. Nel 2020 mi sono 
    #                         sposato con mia moglie e abbiamo fatto un viaggio di nozze in Giappone. 
    #                         Adesso programmo software per una grande azienda tecnologica.''',  # Testo LUNGO (40+ parole)
    #                 'expected_in': {'educazione', 'programmazione', 'matrimonio', 'viaggi'},
    #                 'description': 'Testo lungo - soglie più rigorose'
    #             },
    #             {
    #                 'text': 'Programmo software, mi sono sposato e ho studiato all università',
    #                 'expected_in': {'programmazione', 'matrimonio', 'educazione'},
    #                 'description': 'MAtch incriminato'
    #             }
    #         ]
            
    #         print("\n🔄 Confronto adaptive vs original...")
            
    #         for i, case in enumerate(test_cases):
    #             text = case['text']
    #             expected = case['expected_in']
    #             description = case['description']
                
    #             print(f"\n📝 Test Case {i+1}: {description}")
    #             print(f"   Testo: {len(text.split())} parole")
                
    #             # Test con versione originale
    #             covered_original, coverage_original = detect_covered_topics(text, topics)
                
    #             # Test con versione adaptive
    #             covered_adaptive, coverage_adaptive = adaptive_topic_detection(text, topics)
                
    #             print(f"   📊 Original:  {covered_original} | Coverage: {coverage_original:.2%}")
    #             print(f"   📊 Adaptive:  {covered_adaptive} | Coverage: {coverage_adaptive:.2%}")
                
    #             # Verifica che adaptive funzioni
    #             self.assertIsInstance(covered_adaptive, set)
    #             self.assertIsInstance(coverage_adaptive, float)
    #             self.assertGreaterEqual(coverage_adaptive, 0.0)
    #             self.assertLessEqual(coverage_adaptive, 1.0)
                
    #             # Verifica che adaptive sia almeno buono quanto original
    #             self.assertGreaterEqual(len(covered_adaptive), len(covered_original),
    #                                   f"Adaptive dovrebbe trovare almeno quanti topic di original")
                
    #             # Verifica che almeno alcuni topic attesi siano trovati
    #             found_expected = len(expected.intersection(covered_adaptive))
    #             self.assertGreater(found_expected, 0,
    #                              f"Adaptive dovrebbe trovare almeno un topic atteso: {expected}")
                
    #             # Confronto miglioramento
    #             improvement = len(covered_adaptive) - len(covered_original)
    #             if improvement > 0:
    #                 print(f"   ✅ Adaptive trova {improvement} topic in più")
    #             elif improvement == 0:
    #                 print(f"   ⚖️ Stesso risultato")
    #             else:
    #                 print(f"   ⚠️ Adaptive trova {abs(improvement)} topic in meno")
            
    #         # Test soglie specifiche
    #         print("\n🔄 Test soglie adattive specifiche...")
            
    #         # Testo corto - dovrebbe usare soglie più basse
    #         short_text = "studio"
    #         covered_short, _ = adaptive_topic_detection(short_text, topics)
    #         print(f"   Testo corto (1 parola): {covered_short}")
            
    #         # Molti topic - dovrebbe usare soglie più alte  
    #         many_topics = topics * 2  # Duplica per avere 8 topic
    #         covered_many, _ = adaptive_topic_detection("programmazione università", many_topics)
    #         print(f"   Molti topic (8): {len(covered_many)} trovati")
            
    #         print(f"✅ Test adaptive completato con successo")
            
    #     except ImportError as e:
    #         if "adaptive_topic_detection" in str(e):
    #             self.fail("❌ adaptive_topic_detection non implementata in topic_detection.py")
    #         else:
    #             self.fail(f"❌ Errore import: {e}")
    #     except Exception as e:
    #         self.fail(f"❌ Errore nel test adaptive: {e}")



#     def test_11_topic_objects_and_detection(self):
#         """Test 11: Verifica topic_objects_from_meta e detect_covered_topics"""
#         print("\n🧪 Test 11: Topic Objects e Detection")
        
#         try:
#             # Dati di test strutturati
#             test_data = {
#                 'subtopics': ['programmazione', 'matrimonio', 'educazione', 'viaggi'],
#                 'keywords': [
#                     ['python', 'codice', 'sviluppo', 'software'],
#                     ['sposare', 'moglie', 'marito', 'cerimonia'],
#                     ['università', 'studio', 'laurea', 'corso'],
#                     ['vacanza', 'aereo', 'hotel', 'turismo']
#                 ],
#                 'lemma_sets': [],  # Sarà popolato dinamicamente
#                 'fuzzy_norms': [],  # Sarà popolato dinamicamente
#                 'vectors': []      # Sarà popolato dinamicamente
#             }
            
#             print("🔄 Generazione metadati con TopicMetaBuilder...")
            
#             # Genera metadati per ogni topic usando TopicMetaBuilder
#             for keywords in test_data['keywords']:
#                 lemmas, fuzzy_norm, vector = TopicMetaBuilder.build(keywords)
#                 test_data['lemma_sets'].append(lemmas)
#                 test_data['fuzzy_norms'].append(fuzzy_norm)
#                 test_data['vectors'].append(vector)
            
#             print(f"✅ Metadati generati per {len(test_data['subtopics'])} topic")
            
#             # ===============================
#             # TEST topic_objects_from_meta
#             # ===============================
#             print("\n🔄 Test topic_objects_from_meta...")
            
#             topics = topic_objects_from_meta(
#                 subtopics=test_data['subtopics'],
#                 keywords=test_data['keywords'],
#                 lemma_sets=test_data['lemma_sets'],
#                 fuzzy_norms=test_data['fuzzy_norms'],
#                 vectors=test_data['vectors']
#             )
#  #           print(f"   Generati {len(topics)} oggetti Topic \n campione tipo: {topics[0]} \n tipo: {type(topics[0])}")
            
#             # Verifica base
#             self.assertIsInstance(topics, list)
#             self.assertEqual(len(topics), len(test_data['subtopics']))

#             # Verifica ogni Topic creato
#             for i, topic in enumerate(topics):
#                 self.assertIsInstance(topic, Topic)
#                 self.assertEqual(topic.name, test_data['subtopics'][i])
#                 self.assertIsInstance(topic.keywords, list)
#                 self.assertIsInstance(topic.lemma_set, set)
#                 self.assertIsInstance(topic.fuzzy_norm, str)
#                 self.assertIsInstance(topic.vector, np.ndarray)
                
#                 # Verifica contenuto
#                 self.assertEqual(topic.keywords, test_data['keywords'][i])
#                 self.assertEqual(topic.lemma_set, set(test_data['lemma_sets'][i]))
#                 self.assertEqual(topic.fuzzy_norm, test_data['fuzzy_norms'][i])
#                 self.assertEqual(topic.vector.shape[0], len(test_data['vectors'][i]))
                
#                 print(f"   ✅ Topic '{topic.name}': {len(topic.keywords)} keywords, {len(topic.lemma_set)} lemmas")
#             print(f"✅ topic_objects_from_meta: {len(topics)} Topic creati correttamente")
            
#             # ===============================
#             # TEST detect_covered_topics
#             # ===============================
#             print("\n🔄 Test detect_covered_topics...")
            
#             # Test cases con diversi livelli di matching
#             test_cases = [
#                 {
#                     'text': 'Sto programmando in Python e sviluppando software',
#                     'expected_topics': {'programmazione'},
#                     'description': 'Match esatto lemmi (Livello 1)'
#                 },
#                 {
#                     'text': 'Mi sono sposato con mia moglie in una bella cerimonia',
#                     'expected_topics': {'matrimonio'},
#                     'description': 'Match fuzzy (Livello 2)'
#                 },
#                 {
#                     'text': 'Ho completato i miei studi universitari e ottenuto la laurea',
#                     'expected_topics': {'educazione'},
#                     'description': 'Match semantico (Livello 3)'
#                 },
#                 {
#                     'text': 'Programmo software, mi sono sposato e ho studiato all università',
#                     'expected_topics': {'programmazione', 'matrimonio', 'educazione'},
#                     'description': 'Match multipli'
#                 },
#                 {
#                     'text': 'Oggi è una bella giornata di sole',
#                     'expected_topics': set(),
#                     'description': 'Nessun match'
#                 }
#             ]
#             # for case in test_cases:
#             #     print(f"\n📝 Test: {case['description']}")
#             #     print(f"   Testo: '{case['text'][:50]}...'")
                
#             #     covered_topics, coverage = detect_covered_topics(case['text'], topics)
                
#             #     # Verifica tipi di ritorno
#             #     self.assertIsInstance(covered_topics, set)
#             #     self.assertIsInstance(coverage, float)
                
#             #     # Verifica range coverage
#             #     self.assertGreaterEqual(coverage, 0.0)
#             #     self.assertLessEqual(coverage, 1.0)
                
#             #     # Verifica topic coperti
#             #     print(f"   Topic trovati: {covered_topics}")
#             #     print(f"   Coverage: {coverage:.2%}")
                
#             #     # Verifica che i topic attesi siano inclusi (permette topic aggiuntivi)
#             #     missing_topics = case['expected_topics'] - covered_topics
#             #     self.assertEqual(len(missing_topics), 0, 
#             #                    f"Topic attesi non trovati: {missing_topics}")
                
#             #     # Verifica coverage coerente
#             #     expected_coverage = len(case['expected_topics']) / len(topics)
#             #     if case['expected_topics']:
#             #         self.assertGreaterEqual(coverage, expected_coverage * 0.8,  # Tolleranza 20%
#             #                               f"Coverage troppo bassa: {coverage:.2%} vs atteso {expected_coverage:.2%}")
         
#             #     print(f"   ✅ Match verificato")

#             print(f"\n📝 Test: {test_cases[3]['description']}")
#             print(f"   Testo: '{test_cases[3]['text'][:50]}...'")
                
#             covered_topics, coverage = detect_covered_topics(test_cases[3]['text'], topics)
                
#             # Verifica tipi di ritorno
#             self.assertIsInstance(covered_topics, set)
#             self.assertIsInstance(coverage, float)
#             print("ciao",type(covered_topics), type(coverage))
#             # Verifica range coverage
#             self.assertGreaterEqual(coverage, 0.0)
#             self.assertLessEqual(coverage, 1.0)
            
#             # Verifica topic coperti
#             print(f"   Topic trovati: {covered_topics}")
#             print(f"   Coverage: {coverage:.2%}")
            
#             # Verifica che i topic attesi siano inclusi (permette topic aggiuntivi)
#             missing_topics = test_cases[3]['expected_topics'] - covered_topics
#             self.assertEqual(len(missing_topics), 0, 
#                             f"Topic attesi non trovati: {missing_topics}")
            
#             # Verifica coverage coerente
#             expected_coverage = len(test_cases[3]['expected_topics']) / len(topics)
#             if test_cases[3]['expected_topics']:
#                 self.assertGreaterEqual(coverage, expected_coverage * 0.8,  # Tolleranza 20%
#                                         f"Coverage troppo bassa: {coverage:.2%} vs atteso {expected_coverage:.2%}")
            
#             print(f"   ✅ Match verificato")

#             # ===============================
#             # TEST CASI LIMITE
#             # ===============================
#             print("\n🔄 Test casi limite...")
            
#             # Testo vuoto
#             covered_empty, coverage_empty = detect_covered_topics("", topics)
#             self.assertEqual(covered_empty, set())
#             self.assertEqual(coverage_empty, 0.0)
            
#             # Lista topic vuota
#             covered_no_topics, coverage_no_topics = detect_covered_topics("test", [])
#             self.assertEqual(covered_no_topics, set())
#             self.assertEqual(coverage_no_topics, 0.0)
            
#             print("✅ Casi limite gestiti correttamente")
            
#             # ===============================
#             # TEST RETRO-COMPATIBILITÀ
#             # ===============================
#             print("\n🔄 Test retro-compatibilità topic_from_meta...")
            
#             topics_compat = topic_from_meta(
#                 subtopics=test_data['subtopics'],
#                 lemma_sets=test_data['lemma_sets'],
#                 fuzzy_norms=test_data['fuzzy_norms'],
#                 vectors=test_data['vectors']
#             )
            
#             self.assertIsInstance(topics_compat, list)
#             self.assertEqual(len(topics_compat), len(topics))
            
#             # Verifica che i topic siano equivalenti (tranne keywords)
#             for i, (topic_new, topic_compat) in enumerate(zip(topics, topics_compat)):
#                 self.assertEqual(topic_new.name, topic_compat.name)
#                 self.assertEqual(topic_new.lemma_set, topic_compat.lemma_set)
#                 self.assertEqual(topic_new.fuzzy_norm, topic_compat.fuzzy_norm)
#                 np.testing.assert_array_equal(topic_new.vector, topic_compat.vector)
#                 # topic_compat.keywords dovrebbe essere lista vuota
#                 self.assertEqual(topic_compat.keywords, [])
            
#             print("✅ Retro-compatibilità verificata")
            
#             print(f"\n✅ Test completato: {len(topics)} topic testati su {len(test_cases)} casi")
            
#         except Exception as e:
#             self.fail(f"Errore nel test topic objects e detection: {e}")




def run_tests():
    """Esegue i test con output colorato"""
    print("=" * 60)
    print("🧪 TEST SUITE - TOPIC DETECTION")
    print("=" * 60)
    print("\nPer abilitare più test, decommenta le funzioni nel file")
    print("Ordine consigliato:")
    print("1. Test Level 1 (Unit Test)")
    print("2. Test Level 2 (Integration Test)") 
    print("3. Test Level 3 (Performance Test)")
    print("\n" + "=" * 60)
    
    # Esegui i test
    unittest.main(verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()