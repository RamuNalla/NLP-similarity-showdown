import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class NLPMethodBenchmark:               # a detailed benchmarking of different NLP similarity methods   

    def __init__(self):
        print("Initializing benchmark models...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2)
        )

        self.ground_truth_pairs = [          # sentence pairs with known similarity levels (human annotated )
            # High similarity (0.8-1.0)
            ("The cat sleeps on the bed", "A feline rests on the mattress", 0.9),
            ("I enjoy eating pizza", "Pizza is my favorite food", 0.85),
            ("The car is red", "The automobile is crimson", 0.8),
            
            # Medium similarity (0.4-0.7)
            ("Dogs like to play", "Cats enjoy sleeping", 0.5),
            ("Reading books is fun", "Watching movies is entertaining", 0.6),
            ("The sun is bright", "The moon glows softly", 0.4),
            
            # Low similarity (0.0-0.3)
            ("I love programming", "The weather is cold", 0.1),
            ("Mathematics is difficult", "Flowers smell nice", 0.0),
            ("Computers process data", "Birds fly in the sky", 0.2),
        ]

    def benchmark_speed(self, sentences: List[str], iterations: int = 5) -> Dict[str, float]:       # benchmark speed of methods

        print(f"\n  Speed Benchmark ({iterations} iterations)")
        print("-" * 40)

        results = {}

        tfidf_times = []                    # TF-IDF speed test
        for _ in range(iterations):
            start_time = time.time()
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            end_time = time.time()
            tfidf_times.append(end_time - start_time)

        results['TF-IDF'] = np.mean(tfidf_times)

        transformer_times = []          # Transformer speed test
        for _ in range(iterations):
            start_time = time.time()
            embeddings = self.sentence_model.encode(sentences)
            similarity_matrix = cosine_similarity(embeddings)
            end_time = time.time()
            transformer_times.append(end_time - start_time)
        
        results['Transformer'] = np.mean(transformer_times)
        
        return results
    
    def calculate_accuracy_metrics(self) -> pd.DataFrame:          # calculate accuracy metrics against ground truth

        print("\n  Accuracy Benchmark")
        print("-" * 40)

        results = []

        for sent1, sent2, ground_truth in self.ground_truth_pairs:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([sent1, sent2])
            tfidf_sim = cosine_similarity(tfidf_matrix)[0, 1]

            embeddings = self.sentence_model.encode([sent1, sent2])
            transformer_sim = cosine_similarity(embeddings.reshape(2, -1))[0, 1]

            results.append({
                'Sentence 1': sent1,
                'Sentence 2': sent2,
                'Ground Truth': ground_truth,
                'TF-IDF Score': tfidf_sim,
                'Transformer Score': transformer_sim,
                'TF-IDF Error': abs(tfidf_sim - ground_truth),
                'Transformer Error': abs(transformer_sim - ground_truth)
            })
        
        df = pd.DataFrame(results)

        tfidf_mae = df['TF-IDF Error'].mean()                   # overall metrics
        transformer_mae = df['Transformer Error'].mean()

        tfidf_rmse = np.sqrt(df['TF-IDF Error'].pow(2).mean())
        transformer_rmse = np.sqrt(df['Transformer Error'].pow(2).mean())

        print(f"TF-IDF Mean Absolute Error: {tfidf_mae:.4f}")
        print(f"Transformer Mean Absolute Error: {transformer_mae:.4f}")
        print(f"TF-IDF RMSE: {tfidf_rmse:.4f}")
        print(f"Transformer RMSE: {transformer_rmse:.4f}")
        
        return df
