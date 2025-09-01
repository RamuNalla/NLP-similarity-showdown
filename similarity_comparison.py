import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SimilarityAnalyzer:

    def __init__(self):
        
        print("Loading Sentence Transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')       # initialize the transformer model

        self.tfidf_vectorizer = TfidfVectorizer(                    # initialize TF-IDF vectorizer
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )

        self.sample_sentences = [
            "The cat is sleeping on the sofa",
            "A feline is resting on the couch",
            "The dog is running in the park",
            "A canine is jogging through the garden",
            "I love eating pizza for dinner",
            "Pizza is my favorite meal in the evening",
            "The weather is beautiful today",
            "It's a gorgeous day outside",
            "Machine learning is fascinating",
            "AI and deep learning are interesting topics",
            "The car broke down on the highway",
            "My vehicle stopped working on the freeway"
        ]
    

    def get_tfidf_similarity(self, sentences: List[str]) -> np.ndarray:         # cosine similarity calculation using TF-IDF vectors

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    
    def get_transformer_similarity(self, sentences: List[str]) -> np.ndarray:       # cosine similarity calculation using Sentence Transformers

        embeddings = self.sentence_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def calculate_pairwise_similarity(self, sentence1: str, sentence2: str) -> Dict[str, float]:        # calculate similarity using all methods

        sentences = [sentence1, sentence2]
        
        tfidf_sim = self.get_tfidf_similarity(sentences)[0, 1]          # TF-IDF similarity
        transformer_sim = self.get_transformer_similarity(sentences)[0, 1]      # Transformer similarity
        
        return {
            'TF-IDF': tfidf_sim,
            'Transformer': transformer_sim
        }
    

    def analyze_sample_pairs(self) -> pd.DataFrame:         # predefined sentence pairs analysis to show method differences

        sentence_pairs = [
            ("The cat is sleeping on the sofa", "A feline is resting on the couch"),
            ("The dog is running in the park", "A canine is jogging through the garden"), 
            ("I love eating pizza for dinner", "Pizza is my favorite meal in the evening"),
            ("The weather is beautiful today", "It's a gorgeous day outside"),
            ("Machine learning is fascinating", "AI and deep learning are interesting topics"),
            ("The cat is sleeping on the sofa", "The dog is running in the park"),
            ("I love eating pizza for dinner", "The weather is beautiful today"),
            ("Machine learning is fascinating", "The car broke down on the highway")
        ]
        
        results = []
        for sent1, sent2 in sentence_pairs:
            similarities = self.calculate_pairwise_similarity(sent1, sent2)
            results.append({
                'Sentence 1': sent1,
                'Sentence 2': sent2,
                'TF-IDF Similarity': similarities['TF-IDF'],
                'Transformer Similarity': similarities['Transformer'],
                'Semantic Relationship': self._classify_relationship(sent1, sent2)
            })
        
        return pd.DataFrame(results)


    def _classify_relationship(self, sent1: str, sent2: str) -> str:        # classify the semantic relationship between sentences

        synonymous_pairs = [                        # simple keyword-based classification for demonstration
            ("cat", "feline"), ("dog", "canine"), ("sofa", "couch"),
            ("park", "garden"), ("beautiful", "gorgeous"), ("fascinating", "interesting")
        ]
        
        related_topics = [
            ("pizza", "dinner", "meal"), ("weather", "day"), 
            ("machine learning", "AI", "deep learning"), ("car", "vehicle")
        ]
        
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()
        
        for syn1, syn2 in synonymous_pairs:             # check for synonymous relationships
            if (syn1 in sent1_lower and syn2 in sent2_lower) or (syn2 in sent1_lower and syn1 in sent2_lower):
                return "Synonymous"
        
        for topic_group in related_topics:              # check for related topics
            count1 = sum(1 for word in topic_group if word in sent1_lower)
            count2 = sum(1 for word in topic_group if word in sent2_lower)
            if count1 > 0 and count2 > 0:
                return "Related"
        
        return "Unrelated"
    

    