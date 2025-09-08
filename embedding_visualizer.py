import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import seaborn as sns
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EmbeddingVisualizer:

    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,               # reduced for visualization
            ngram_range=(1, 2)
        )
    
        self.sentences = [
            "The cat is sleeping peacefully",           # animals
            "A feline rests quietly",
            "The dog runs fast",
            "A canine sprints quickly",
            
            "I love eating pizza",                      # food
            "Pizza is delicious food",
            "Ice cream tastes great",
            "Frozen dessert is wonderful",

            "Machine learning is powerful",             # technology
            "AI algorithms are sophisticated",
            "Deep learning networks work well",
            "Neural networks process data",
            
            "The weather is sunny today",               # weather
            "It's a beautiful bright day",
            "Rain falls from dark clouds",
            "Precipitation comes from storms",
            
            "The car drives on roads",                  # transportation
            "Vehicles travel on highways",
            "Planes fly through the sky",
            "Aircraft soar above clouds"
        ]

        self.labels = [
            'Animals', 'Animals', 'Animals', 'Animals',
            'Food', 'Food', 'Food', 'Food', 
            'Technology', 'Technology', 'Technology', 'Technology',
            'Weather', 'Weather', 'Weather', 'Weather',
            'Transportation', 'Transportation', 'Transportation', 'Transportation'
        ]
        
        self.colors = {                         # color mapping for categories  
            'Animals': '#FF6B6B',
            'Food': '#4ECDC4', 
            'Technology': '#45B7D1',
            'Weather': '#96CEB4',
            'Transportation': '#FFEAA7'
        }
    

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:      # get embeddings from both methods

        tfidf_embeddings = self.tfidf_vectorizer.fit_transform(self.sentences).toarray()        # TF-IDF embeddings
        
        transformer_embeddings = self.sentence_model.encode(self.sentences)                 # Transformer embeddings
        
        return tfidf_embeddings, transformer_embeddings


    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:        # Reduce embeddings to 2D for visualization
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:       # t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
        
        return reducer.fit_transform(embeddings)
