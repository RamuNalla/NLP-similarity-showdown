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


    def plot_embeddings(self, embeddings_2d: np.ndarray, title: str, ax):               # Plot 2D embeddings with color coding by semantic category

        for i, (sentence, label) in enumerate(zip(self.sentences, self.labels)):
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                      c=self.colors[label], s=100, alpha=0.7, 
                      label=label if label not in [item.get_text() for item in ax.get_legend_handles_labels()[1]] else "")
        
        for i, (x, y) in enumerate(embeddings_2d):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.8)

    
    def create_embedding_comparison(self):              # create side-by-side comparison of embedding visualizations
        
        tfidf_emb, transformer_emb = self.get_embeddings()
        
        tfidf_2d_tsne = self.reduce_dimensions(tfidf_emb, 'tsne')                   # reduce dimensions (t-sne)
        transformer_2d_tsne = self.reduce_dimensions(transformer_emb, 'tsne')
        
        tfidf_2d_pca = self.reduce_dimensions(tfidf_emb, 'pca')                     # reduce dimensions (pca)
        transformer_2d_pca = self.reduce_dimensions(transformer_emb, 'pca')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))                            # create subplots
        
        self.plot_embeddings(tfidf_2d_tsne, 'TF-IDF Embeddings (t-SNE)', axes[0, 0])    # TF-IDF visualizations
        self.plot_embeddings(tfidf_2d_pca, 'TF-IDF Embeddings (PCA)', axes[1, 0])
         
        self.plot_embeddings(transformer_2d_tsne, 'Transformer Embeddings (t-SNE)', axes[0, 1])     # Transformer visualizations 
        self.plot_embeddings(transformer_2d_pca, 'Transformer Embeddings (PCA)', axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('embedding_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tfidf_emb, transformer_emb
    

    def analyze_embedding_properties(self, tfidf_emb: np.ndarray, transformer_emb: np.ndarray):     # analyze and compare embedding properties
        
        print("\n Embedding Properties Analysis:")
        print("-"*40)
        
        print(f"TF-IDF Embedding Shape: {tfidf_emb.shape}")
        print(f"Transformer Embedding Shape: {transformer_emb.shape}")
        
        print(f"\nTF-IDF Sparsity: {np.mean(tfidf_emb == 0):.2%}")
        print(f"Transformer Sparsity: {np.mean(transformer_emb == 0):.2%}")
        
        print(f"\nTF-IDF Mean Vector Norm: {np.mean(np.linalg.norm(tfidf_emb, axis=1)):.4f}")
        print(f"Transformer Mean Vector Norm: {np.mean(np.linalg.norm(transformer_emb, axis=1)):.4f}")



    def create_similarity_distribution_plot(self):              # compare similarity score distributions
        
        tfidf_emb, transformer_emb = self.get_embeddings()
        
        tfidf_similarities = []
        transformer_similarities = []
        
        for i in range(len(self.sentences)):
            for j in range(i+1, len(self.sentences)):
                tfidf_sim = np.dot(tfidf_emb[i], tfidf_emb[j]) / (
                    np.linalg.norm(tfidf_emb[i]) * np.linalg.norm(tfidf_emb[j])
                )
                transformer_sim = np.dot(transformer_emb[i], transformer_emb[j]) / (
                    np.linalg.norm(transformer_emb[i]) * np.linalg.norm(transformer_emb[j])
                )
                
                tfidf_similarities.append(tfidf_sim)
                transformer_similarities.append(transformer_sim)
        
        plt.figure(figsize=(12, 6))                 # Create distribution plot
        
        plt.subplot(1, 2, 1)
        plt.hist(tfidf_similarities, bins=15, alpha=0.7, color='skyblue', 
                label='TF-IDF', density=True)
        plt.hist(transformer_similarities, bins=15, alpha=0.7, color='lightcoral', 
                label='Transformer', density=True)
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Density')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(tfidf_similarities, transformer_similarities, alpha=0.6, s=60)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.xlabel('TF-IDF Similarity')
        plt.ylabel('Transformer Similarity')
        plt.title('TF-IDF vs Transformer Similarity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():             # run the visualization demo

    print("Starting Embedding Visualization Demo!")
    print("-"*45)
    
    visualizer = EmbeddingVisualizer()
    
    print("\n Creating embedding visualizations...")
    tfidf_emb, transformer_emb = visualizer.create_embedding_comparison()
    
    print("\n Analyzing embedding properties...")
    visualizer.analyze_embedding_properties(tfidf_emb, transformer_emb)
    
    print("\n Creating similarity distribution plots...")
    visualizer.create_similarity_distribution_plot()
    
    print("\n Visualization complete! Generated files:")
    print("- embedding_visualizations.png")
    print("- similarity_distributions.png")
    
    print(f"\n Sentence Reference:")
    for i, sentence in enumerate(visualizer.sentences):
        print(f"{i:2d}: {sentence}")

if __name__ == "__main__":
    main()
