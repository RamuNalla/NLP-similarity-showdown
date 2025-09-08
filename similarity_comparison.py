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

    def create_similarity_heatmaps(self):             # visualize similarity matrices using heatmaps
        
        tfidf_sim = self.get_tfidf_similarity(self.sample_sentences)
        transformer_sim = self.get_transformer_similarity(self.sample_sentences)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(tfidf_sim, annot=True, fmt='.3f', cmap='Blues',         # TF-IDF heatmap
                    xticklabels=range(len(self.sample_sentences)),
                    yticklabels=range(len(self.sample_sentences)),
                    ax=axes[0])
        axes[0].set_title('TF-IDF Cosine Similarity Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentence Index')
        axes[0].set_ylabel('Sentence Index')
        
        sns.heatmap(transformer_sim, annot=True, fmt='.3f', cmap='Reds',        # Transformer heatmap
                    xticklabels=range(len(self.sample_sentences)),
                    yticklabels=range(len(self.sample_sentences)),
                    ax=axes[1])
        axes[1].set_title('Transformer Cosine Similarity Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sentence Index')
        axes[1].set_ylabel('Sentence Index')
        
        plt.tight_layout()
        plt.savefig('similarity_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tfidf_sim, transformer_sim
    

    def compare_methods_detailed(self) -> pd.DataFrame:     # detailed comparison of methods on sample pairs
        
        df = self.analyze_sample_pairs()
        
        df['Difference (Trans - TFIDF)'] = df['Transformer Similarity'] - df['TF-IDF Similarity']
        
        df = df.sort_values(['Semantic Relationship', 'Transformer Similarity'], ascending=[True, False])
        
        return df


    def visualize_method_comparison(self, df: pd.DataFrame):            # create visualizations
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].scatter(df['TF-IDF Similarity'], df['Transformer Similarity'],   # TF-IDF vs Transformer similarity
                          c=['red' if rel == 'Synonymous' else 'blue' if rel == 'Related' else 'gray' 
                             for rel in df['Semantic Relationship']], alpha=0.7, s=100)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)  # diagonal line
        axes[0, 0].set_xlabel('TF-IDF Similarity')
        axes[0, 0].set_ylabel('Transformer Similarity')
        axes[0, 0].set_title('TF-IDF vs Transformer Similarity')
        axes[0, 0].grid(True, alpha=0.3)
        
        avg_similarities = df.groupby('Semantic Relationship')[['TF-IDF Similarity', 'Transformer Similarity']].mean()      #  Bar plot - Average similarity by relationship type
        x = np.arange(len(avg_similarities.index))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, avg_similarities['TF-IDF Similarity'], width, 
                      label='TF-IDF', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, avg_similarities['Transformer Similarity'], width,
                      label='Transformer', alpha=0.8, color='lightcoral')
        
        axes[0, 1].set_xlabel('Semantic Relationship')
        axes[0, 1].set_ylabel('Average Similarity Score')
        axes[0, 1].set_title('Average Similarity by Relationship Type')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(avg_similarities.index)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        methods_data = [df['TF-IDF Similarity'], df['Transformer Similarity']]      # Box plot - Distribution of similarities
        axes[1, 0].boxplot(methods_data, labels=['TF-IDF', 'Transformer'])
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_title('Distribution of Similarity Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(df['Difference (Trans - TFIDF)'], bins=10, alpha=0.7, color='green', edgecolor='black')  # Difference histogram
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Difference (Transformer - TF-IDF)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Transformer vs TF-IDF Performance Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('method_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show() 

def main():         # main function to run the complete similarity analysis
        
    print("Starting NLP Similarity Showdown!")
    print("-"*50)
    
    analyzer = SimilarityAnalyzer()     # Initialize analyzer
    
    print("\nAnalyzing similarity matrices...")
    tfidf_matrix, transformer_matrix = analyzer.create_similarity_heatmaps()            # Create and display heatmaps
    
    print("\nPerforming detailed method comparison...")
    comparison_df = analyzer.compare_methods_detailed()             # detailed method comparison

    print("\nDetailed Similarity Comparison Results:")              # display results
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv('similarity_comparison_results.csv', index=False)      # save results to CSV
    
    print("\nCreating visualization comparisons...")
    analyzer.visualize_method_comparison(comparison_df)             # create comparison visualizations
    
    print("\nExample: Two similar sentences comparison:")
    example_similarities = analyzer.calculate_pairwise_similarity(
        "The cat is sleeping on the sofa",
        "A feline is resting on the couch"
    )
    
    for method, score in example_similarities.items():
        print(f"{method}: {score:.4f}")
    
    print("\nExample: Two different sentences comparison:")
    different_similarities = analyzer.calculate_pairwise_similarity(
        "I love eating pizza for dinner",
        "The car broke down on the highway"
    )
    
    for method, score in different_similarities.items():
        print(f"{method}: {score:.4f}")
    
    print("\nAnalysis complete! Check the generated files:")
    print("- similarity_matrices_comparison.png")
    print("- method_comparison_analysis.png") 
    print("- similarity_comparison_results.csv")

if __name__ == "__main__":
    main()