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

    
    def scalability_test(self) -> Dict[str, List[float]]:           # test how methods scale with increasing dataset size
        
        print("\n Scalability Test")
        print("-" * 20)
        
        base_sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning algorithms process large datasets",
            "Beautiful flowers bloom in the spring garden",
            "Technology advances rapidly in modern times",
            "Ocean waves crash against the rocky shore"
        ]
        
        dataset_sizes = [10, 25, 50, 100, 200]
        tfidf_times = []
        transformer_times = []
        
        for size in dataset_sizes:
            test_sentences = []
            for i in range(size):                       # create dataset by repeating and slightly modifying sentences
                base_idx = i % len(base_sentences)
                sentence = base_sentences[base_idx] + f" (variant {i})"
                test_sentences.append(sentence)
            
            print(f"Testing with {size} sentences...")
            
            start_time = time.time()                    # Time TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(test_sentences)
            cosine_similarity(tfidf_matrix)
            tfidf_time = time.time() - start_time
            tfidf_times.append(tfidf_time)
            
            start_time = time.time()                # Time Transformer
            embeddings = self.sentence_model.encode(test_sentences)
            cosine_similarity(embeddings)
            transformer_time = time.time() - start_time
            transformer_times.append(transformer_time)
        
        return {
            'dataset_sizes': dataset_sizes,
            'tfidf_times': tfidf_times,
            'transformer_times': transformer_times
        }



    def create_benchmark_visualizations(self, speed_results: Dict[str, float], 
                                      accuracy_df: pd.DataFrame, 
                                      scalability_results: Dict[str, List[float]]):     # Create comprehensive benchmark visualizations
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(speed_results.keys())                # Speed comparison
        times = list(speed_results.values())
        
        bars = axes[0, 0].bar(methods, times, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 0].set_ylabel('Average Time (seconds)')
        axes[0, 0].set_title('Speed Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, times):          # Add value labels
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time_val:.4f}s', ha='center', va='bottom')
        
        mae_scores = [accuracy_df['TF-IDF Error'].mean(), accuracy_df['Transformer Error'].mean()]          # Accuracy comparison
        bars = axes[0, 1].bar(methods, mae_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Accuracy Comparison (Lower is Better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, mae in zip(bars, mae_scores):          # Add value labels
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mae:.4f}', ha='center', va='bottom')
        
        sizes = scalability_results['dataset_sizes']        # Scalability comparison
        axes[1, 0].plot(sizes, scalability_results['tfidf_times'], 'o-', 
                       label='TF-IDF', color='skyblue', linewidth=2, markersize=8)
        axes[1, 0].plot(sizes, scalability_results['transformer_times'], 's-', 
                       label='Transformer', color='lightcoral', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Processing Time (seconds)')
        axes[1, 0].set_title('Scalability Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].boxplot([accuracy_df['TF-IDF Error'], accuracy_df['Transformer Error']],     # Error distribution
                          labels=['TF-IDF', 'Transformer'])
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()


    def generate_benchmark_report(self, speed_results: Dict[str, float], 
                                accuracy_df: pd.DataFrame) -> str:          # generate benchmark report
        
        report = f"""
# NLP Methods Benchmark Report

## Speed Performance
- TF-IDF: {speed_results['TF-IDF']:.4f} seconds
- Transformer: {speed_results['Transformer']:.4f} seconds
- Speed Ratio: {speed_results['Transformer']/speed_results['TF-IDF']:.2f}x slower for Transformers

## Accuracy Performance (Mean Absolute Error)
- TF-IDF MAE: {accuracy_df['TF-IDF Error'].mean():.4f}
- Transformer MAE: {accuracy_df['Transformer Error'].mean():.4f}
- Accuracy Improvement: {((accuracy_df['TF-IDF Error'].mean() - accuracy_df['Transformer Error'].mean())/accuracy_df['TF-IDF Error'].mean()*100):.1f}%

## Key Findings
- Transformers are more accurate but slower than TF-IDF
- Transformers better capture semantic similarity
- TF-IDF is suitable for large-scale, real-time applications
- Transformers excel in tasks requiring deep semantic understanding

## Recommendations
- Use TF-IDF for: Large datasets, real-time systems, keyword-based similarity
- Use Transformers for: Semantic search, content recommendation, high-accuracy requirements
        """
        
        return report
    

def main():         # Run benchmarking suite complete
    
    print(" Starting NLP Methods Benchmark Suite!")
    print("-"*50)
    
    benchmark = NLPMethodBenchmark()
    
    speed_results = benchmark.benchmark_speed(benchmark.sentences[:10])     # Speed benchmark
     
    accuracy_df = benchmark.calculate_accuracy_metrics()                    # Accuracy benchmark 
    
    print("\n Running scalability test...")                                 # Scalability test
    scalability_results = benchmark.scalability_test()
    
    print("\n Creating benchmark visualizations...")                        # Create visualizations
    benchmark.create_benchmark_visualizations(speed_results, accuracy_df, scalability_results)
    
    report = benchmark.generate_benchmark_report(speed_results, accuracy_df)    # Generate report
    
    with open('benchmark_report.md', 'w') as f:                                  # Save report
        f.write(report)
    
    accuracy_df.to_csv('accuracy_analysis.csv', index=False)                    # Save detailed results
    
    print("\n Benchmark Summary:")                                            # Print summary
    print(report)
    
    print("\n Benchmark complete! Generated files:")
    print("- benchmark_results.png")
    print("- benchmark_report.md")
    print("- accuracy_analysis.csv")


if __name__ == "__main__":
    main()