import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def preprocess_text(text: str) -> str:              # basic text preprocessing
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def load_custom_dataset(file_path: str) -> List[str]:       # load custom dataset from file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]
        return sentences
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return []


def calculate_correlation(scores1: List[float], scores2: List[float]) -> Dict[str, float]:      # calculate correlation between two sets of similarity scores

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    pearson_corr = np.corrcoef(scores1, scores2)[0, 1]          # Pearson correlation
    
    from scipy.stats import spearmanr                           # Spearman rank correlation
    spearman_corr, _ = spearmanr(scores1, scores2)

    mae = mean_absolute_error(scores1, scores2)                  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(scores1, scores2))          # Root Mean Square Error
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'mae': mae,
        'rmse': rmse
    } 


def create_similarity_summary_table(similarities_dict: Dict[str, List[float]], 
                                  sentence_pairs: List[Tuple[str, str]]) -> pd.DataFrame:           # create a summary table of similarity results

    data = []
    
    for i, (sent1, sent2) in enumerate(sentence_pairs):
        row = {
            'Pair_ID': i + 1,
            'Sentence_1': sent1[:50] + "..." if len(sent1) > 50 else sent1,
            'Sentence_2': sent2[:50] + "..." if len(sent2) > 50 else sent2,
        }

        for method, scores in similarities_dict.items():        # add similarity scores for each method
            row[f'{method}_Similarity'] = scores[i]
        
        if len(similarities_dict) >= 2:                         # compute difference if at least two methods        
            methods = list(similarities_dict.keys())
            row['Difference'] = similarities_dict[methods[1]][i] - similarities_dict[methods[0]][i]
        
        data.append(row)
    
    return pd.DataFrame(data) 


def plot_similarity_distribution(similarities_dict: Dict[str, List[float]],             # plot distribution of similarity scores
                               title: str = "Similarity Score Distributions"):

    plt.figure(figsize=(12, 6))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, (method, scores) in enumerate(similarities_dict.items()):
        plt.hist(scores, bins=15, alpha=0.7, label=method, 
                color=colors[i % len(colors)], density=True)
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('similarity_distributions_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()



def print_method_comparison_summary(tfidf_scores: List[float], 
                                  transformer_scores: List[float],
                                  sentence_pairs: List[Tuple[str, str]]):       # print comparison summary between two methods
    
    print("\n Method Comparison Summary")
    print("="*50)
    
    correlations = calculate_correlation(tfidf_scores, transformer_scores)      # calculate correlations
    
    print(f"Pearson Correlation: {correlations['pearson_correlation']:.4f}")
    print(f"Spearman Correlation: {correlations['spearman_correlation']:.4f}")
    print(f"Mean Absolute Difference: {correlations['mae']:.4f}")
    
    differences = np.array(transformer_scores) - np.array(tfidf_scores)         # find the differences
    max_diff_idx = np.argmax(np.abs(differences))
    
    print(f"\nBiggest Difference (Index {max_diff_idx}):")
    print(f"Sentences: '{sentence_pairs[max_diff_idx][0]}'")
    print(f"          '{sentence_pairs[max_diff_idx][1]}'") 
    print(f"TF-IDF: {tfidf_scores[max_diff_idx]:.4f}")
    print(f"Transformer: {transformer_scores[max_diff_idx]:.4f}")
    print(f"Difference: {differences[max_diff_idx]:.4f}")


def export_embeddings_for_analysis(sentences: List[str], 
                                 tfidf_vectorizer, 
                                 sentence_model) -> Tuple[np.ndarray, np.ndarray]:      # export embeddings for external analysis
    
    tfidf_embeddings = tfidf_vectorizer.fit_transform(sentences).toarray()      # Get embeddings
    transformer_embeddings = sentence_model.encode(sentences)
    
    np.save('tfidf_embeddings.npy', tfidf_embeddings)                   # Save to files
    np.save('transformer_embeddings.npy', transformer_embeddings)
    
    with open('sentences.txt', 'w', encoding='utf-8') as f:             # Save sentences
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(" Embeddings exported:")
    print("- tfidf_embeddings.npy")
    print("- transformer_embeddings.npy") 
    print("- sentences.txt")
    
    return tfidf_embeddings, transformer_embeddings


def create_feature_importance_analysis(tfidf_vectorizer, sample_sentence: str):   # analyze which features are important for TF-IDF       
    
    feature_names = tfidf_vectorizer.get_feature_names_out()            # Get feature names and vector for sample sentence
    vector = tfidf_vectorizer.transform([sample_sentence]).toarray()[0]
    

    top_indices = np.argsort(vector)[-10:][::-1]                        # Top 10 features
    
    print(f"\n🔍 TF-IDF Feature Analysis for: '{sample_sentence}'")
    print("-" * 60)
    print(f"{'Feature':<20} {'TF-IDF Score':<15}")
    print("-" * 35)
    
    for idx in top_indices:
        if vector[idx] > 0:
            print(f"{feature_names[idx]:<20} {vector[idx]:<15.4f}")

    