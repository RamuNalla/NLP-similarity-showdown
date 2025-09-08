from similarity_comparison import SimilarityAnalyzer
import pandas as pd

def interactive_similarity_demo():              # Interactive demo function

    print("Interactive NLP Similarity Demo")
    print("-"*40)
    print("Compare any two sentences using different NLP methods!\n")
    
    analyzer = SimilarityAnalyzer()
    
    while True:
        print("\nEnter two sentences to compare (or 'quit' to exit):")
        
        sentence1 = input("Sentence 1: ").strip()
        if sentence1.lower() == 'quit':
            break
            
        sentence2 = input("Sentence 2: ").strip()
        if sentence2.lower() == 'quit':
            break
        
        if not sentence1 or not sentence2:
            print("Please enter valid sentences!")
            continue
        
        print(f"\nAnalyzing similarities...")
        similarities = analyzer.calculate_pairwise_similarity(sentence1, sentence2)
        
        print(f"\n Similarity Results:")
        print(f"{'Method':<15} {'Similarity Score':<15} {'Interpretation'}")
        print("-" * 45)
        
        for method, score in similarities.items():
            interpretation = interpret_similarity(score)
            print(f"{method:<15} {score:<15.4f} {interpretation}")
        
        diff = similarities['Transformer'] - similarities['TF-IDF']             # difference
        print(f"\nTransformer captures {abs(diff):.4f} {'more' if diff > 0 else 'less'} similarity than TF-IDF")
        
        print(f"\n Analysis:")            # explanation
        if diff > 0.1:
            print("   Transformers detected semantic similarity that TF-IDF missed!")
        elif diff < -0.1:
            print("   TF-IDF found lexical similarity that transformers deemed less meaningful.")
        else:
            print("   Both methods agree on the similarity level.")


def interpret_similarity(score: float) -> str:          # human-readable interpretation

    if score >= 0.8:
        return "Very Similar"
    elif score >= 0.6:
        return "Similar"
    elif score >= 0.4:
        return "Somewhat Similar"
    elif score >= 0.2:
        return "Slightly Similar"
    else:
        return "Not Similar"
    

def batch_analysis_demo():                          # Batch analysis demo function
    print("\n Batch Analysis Demo")
    print("-"*30)

    custom_sentences = [                             # custom sentences
       "Artificial intelligence is transforming technology",
       "AI is revolutionizing tech industry",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning algorithms are powerful tools",
        "ML models can solve complex problems",
        "The sun is shining brightly today"
    ]
    
    analyzer = SimilarityAnalyzer()
    
    print("Analyzing custom sentence set...")
    
    tfidf_sim = analyzer.get_tfidf_similarity(custom_sentences)         # get similarities
    transformer_sim = analyzer.get_transformer_similarity(custom_sentences)
    
    results = []                    # custom dataframe
    for i in range(len(custom_sentences)):
        for j in range(i+1, len(custom_sentences)):
            results.append({
                'Sentence A': custom_sentences[i][:50] + "..." if len(custom_sentences[i]) > 50 else custom_sentences[i],
                'Sentence B': custom_sentences[j][:50] + "..." if len(custom_sentences[j]) > 50 else custom_sentences[j],
                'TF-IDF': tfidf_sim[i, j],
                'Transformer': transformer_sim[i, j],
                'Difference': transformer_sim[i, j] - tfidf_sim[i, j]
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Difference', ascending=False)
    
    print("\n Top differences (where Transformers outperform TF-IDF):")
    print(df.head(3).to_string(index=False))
    
    return df


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Interactive mode (compare your own sentences)")
    print("2. Batch analysis demo")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        interactive_similarity_demo()
    
    if choice in ['2', '3']:
        batch_analysis_demo()
    
    print("\nThanks for using the NLP Similarity Demo!")
