import sys
import subprocess
import os
from pathlib import Path

def check_python_version():                     # Check if Python version is compatible
    
    if sys.version_info < (3, 7):
        print(" Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f" Python version: {sys.version}")
    return True


def download_models():                          # Download required models
    
    print("\n Downloading NLP models...")
    try:
        from sentence_transformers import SentenceTransformer
        import nltk
        
        print("Downloading sentence transformer model...")      # Download sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" Sentence transformer model downloaded!")
        
        print("Downloading NLTK data...")                       # Download NLTK data if needed
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print(" NLTK data downloaded!")
        
        return True
    except Exception as e:
        print(f" Failed to download models: {e}")
        return False

def create_directories():                       # create necessary directories
    print("\n Creating directories...")
    directories = ['outputs', 'data', 'results']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f" Created directory: {directory}")

def run_initial_test():     # Run a quick test to ensure everything works
    print("\n Running initial test...")
    try:
        from similarity_comparison import SimilarityAnalyzer
        
        analyzer = SimilarityAnalyzer()
        test_result = analyzer.calculate_pairwise_similarity(
            "Hello world", "Hi there"
        )

        print(" Test successful!")
        print(f"Sample similarity scores: {test_result}")
        return True
    except Exception as e:
        print(f" Test failed: {e}")
        return False

def main():
    
    print(" NLP Similarity Showdown Setup")
    print("-" * 40)
    
    if not check_python_version():
        return False
        
    create_directories()
    
    if not download_models():
        return False
    
    if not run_initial_test():
        return False
    
    print("\n Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python similarity_comparison.py' for full analysis")
    print("2. Run 'python interactive_demo.py' for interactive testing")
    print("3. Check README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n Setup failed! Please check the error messages above.")
        sys.exit(1)