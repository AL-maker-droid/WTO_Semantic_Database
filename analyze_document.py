from pathlib import Path
import sys
import os
from typing import Union

# Add the scripts directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Now import the local modules
from pdf_text_extractor import extract_text_from_pdf
from hierarchical_embedder import HierarchicalEmbedder, save_embeddings
from semantic_searcher import SemanticSearcher, print_results

class DocumentAnalyzer:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_document(self, input_path: Union[str, Path]) -> None:
        """Process a new document through the pipeline"""
        input_path = Path(input_path)
        
        # Generate output paths
        base_name = input_path.stem
        text_path = self.processed_dir / f"{base_name}_extracted.txt"
        embeddings_path = self.processed_dir / f"{base_name}_embeddings.json"
        
        print("\n=== Processing Document ===")
        print(f"Input file: {input_path}")
        
        # Step 1: Extract text from PDF
        print("\n1. Extracting text from PDF...")
        extract_text_from_pdf(input_path, text_path)
        
        # Step 2: Generate embeddings
        print("\n2. Generating hierarchical embeddings...")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        embedder = HierarchicalEmbedder()
        embeddings = embedder.process_document(text)
        save_embeddings(embeddings, embeddings_path)
        
        print(f"\nDocument processed successfully!")
        print(f"- Text extracted to: {text_path}")
        print(f"- Embeddings saved to: {embeddings_path}")
        
        return embeddings_path

    def search_document(self, embeddings_path: Path, query: str, 
                       top_k: int = 3, threshold: float = 0.4) -> None:
        """Search through a processed document"""
        print("\n=== Searching Document ===")
        print(f"Query: {query}")
        
        searcher = SemanticSearcher(embeddings_path)
        results = searcher.search(query, top_k=top_k, threshold=threshold)
        print_results(results)
        
        return results

def print_usage():
    print("""
Usage:
    1. To process a new document:
       python analyze_document.py process <path_to_pdf>
       
    2. To search in a processed document:
       python analyze_document.py search <path_to_embeddings> "<search_query>"
       
Example:
    python analyze_document.py process data/raw/document.pdf
    python analyze_document.py search data/processed/document_embeddings.json "environmental protection"
    """)

if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    try:
        if command == "process":
            if len(sys.argv) != 3:
                print("Error: Please provide the path to the PDF file")
                print_usage()
                sys.exit(1)
                
            pdf_path = sys.argv[2]
            analyzer.process_document(pdf_path)
            
        elif command == "search":
            if len(sys.argv) < 4:
                print("Error: Please provide both embeddings path and search query")
                print_usage()
                sys.exit(1)
                
            embeddings_path = Path(sys.argv[2])
            query = sys.argv[3]
            threshold = 0.3  # Lower default threshold
            
            if len(sys.argv) > 4 and sys.argv[4] == "--threshold":
                threshold = float(sys.argv[5])
            
            analyzer.search_document(embeddings_path, query, threshold=threshold)
            
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1) 