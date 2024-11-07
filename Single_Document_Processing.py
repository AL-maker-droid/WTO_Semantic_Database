from pathlib import Path
import sys
import os
import argparse
import json
import jsonlines
import datetime

# Ensure the current directory is in the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from core.pdf_text_extractor import extract_text_from_pdf
from core.hierarchical_embedder import HierarchicalEmbedder, save_embeddings
from core.semantic_searcher import AdvancedSemanticSearcher, print_advanced_results

def setup_directories(base_dir=None):
    """
    Ensure necessary directories exist
    """
    base_dir = base_dir or Path.cwd()
    dirs = {
        'raw': base_dir / 'data' / 'raw',
        'processed_text': base_dir / 'data' / 'processed' / 'text',
        'processed_embeddings': base_dir / 'data' / 'processed' / 'embeddings',
        'processed_results': base_dir / 'data' / 'processed' / 'results'  # New directory for results
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def check_existing_files(pdf_path, dirs):
    """
    Check if text extraction and embeddings already exist
    
    Returns:
        tuple: (text_path, embeddings_path, is_processed)
    """
    text_filename = f"{pdf_path.stem}_extracted.txt"
    embeddings_filename = f"{pdf_path.stem}_embeddings.json"
    
    text_path = dirs['processed_text'] / text_filename
    embeddings_path = dirs['processed_embeddings'] / embeddings_filename
    
    is_processed = text_path.exists() and embeddings_path.exists()
    
    return text_path, embeddings_path, is_processed

def process_document(pdf_path, text_path, embeddings_path, force_reprocess=False):
    """
    Process PDF document, generating text and embeddings
    """
    # Extract text if not already extracted or force reprocessing
    if not text_path.exists() or force_reprocess:
        print("1. Extracting text from PDF...")
        extract_text_from_pdf(pdf_path, text_path)
    else:
        print(f"1. Using existing text extraction: {text_path}")
    
    # Read extracted text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Generate embeddings if not already generated or force reprocessing
    if not embeddings_path.exists() or force_reprocess:
        print("2. Generating hierarchical embeddings...")
        embedder = HierarchicalEmbedder()
        embeddings = embedder.process_document(text, document_name=pdf_path.name)
        
        # Save embeddings
        save_embeddings(embeddings, embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")
    else:
        print(f"2. Using existing embeddings: {embeddings_path}")
    
    return embeddings_path

def save_search_results(pdf_path, queries, results, dirs):
    """
    Save search results in multiple formats
    
    Args:
        pdf_path (Path): Path to the original PDF
        queries (list): Search queries
        results (list): Search results
        dirs (dict): Directory paths
    
    Returns:
        tuple: Paths to generated result files
    """
    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{pdf_path.stem}_{timestamp}"
    
    # JSONL file for machine-readable results
    jsonl_path = dirs['processed_results'] / f"{base_filename}_results.jsonl"
    with jsonlines.open(jsonl_path, mode='w') as writer:
        for query, query_results in zip(queries, results):
            writer.write({
                'query': query,
                'timestamp': timestamp,
                'source_pdf': str(pdf_path),
                'results': query_results
            })
    
    # Markdown file for human-readable results
    md_path = dirs['processed_results'] / f"{base_filename}_results.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Semantic Search Results\n\n")
        f.write(f"**Source PDF:** {pdf_path}\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")
        
        for query, query_results in zip(queries, results):
            f.write(f"## Query: '{query}'\n\n")
            
            if query_results:
                for i, result in enumerate(query_results, 1):
                    f.write(f"### Result {i}\n")
                    f.write(f"**Relevance Score:** {result.get('score', 'N/A')}\n\n")
                    f.write(f"**Text:** {result.get('text', 'No text available')}\n\n")
            else:
                f.write("*No results found*\n\n")
    
    return jsonl_path, md_path

def perform_semantic_search(embeddings_path, queries, threshold=0.3, top_k=3):
    """
    Perform semantic search on the document
    
    Returns:
        list: List of search results for each query
    """
    print("\n3. Running advanced semantic searches...")
    searcher = AdvancedSemanticSearcher(embeddings_path)
    
    # Store results for each query
    all_results = []
    
    # Perform searches
    for query in queries:
        print(f"\n--- Searching for: '{query}' ---")
        try:
            results = searcher.search(query, top_k=top_k, threshold=threshold)
            
            if results:
                print_advanced_results(results)
                all_results.append(results)
            else:
                print(f"No results found for query: '{query}' with current threshold.")
                all_results.append([])
        except Exception as e:
            print(f"Error searching query '{query}': {e}")
            all_results.append([])
    
    return all_results

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Semantic Document Search')
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--queries', nargs='+', 
                        default=[
                            "Environmental protection in trade", 
                            "Sustainability and economic policy", 
                            "Ecological economics and degrowth"
                        ],
                        help='Search queries')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Relevance threshold for search results')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='Number of top results to display')
    parser.add_argument('--force', action='store_true', 
                        help='Force reprocessing of document')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories()
    
    # Convert PDF path to Path object
    pdf_path = Path(args.pdf)
    
    # Check existing files
    text_path, embeddings_path, is_processed = check_existing_files(pdf_path, dirs)
    
    # Process document
    processed_embeddings_path = process_document(
        pdf_path, 
        text_path, 
        embeddings_path, 
        force_reprocess=args.force
    )
    
    # Perform semantic search
    search_results = perform_semantic_search(
        processed_embeddings_path, 
        args.queries, 
        threshold=args.threshold,
        top_k=args.top_k
    )
    
    # Save search results
    jsonl_path, md_path = save_search_results(
        pdf_path, 
        args.queries, 
        search_results, 
        dirs
    )
    
    print(f"\nResults saved:")
    print(f"- JSONL: {jsonl_path}")
    print(f"- Markdown: {md_path}")

if __name__ == "__main__":
    main() 