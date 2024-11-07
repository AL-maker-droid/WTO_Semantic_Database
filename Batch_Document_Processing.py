from pathlib import Path
import sys
import os
import argparse
import json
import jsonlines
import datetime
from tqdm import tqdm
import multiprocessing
from functools import partial

# Ensure the current directory is in the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from Single_Document_Processing import (
    setup_directories, 
    check_existing_files, 
    process_document, 
    perform_semantic_search, 
    save_search_results
)

def process_single_pdf(pdf_path, dirs, queries, threshold, top_k, force_reprocess):
    """
    Process a single PDF document
    
    Args:
        pdf_path (Path): Path to PDF file
        dirs (dict): Directory paths
        queries (list): Search queries
        threshold (float): Relevance threshold
        top_k (int): Number of top results
        force_reprocess (bool): Force reprocessing flag
    
    Returns:
        tuple: PDF path, search results, result file paths
    """
    try:
        # Check existing files
        text_path, embeddings_path, is_processed = check_existing_files(pdf_path, dirs)
        
        # Process document
        processed_embeddings_path = process_document(
            pdf_path, 
            text_path, 
            embeddings_path, 
            force_reprocess=force_reprocess
        )
        
        # Perform semantic search
        search_results = perform_semantic_search(
            processed_embeddings_path, 
            queries, 
            threshold=threshold,
            top_k=top_k
        )
        
        # Save search results
        jsonl_path, md_path = save_search_results(
            pdf_path, 
            queries, 
            search_results, 
            dirs
        )
        
        return pdf_path, search_results, (jsonl_path, md_path)
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return pdf_path, None, None

def batch_process_pdfs(pdf_directory, queries, threshold, top_k, force_reprocess, batch_size=5):
    """
    Process PDFs in batches
    
    Args:
        pdf_directory (str): Directory containing PDFs
        queries (list): Search queries
        threshold (float): Relevance threshold
        top_k (int): Number of top results
        force_reprocess (bool): Force reprocessing flag
        batch_size (int): Number of PDFs to process simultaneously
    """
    # Setup directories
    dirs = setup_directories()
    
    # Find all PDF files
    pdf_paths = list(Path(pdf_directory).glob('*.pdf'))
    
    # Create progress bar
    with tqdm(total=len(pdf_paths), desc="Processing PDFs", unit="pdf") as pbar:
        # Process PDFs in batches
        for i in range(0, len(pdf_paths), batch_size):
            batch_pdfs = pdf_paths[i:i+batch_size]
            
            # Use multiprocessing for batch processing
            with multiprocessing.Pool(processes=min(batch_size, multiprocessing.cpu_count())) as pool:
                process_func = partial(
                    process_single_pdf, 
                    dirs=dirs, 
                    queries=queries, 
                    threshold=threshold, 
                    top_k=top_k, 
                    force_reprocess=force_reprocess
                )
                
                # Process batch
                batch_results = pool.map(process_func, batch_pdfs)
                
                # Update progress bar
                pbar.update(len(batch_pdfs))
                
                # Log batch results
                for pdf_path, results, result_paths in batch_results:
                    if results:
                        print(f"\nProcessed: {pdf_path}")
                        if result_paths:
                            print(f"- JSONL: {result_paths[0]}")
                            print(f"- Markdown: {result_paths[1]}")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Batch Semantic Document Search')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing PDFs')
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
                        help='Force reprocessing of documents')
    parser.add_argument('--batch_size', type=int, default=5, 
                        help='Number of PDFs to process simultaneously')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Batch process PDFs
    batch_process_pdfs(
        args.dir, 
        args.queries, 
        args.threshold, 
        args.top_k, 
        args.force,
        args.batch_size
    )

if __name__ == "__main__":
    main() 