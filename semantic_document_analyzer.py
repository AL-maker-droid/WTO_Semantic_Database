"""
Semantic Document Analysis Orchestrator

This script provides a comprehensive interface for semantic document analysis across single or multiple documents.

Key Features:
- Flexible query definition
- Support for single and batch document processing
- Configurable search parameters
- Automated text extraction, embedding generation, and semantic search

Workflow:
1. Define document sources (single PDF or directory of PDFs)
2. Specify semantic search queries
3. Configure search parameters (relevance threshold, top results)
4. Automatically:
   - Extract text from PDFs
   - Generate hierarchical embeddings
   - Perform semantic searches
   - Save results in machine and human-readable formats

Usage Examples:
1. Single Document:
   python semantic_document_analyzer.py 
     --mode single 
     --pdf data/raw/document.pdf 
     --queries "Environmental policy" "Trade regulations"

2. Batch Processing:
   python semantic_document_analyzer.py 
     --mode batch 
     --dir data/raw/pdfs 
     --queries "Climate change" "Economic sustainability"
"""

import argparse
from pathlib import Path
import sys

# Ensure the current directory is in the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from Single_Document_Processing import main as single_document_process
from Batch_Document_Processing import main as batch_document_process

def validate_inputs(args):
    """
    Validate user inputs and provide helpful error messages
    """
    if args.mode == 'single' and not args.pdf:
        raise ValueError("For single mode, --pdf is required")
    
    if args.mode == 'batch' and not args.dir:
        raise ValueError("For batch mode, --dir is required")
    
    if not args.queries:
        raise ValueError("At least one query is required")

def main():
    # Setup argument parser with comprehensive options
    parser = argparse.ArgumentParser(description='Semantic Document Analysis Orchestrator')
    
    # Mode selection
    parser.add_argument('--mode', 
                        choices=['single', 'batch'], 
                        default='single', 
                        help='Processing mode: single document or batch processing')
    
    # Document source
    parser.add_argument('--pdf', 
                        type=str, 
                        help='Path to single PDF for processing')
    parser.add_argument('--dir', 
                        type=str, 
                        help='Directory containing PDFs for batch processing')
    
    # Query configuration
    parser.add_argument('--queries', 
                        nargs='+', 
                        default=[
                            "Environmental protection in trade", 
                            "Sustainability and economic policy", 
                            "Ecological economics and degrowth"
                        ],
                        help='Semantic search queries')
    
    # Search parameters
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.3, 
                        help='Relevance threshold for search results')
    parser.add_argument('--top_k', 
                        type=int, 
                        default=3, 
                        help='Number of top results to display')
    
    # Reprocessing flag
    parser.add_argument('--force', 
                        action='store_true', 
                        help='Force reprocessing of documents')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Prepare arguments for processing functions
        process_args = ['--queries'] + args.queries + \
            ['--threshold', str(args.threshold), 
             '--top_k', str(args.top_k)]
        
        if args.force:
            process_args.append('--force')
        
        # Select processing mode
        if args.mode == 'single':
            process_args.extend(['--pdf', args.pdf])
            sys.argv = ['Single_Document_Processing.py'] + process_args
            single_document_process()
        
        elif args.mode == 'batch':
            process_args.extend(['--dir', args.dir])
            sys.argv = ['Batch_Document_Processing.py'] + process_args
            batch_document_process()
    
    except Exception as e:
        print(f"Error in document processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 