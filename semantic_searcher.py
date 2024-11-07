from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import torch

class SemanticSearcher:
    def __init__(self, embeddings_path: Path):
        """Initialize the searcher with pre-computed embeddings"""
        #print("\n=== Initialization ===")
        #print(f"PyTorch device availability:")
        #print(f"- MPS available: {torch.backends.mps.is_available()}")
        #print(f"- CUDA available: {torch.cuda.is_available()}")
        
        # Force CPU for model
        self.model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        #print(f"Model device: {next(self.model.parameters()).device}")
        #print("Loading embeddings from:", embeddings_path)
        
        # Load embeddings
        with open(embeddings_path, 'r') as f:
            self.hierarchical_embeddings = json.load(f)
        #print("Embeddings loaded, converting to tensors...")
        
        # Convert stored embeddings back to tensors
        self._convert_embeddings_to_tensors(self.hierarchical_embeddings)
        #print("Conversion complete")
    
    def _convert_embeddings_to_tensors(self, obj):
        """Recursively convert stored embeddings back to tensors"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'embedding':
                    tensor = torch.tensor(v)
                    obj[k] = tensor
                    #print(f"Converted embedding tensor device: {tensor.device}")
                elif isinstance(v, (dict, list)):
                    self._convert_embeddings_to_tensors(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._convert_embeddings_to_tensors(item)
    
    def compute_similarity(self, query_embedding: torch.Tensor, 
                         document_embedding: torch.Tensor) -> float:
        """Compute cosine similarity between query and document embeddings"""
        #print("\n=== Computing Similarity ===")
        #print(f"Query embedding device: {query_embedding.device}")
        #print(f"Document embedding device: {document_embedding.device}")
        
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            document_embedding.unsqueeze(0)
        )
        return float(similarity)
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """Search through the hierarchical embeddings using the query"""
        #print("\n=== Starting Search ===")
        #print(f"Generating embedding for query: {query}")
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        #print(f"Query embedding device after encode: {query_embedding.device}")
        #print(f"Query embedding shape: {query_embedding.shape}")

        #print("\nAttempting first similarity computation...")
        results = []
        
        # Search through sections
        for section_idx, section in enumerate(self.hierarchical_embeddings['sections']):
            section_similarity = self.compute_similarity(query_embedding, section['embedding'])
            
            if section_similarity > threshold:
                section_result = {
                    'section_text': section['text'].split('\n')[0],
                    'section_similarity': section_similarity,
                    'paragraphs': []
                }
                
                for para in section['paragraphs']:
                    para_similarity = self.compute_similarity(query_embedding, para['embedding'])
                    
                    if para_similarity > threshold:
                        para_result = {
                            'paragraph_text': para['text'],
                            'paragraph_similarity': para_similarity,
                            'sentences': []
                        }
                        
                        for sent in para['sentences']:
                            sent_similarity = self.compute_similarity(
                                query_embedding, sent['embedding']
                            )
                            
                            if sent_similarity > threshold:
                                para_result['sentences'].append({
                                    'sentence_text': sent['text'],
                                    'sentence_similarity': sent_similarity
                                })
                        
                        if para_result['sentences']:
                            section_result['paragraphs'].append(para_result)
                
                if section_result['paragraphs']:
                    results.append(section_result)
        
        results.sort(key=lambda x: x['section_similarity'], reverse=True)
        return results[:top_k]

def print_results(results: List[Dict]):
    """Pretty print the search results"""
    print("\nSearch Results:")
    print("=" * 80)
    
    for i, section in enumerate(results, 1):
        print(f"\n{i}. Section: {section['section_text']}")
        print(f"   Relevance: {section['section_similarity']:.3f}")
        
        for para in section['paragraphs']:
            print(f"\n   Paragraph (relevance: {para['paragraph_similarity']:.3f}):")
            print(f"   {para['paragraph_text'][:200]}...")
            
            print("\n   Most relevant sentences:")
            for sent in sorted(para['sentences'], 
                            key=lambda x: x['sentence_similarity'], 
                            reverse=True)[:3]:
                print(f"   - {sent['sentence_text']}")
                print(f"     Relevance: {sent['sentence_similarity']:.3f}")
        
        print("\n" + "-" * 80)

if __name__ == "__main__":
    # Initialize searcher
    embeddings_path = Path("hierarchical_embeddings.json")
    searcher = SemanticSearcher(embeddings_path)
    
    # Example queries related to environmental protection and non-human agency
    queries = [
        "Environmental protection and sustainability in trade",
        "Natural resource conservation and biodiversity",
        "Animal welfare and species protection",
        "Environmental impact of agricultural practices",
        "Conservation of natural resources and ecosystems"
    ]
    
    # Run searches
    for query in queries:
        print(f"\nQuery: {query}")
        results = searcher.search(query, top_k=3, threshold=0.4)
        print_results(results)

# Rest of the code remains the same... 