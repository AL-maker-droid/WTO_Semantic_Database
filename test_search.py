from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import torch
import numpy as np

def load_embeddings(embeddings_path):
    """Load and verify embeddings file"""
    print(f"\nLoading embeddings from: {embeddings_path}")
    
    with open(embeddings_path, 'r') as f:
        data = json.load(f)
    
    print("\nEmbeddings structure:")
    print(f"- Document level: {'embedding' in data['document']}")
    print(f"- Number of sections: {len(data['sections'])}")
    for i, section in enumerate(data['sections']):
        print(f"\nSection {i+1}:")
        print(f"- Has embedding: {'embedding' in section}")
        print(f"- Number of paragraphs: {len(section['paragraphs'])}")
        print(f"- First few words: {section['text'][:100]}...")
    
    return data

def compute_similarity(query_embedding, doc_embedding):
    """Compute cosine similarity between embeddings"""
    if isinstance(query_embedding, list):
        query_embedding = torch.tensor(query_embedding)
    if isinstance(doc_embedding, list):
        doc_embedding = torch.tensor(doc_embedding)
    query_embedding = torch.tensor(query_embedding)
    doc_embedding = torch.tensor(doc_embedding)
    
    similarity = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        doc_embedding.unsqueeze(0)
    )
    return float(similarity)

def manual_search(embeddings_path, query, threshold=0.3):
    """Manually perform semantic search"""
    print(f"\nPerforming search for query: '{query}'")
    print(f"Using threshold: {threshold}")
    
    # Load model and generate query embedding
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Load embeddings
    data = load_embeddings(embeddings_path)
    
    # Search through sections
    results = []
    print("\nSearching through sections...")
    
    for section_idx, section in enumerate(data['sections']):
        section_embedding = section['embedding']
        similarity = compute_similarity(query_embedding, section_embedding)
        
        if similarity > threshold:
            print(f"\nFound matching section (similarity: {similarity:.3f}):")
            print(f"First 100 chars: {section['text'][:100]}...")
            
            # Search through paragraphs
            for para in section['paragraphs']:
                para_similarity = compute_similarity(query_embedding, para['embedding'])
                if para_similarity > threshold:
                    results.append({
                        'section_idx': section_idx,
                        'similarity': para_similarity,
                        'text': para['text']
                    })
    
    # Sort and display results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nTop Results:")
    print("=" * 80)
    for result in results[:3]:
        print(f"\nSimilarity: {result['similarity']:.3f}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    embeddings_path = Path("data/processed/example_embeddings.json")
    
    # Test multiple queries
    queries = [
        "trade",
        "environmental protection",
        "animal welfare",
        "sustainable development"
    ]
    
    for query in queries:
        results = manual_search(embeddings_path, query)
        input("\nPress Enter to try next query...") 