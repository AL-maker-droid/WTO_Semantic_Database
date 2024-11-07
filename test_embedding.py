from pathlib import Path
from hierarchical_embedder import HierarchicalEmbedder, save_embeddings

def main():
    # Load extracted text
    input_path = Path("data/processed/Doc1_extracted.txt")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create embeddings
    embedder = HierarchicalEmbedder()
    embeddings = embedder.process_document(text)
    
    # Enrich embeddings
    enriched_embeddings = embedder.enrich_embeddings(embeddings)
    
    # Save embeddings
    output_path = Path("data/processed/Doc1_hierarchical_embeddings.json")
    save_embeddings(enriched_embeddings, output_path)
    print(f"Hierarchical embeddings saved to {output_path}")

if __name__ == "__main__":
    main() 
