from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from pathlib import Path
import json
import re

class HierarchicalChunker:
    def __init__(self):
        # Load spaCy model for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_sections(self, text):
        """Extract major sections based on Article headers"""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if re.match(r'^Article \d+|^ANNEX \d+|^Part [IVX]+', line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections
    
    def extract_paragraphs(self, section):
        """Extract paragraphs from a section"""
        # Split on double newlines and filter empty paragraphs
        paragraphs = [p.strip() for p in section.split('\n\n')]
        return [p for p in paragraphs if p and len(p.split()) > 5]
    
    def extract_sentences(self, paragraph):
        """Extract sentences using spaCy"""
        doc = self.nlp(paragraph)
        return [str(sent).strip() for sent in doc.sents if len(str(sent).split()) > 3]
    
    def create_hierarchy(self, text):
        """Create full hierarchical structure of the document"""
        hierarchy = {
            'document': text,
            'sections': []
        }
        
        sections = self.extract_sections(text)
        
        for section in sections:
            section_dict = {
                'text': section,
                'paragraphs': []
            }
            
            paragraphs = self.extract_paragraphs(section)
            for para in paragraphs:
                para_dict = {
                    'text': para,
                    'sentences': self.extract_sentences(para)
                }
                section_dict['paragraphs'].append(para_dict)
                
            hierarchy['sections'].append(section_dict)
            
        return hierarchy

class HierarchicalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.chunker = HierarchicalChunker()
        
    def generate_embeddings(self, text_chunks):
        """Generate embeddings for a list of text chunks"""
        return self.model.encode(text_chunks, convert_to_tensor=True)
    
    def process_document(self, text):
        """Process document and generate hierarchical embeddings"""
        # Create hierarchical structure
        hierarchy = self.chunker.create_hierarchy(text)
        
        # Generate embeddings at each level
        embeddings = {
            'document': {
                'text': hierarchy['document'],
                'embedding': self.generate_embeddings([hierarchy['document']])[0]
            },
            'sections': []
        }
        
        # Process sections
        for section in hierarchy['sections']:
            section_emb = {
                'text': section['text'],
                'embedding': self.generate_embeddings([section['text']])[0],
                'paragraphs': []
            }
            
            # Process paragraphs
            for para in section['paragraphs']:
                para_emb = {
                    'text': para['text'],
                    'embedding': self.generate_embeddings([para['text']])[0],
                    'sentences': []
                }
                
                # Process sentences
                for sent in para['sentences']:
                    sent_emb = {
                        'text': sent,
                        'embedding': self.generate_embeddings([sent])[0]
                    }
                    para_emb['sentences'].append(sent_emb)
                    
                section_emb['paragraphs'].append(para_emb)
            
            embeddings['sections'].append(section_emb)
            
        return embeddings

def save_embeddings(embeddings, output_path):
    """Save embeddings to disk"""
    # Convert embeddings to lists for JSON serialization
    def convert_embeddings(obj):
        if isinstance(obj, dict):
            return {k: convert_embeddings(v) if k != 'embedding' 
                   else v.tolist() for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_embeddings(item) for item in obj]
        return obj
    
    serializable_embeddings = convert_embeddings(embeddings)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_embeddings, f, indent=2)

if __name__ == "__main__":
    # Load extracted text
    input_path = Path("data/processed/LT:UR:A-1A:2_extracted.txt")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create embeddings
    embedder = HierarchicalEmbedder()
    embeddings = embedder.process_document(text)
    
    # Save embeddings
    output_path = Path("data/processed/hierarchical_embeddings.json")
    save_embeddings(embeddings, output_path)
    print(f"Hierarchical embeddings saved to {output_path}") 