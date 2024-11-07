from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from domain.degrowth import DegrowthOntology
from domain.legal_trade import LegalTradeOntology
import spacy
import numpy as np
from pathlib import Path
import json
import re
import torch

class HierarchicalChunker:
    def __init__(self):
        # Load spaCy model for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_sections(self, text):
        """Extract major sections based on headers"""
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
        paragraphs = [p.strip() for p in section.split('\n\n')]
        return [p for p in paragraphs if p and len(p.split()) > 5]
    
    def extract_sentences(self, paragraph):
        """Extract sentences using spaCy"""
        doc = self.nlp(paragraph)
        return [str(sent).strip() for sent in doc.sents if len(str(sent).split()) > 3]

class LegalBERTEmbedder:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        """
        Initialize Legal-BERT embedding model with robust error handling
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f"Successfully initialized Legal-BERT model: {model_name}")
        except Exception as e:
            print(f"Error initializing Legal-BERT model: {e}")
            raise
    
    def embed_text(self, texts, max_length=512):
        """
        Generate embeddings for multiple texts with mean pooling
        
        Args:
            texts (list): List of text strings
            max_length (int): Maximum sequence length
        
        Returns:
            torch.Tensor: Embeddings for input texts
        """
        # Tokenize batch of texts
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = sum_embeddings / sum_mask
        
        return embeddings

class HierarchicalEmbedder:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        """
        Robust model initialization with Legal-BERT
        """
        # Initialize Legal-BERT embedder
        self.legal_bert = LegalBERTEmbedder(model_name)
        
        # Text chunking utility
        self.chunker = HierarchicalChunker()
        
        # Domain knowledge for context enrichment
        self.domain_knowledge = self._load_domain_knowledge()
    
    def _load_domain_knowledge(self):
        """
        Combine domain-specific ontologies for comprehensive context
        
        Returns:
            list: Combined list of concepts from multiple domains
        """
        degrowth_concepts = DegrowthOntology().get_concepts()
        legal_trade_concepts = LegalTradeOntology().get_concepts()
        
        # Combine and deduplicate concepts
        return list(set(degrowth_concepts + legal_trade_concepts))
    
    def process_document(self, text, document_name=None):
        """
        Process document and generate hierarchical embeddings
        
        Args:
            text (str): Full document text
            document_name (str, optional): Name of the source document
        
        Returns:
            dict: Hierarchical embeddings with context and metadata
        """
        # Create hierarchical structure with metadata
        hierarchy = {
            'metadata': {
                'document_name': document_name or 'Unknown Document',
                'total_sections': 0,
                'total_paragraphs': 0,
                'total_sentences': 0
            },
            'document': {
                'text': text,
                'embedding': self.generate_embeddings([text])[0].tolist()
            },
            'sections': []
        }
        
        # Process sections
        sections = self.chunker.extract_sections(text)
        hierarchy['metadata']['total_sections'] = len(sections)
        
        for section_text in sections:
            section_emb = {
                'text': section_text,
                'embedding': self.generate_embeddings([section_text])[0].tolist(),
                'paragraphs': []
            }
            
            # Process paragraphs
            paragraphs = self.chunker.extract_paragraphs(section_text)
            hierarchy['metadata']['total_paragraphs'] += len(paragraphs)
            
            for para_text in paragraphs:
                para_emb = {
                    'text': para_text,
                    'embedding': self.generate_embeddings([para_text])[0].tolist(),
                    'sentences': []
                }
                
                # Process sentences
                sentences = self.chunker.extract_sentences(para_text)
                hierarchy['metadata']['total_sentences'] += len(sentences)
                
                for sent_text in sentences:
                    sent_emb = {
                        'text': sent_text,
                        'embedding': self.generate_embeddings([sent_text])[0].tolist()
                    }
                    para_emb['sentences'].append(sent_emb)
                
                section_emb['paragraphs'].append(para_emb)
            
            hierarchy['sections'].append(section_emb)
        
        # Enrich embeddings with domain context
        return self.enrich_embeddings(hierarchy)
    
    def generate_embeddings(self, text_chunks):
        """
        Generate embeddings using Legal-BERT
        
        Args:
            text_chunks (list): List of text strings to embed
        
        Returns:
            torch.Tensor: Embeddings for input text chunks
        """
        return self.legal_bert.embed_text(text_chunks)
    
    def enrich_embeddings(self, embeddings):
        """
        Enrich embeddings with domain-specific context
        
        Args:
            embeddings (dict): Hierarchical embeddings
        
        Returns:
            dict: Enriched embeddings with domain context
        """
        def add_domain_context(embedding, text):
            """Add domain-specific context tags"""
            context_tags = [
                tag for tag in self.domain_knowledge
                if tag in text.lower()
            ]
            embedding['domain_context'] = context_tags
            return embedding
        
        def process_hierarchical_embeddings(obj):
            """Recursively process embeddings"""
            if isinstance(obj, dict):
                if 'text' in obj and 'embedding' in obj:
                    obj = add_domain_context(obj, obj['text'])
                
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        obj[key] = process_hierarchical_embeddings(value)
            
            elif isinstance(obj, list):
                obj = [process_hierarchical_embeddings(item) for item in obj]
            
            return obj
        
        return process_hierarchical_embeddings(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to disk with JSON serialization"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2)

if __name__ == "__main__":
    # Load extracted text
    input_path = Path("data/processed/LT:UR:A-1A:2_extracted.txt")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create embeddings
    embedder = HierarchicalEmbedder()
    embeddings = embedder.process_document(text)
    
    # Enrich embeddings
    enriched_embeddings = embedder.enrich_embeddings(embeddings)
    
    # Save embeddings
    output_path = Path("data/processed/hierarchical_embeddings.json")
    save_embeddings(enriched_embeddings, output_path)
    print(f"Hierarchical embeddings saved to {output_path}") 
