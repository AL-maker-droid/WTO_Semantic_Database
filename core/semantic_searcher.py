from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import torch
import spacy

# Import the new query augmenter
from domain.domain_query import QueryAugmenter

class LegalBERTSearcher:
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
            
            print(f"Successfully initialized Legal-BERT model for searching: {model_name}")
        except Exception as e:
            print(f"Error initializing Legal-BERT model: {e}")
            raise
    
    def embed_text(self, texts, max_length=512):
        """
        Generate embeddings for multiple texts with mean pooling
        """
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embeddings = sum_embeddings / sum_mask
        
        return embeddings

class AdvancedSemanticSearcher:
    def __init__(self, embeddings_path: Path, model_name="nlpaueb/legal-bert-base-uncased"):
        """
        Advanced semantic search with multi-modal relevance assessment
        """
        # Embedding model
        self.legal_bert = LegalBERTSearcher(model_name)
        
        # NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Query augmentation
        self.query_augmenter = QueryAugmenter()
        
        # Load embeddings
        with open(embeddings_path, 'r') as f:
            self.hierarchical_embeddings = json.load(f)
    
    def keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword relevance using NLP techniques
        """
        doc = self.nlp(text.lower())
        
        # Token-level matching
        token_matches = sum(
            1 for token in doc 
            if token.lemma_ in keywords or 
               any(keyword in token.lemma_ for keyword in keywords)
        )
        
        # Normalized by text length
        return token_matches / len(doc) if len(doc) > 0 else 0
    
    def semantic_similarity(self, query: str, text: str) -> float:
        """
        Compute semantic similarity using embeddings
        """
        query_emb = self.legal_bert.embed_text([query])[0]
        text_emb = self.legal_bert.embed_text([text])[0]
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), 
            text_emb.unsqueeze(0)
        ).item()
        
        return cosine_sim
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict]:
        """
        Advanced semantic search with query augmentation
        """
        # Augment query
        augmented_queries = self.query_augmenter.augment_query(query)
        
        # Comprehensive search across augmented queries
        comprehensive_results = []
        for augmented_query in augmented_queries:
            # Score all sections
            scored_sections = [
                self._contextual_relevance_scoring(augmented_query, section)
                for section in self.hierarchical_embeddings['sections']
            ]
            
            # Filter and collect results
            relevant_sections = [
                {**section, 'metadata': self.hierarchical_embeddings.get('metadata', {})}
                for section in scored_sections 
                if section['score'] > threshold and section['paragraphs']
            ]
            
            comprehensive_results.extend(relevant_sections)
        
        # Deduplicate and sort results
        unique_results = {
            section['text']: section 
            for section in comprehensive_results
        }
        
        return sorted(
            unique_results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:top_k]
    
    def _contextual_relevance_scoring(self, query: str, section: Dict) -> Dict:
        """
        Multi-modal relevance assessment
        """
        # Extract key concepts from query for keyword matching
        key_concepts = self.query_augmenter.extract_key_concepts(query)
        
        # Compute scores
        semantic_score = self.semantic_similarity(query, section['text'])
        keyword_score = self.keyword_relevance(section['text'], key_concepts)
        
        # Weighted combination
        combined_score = (0.6 * semantic_score) + (0.4 * keyword_score)
        
        # Paragraph-level scoring
        paragraphs = []
        for para in section.get('paragraphs', []):
            para_semantic = self.semantic_similarity(query, para['text'])
            para_keyword = self.keyword_relevance(para['text'], key_concepts)
            para_combined_score = (0.6 * para_semantic) + (0.4 * para_keyword)
            
            if para_combined_score > 0.3:  # Adjustable threshold
                paragraphs.append({
                    'text': para['text'],
                    'score': para_combined_score
                })
        
        return {
            'text': section['text'],
            'score': combined_score,
            'paragraphs': paragraphs
        }
    
    def hierarchical_search(self, query: str, top_k: int = 3, threshold: float = 0.4) -> List[Dict]:
        """
        Hierarchical semantic search with progressive refinement
        """
        # Augment query
        augmented_queries = self.query_augmenter.augment_query(query)
        
        # Comprehensive hierarchical search
        comprehensive_results = []
        for augmented_query in augmented_queries:
            # First find relevant sections
            scored_sections = []
            for section in self.hierarchical_embeddings['sections']:
                section_score = self.semantic_similarity(augmented_query, section['text'])
                
                if section_score > threshold:
                    # For relevant sections, find relevant paragraphs
                    scored_paragraphs = []
                    for para in section['paragraphs']:
                        para_score = self.semantic_similarity(augmented_query, para['text'])
                        
                        if para_score > threshold:
                            # For relevant paragraphs, find key sentences
                            scored_sentences = []
                            for sent in para.get('sentences', []):
                                sent_score = self.semantic_similarity(augmented_query, sent['text'])
                                if sent_score > threshold:
                                    scored_sentences.append({
                                        'text': sent['text'],
                                        'score': sent_score
                                    })
                            
                            # Only include paragraphs with relevant sentences
                            if scored_sentences:
                                scored_paragraphs.append({
                                    'text': para['text'],
                                    'score': para_score,
                                    'key_sentences': sorted(
                                        scored_sentences,
                                        key=lambda x: x['score'],
                                        reverse=True
                                    )
                                })
                    
                    # Only include sections with relevant paragraphs
                    if scored_paragraphs:
                        scored_sections.append({
                            'title': self._extract_section_title(section['text']),
                            'text': section['text'],
                            'score': section_score,
                            'paragraphs': sorted(
                                scored_paragraphs,
                                key=lambda x: x['score'],
                                reverse=True
                            )
                        })
            
            comprehensive_results.extend(scored_sections)
        
        # Deduplicate and return top results
        unique_results = {
            section['text']: section 
            for section in comprehensive_results
        }
        
        return sorted(
            unique_results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:top_k]
    
    def _extract_section_title(self, section_text: str, max_length: int = 100) -> str:
        """
        Extract a concise section title
        """
        # Remove citations and references
        clean_text = section_text.split('\n')[0]  # Take first line
        return clean_text[:max_length] + '...' if len(clean_text) > max_length else clean_text

# Existing print functions remain the same 

def print_advanced_results(results: List[Dict]):
    """
    Enhanced result printing with detailed scoring, document name, and metadata
    """
    print("\nAdvanced Search Results:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        # Extract document metadata if available
        document_name = result.get('metadata', {}).get('document_name', 'Unknown Document')
        
        print(f"\nResult {i}:")
        print(f"Document: {document_name}")
        print(f"Overall Relevance: {result['score']:.4f}")
        print(f"Section Preview: {result['text'][:300]}...\n")
        
        print("Relevant Paragraphs:")
        for j, para in enumerate(result['paragraphs'], 1):
            print(f"  {j}. Paragraph Relevance: {para['score']:.4f}")
            print(f"     {para['text'][:250]}...\n")
        
        print("-" * 80)