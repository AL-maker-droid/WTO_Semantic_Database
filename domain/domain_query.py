from typing import List, Dict, Optional
import spacy
import nltk
from nltk.corpus import wordnet as wn

# Download necessary NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class QueryAugmenter:
    def __init__(self, domains: Optional[List[str]] = None):
        """
        Flexible query augmentation system
        
        Args:
            domains (List[str], optional): Domains to consider
        """
        # Load spaCy for advanced NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Default domains if none provided
        self.domains = domains or [
            'academic', 
            'economic', 
            'environmental', 
            'social', 
            'policy'
        ]
        
        # Configurable expansion strategies
        self.expansion_strategies = {
            'academic': [
                "scholarly perspective on {query}",
                "research implications of {query}",
                "theoretical framework of {query}"
            ],
            'economic': [
                "{query} in economic context",
                "economic policy implications of {query}",
                "market perspectives on {query}"
            ],
            'environmental': [
                "ecological dimensions of {query}",
                "sustainability context of {query}",
                "environmental impact of {query}"
            ],
            'social': [
                "social justice perspective on {query}",
                "community implications of {query}",
                "equity considerations of {query}"
            ],
            'policy': [
                "policy framework for {query}",
                "regulatory implications of {query}",
                "systemic approach to {query}"
            ]
        }
    
    def get_wordnet_synonyms(self, word: str, limit: int = 3) -> List[str]:
        """
        Retrieve WordNet synonyms for a given word
        
        Args:
            word (str): Input word
            limit (int): Maximum number of synonyms to return
        
        Returns:
            List[str]: List of synonyms
        """
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                # Avoid underscores and duplicates
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
                    if len(synonyms) >= limit:
                        break
            if len(synonyms) >= limit:
                break
        
        return list(synonyms)
    
    def extract_key_concepts(self, query: str) -> List[str]:
        """
        Extract key conceptual terms from the query
        
        Args:
            query (str): Input query
        
        Returns:
            List[str]: Key conceptual terms
        """
        doc = self.nlp(query)
        
        # Focus on nouns and proper nouns
        key_terms = [
            token.lemma_ for token in doc 
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.lemma_) > 2
        ]
        
        return list(set(key_terms))
    
    def augment_query(
        self, 
        query: str, 
        strategies: Optional[List[str]] = None,
        include_synonyms: bool = True
    ) -> List[str]:
        """
        Augment query with domain-specific and semantic expansions
        
        Args:
            query (str): Original query
            strategies (List[str], optional): Specific strategies to use
            include_synonyms (bool): Whether to include WordNet synonyms
        
        Returns:
            List[str]: Augmented queries
        """
        # Default to all domains if no strategies specified
        strategies = strategies or self.domains
        
        # Start with original query
        augmented_queries = [query]
        
        # Extract key concepts
        key_terms = self.extract_key_concepts(query)
        
        # Add synonyms if requested
        if include_synonyms:
            for term in key_terms:
                synonyms = self.get_wordnet_synonyms(term)
                augmented_queries.extend(synonyms)
        
        # Add domain-specific expansions
        for domain in strategies:
            if domain in self.expansion_strategies:
                domain_expansions = [
                    expansion.format(query=query) 
                    for expansion in self.expansion_strategies[domain]
                ]
                augmented_queries.extend(domain_expansions)
        
        return list(set(augmented_queries)) 