class DegrowthOntology:
    def __init__(self):
        self.domain_name = "degrowth"
        self.domain_description = "Degrowth is a concept that advocates for a reduction in economic growth to address environmental and social challenges. It emphasizes the need to transition towards a more sustainable and equitable economy."
        self._concepts = ["sustainability", "circular economy", "fair trade", "degrowth", "ecological economics", "resource conservation", "ecology", "sustainability", "conservation", "ecosystem", "environmental protection", "economic policy", "trade regulation", "economic localization", "conviviality","decolonization", "global south", "unequal exchange", "north south", "metabolism", "planetary boundaries"]
        self._synonyms = {
            "sustainability": ["resilience", "ecological balance"],
            "degrowth": ["post-growth", "alternative economics"],
            "ecological economics": ["environmental economics", "regenerative economics"]
        }
    
    def get_concepts(self):
        """
        Retrieve domain-specific concepts
        
        Returns:
            List of conceptual terms
        """
        return self._concepts
    
    def get_synonyms(self):
        """ 
        Retrieve domain-specific synonyms
        
        Returns:
            Dictionary of synonyms
        """
        return self._synonyms
    
    def get_description(self):
        """
        Retrieve domain description
        
        Returns:
            Domain description string
        """
        return self.domain_description
    
    # Maintain backwards compatibility
    def domain_ontology():
            return DegrowthOntology()
