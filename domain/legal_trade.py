class LegalTradeOntology:
    def __init__(self):
        self.domain_name = "legal_trade"
        self.domain_description = "Legal trade is a portmanteau of trade and law. It reflects the intersection of trade and law. Specifically with regard to the World Trade Organization (WTO) and the agreements and regulations that govern international trade."
        self._concepts = ["free trade", "trade agreement", "trade policy", "trade regulation", "trade negotiation", "trade dispute", "tariff", "subsidy", "import", "export", "customs", "duties", "quota", "sanction", "retaliation", "dispute settlement", "dispute resolution", "dispute settlement mechanism", "dispute settlement procedure", "dispute settlement process"]
        self._synonyms = {
            "trade": ["commerce", "exchange", "transaction"],
            "trade regulation": ["trade policy", "trade governance"],
            "dispute settlement": ["conflict resolution", "trade mediation"]
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
        return LegalTradeOntology()


