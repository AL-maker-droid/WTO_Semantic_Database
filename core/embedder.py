class BaseEmbedder:
    def __init__(self, domain_config=None):
        """
        Flexible embedding with optional domain configuration
        """
        self.domain_config = domain_config or {}
    
    def process_document(self, text):
        """
        Generic document processing method
        Can be extended/overridden by specific implementations
        """
        raise NotImplementedError() 