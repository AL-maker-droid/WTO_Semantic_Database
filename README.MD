# WTO Semantic Database: Advanced Document Analysis Toolkit

## Project Overview
A sophisticated semantic search and analysis toolkit designed for processing and extracting insights from complex legal and policy documents, with a specific focus on trade, environmental policy, and sustainability.

## Codebase Structure

### Core Components
1. `semantic_document_analyzer.py`
   - High-level orchestration script
   - Manages single and batch document processing
   - Provides unified interface for semantic search

2. `Single_Document_Processing.py`
   - Processes individual PDF documents
   - Extracts text
   - Generates hierarchical embeddings
   - Performs semantic search
   - Saves results in multiple formats

3. `Batch_Document_Processing.py`
   - Processes multiple PDFs in parallel
   - Supports batch processing with configurable batch sizes
   - Provides progress tracking
   - Generates results for multiple documents

### Core Modules
- `core/pdf_text_extractor.py`: PDF text extraction
- `core/hierarchical_embedder.py`: Advanced embedding generation
- `core/semantic_searcher.py`: Semantic search and relevance scoring
- `domain/`: Domain-specific ontologies and query augmentation

## Processing Pipeline

1. **Text Extraction**
   - Convert PDF to raw text
   - Clean and preprocess text
   - Segment into hierarchical structure (document → sections → paragraphs → sentences)

2. **Embedding Generation**
   - Use Legal-BERT for contextual embeddings
   - Generate embeddings at multiple granularity levels
   - Enrich with domain-specific context

3. **Semantic Search**
   - Query augmentation
   - Multi-level semantic similarity scoring
   - Hierarchical result retrieval

4. **Result Presentation**
   - Machine-readable (JSONL)
   - Human-readable (Markdown)
   - Detailed relevance scoring

## Model Choices

### Embedding Model
- **Legal-BERT**: Specialized for legal domain
- Advantages:
  - Domain-specific pre-training
  - Better understanding of legal terminology
  - Contextual embedding generation

### Embedding Strategy
- Hierarchical embeddings
- Multi-level semantic representation
- Domain context enrichment

## Limitations

1. **Computational Resources**
   - Requires significant RAM
   - GPU recommended for faster processing
   - Large documents might need optimization

2. **Model Constraints**
   - Limited by Legal-BERT's training data
   - Potential bias in semantic understanding
   - Performance varies across document types

3. **Language Support**
   - Currently English-only
   - Limited multi-lingual capabilities

## Potential Improvements

1. Multi-lingual support
2. More sophisticated domain ontologies
3. Advanced query expansion techniques
4. Integrate more advanced NLP models
5. Add citation and reference tracking

## Installation

### Clone repository
git clone https://github.com/AL-maker-droid/WTO_Semantic_Database
cd wto-semantic-database
### Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
### Install dependencies
pip install -r requirements.txt
Download spaCy model
python -m spacy download en_core_web_sm

### Basic Usage 
#### Single Document Processing
```bash
python semantic_document_analyzer.py \
--mode single \
--pdf data/raw/document.pdf \
--queries "Environmental protection" "Trade policy"
```
#### Batch Document Processing 
```bash
python semantic_document_analyzer.py \
--mode batch \
--dir data/raw/pdfs \
--queries "Climate change" "Economic sustainability"
```
#### Custom scripts creation 

1. Leverage existing modules
2. Extend `HierarchicalEmbedder`
3. Create custom query augmentation
4. Implement specialized search strategies

Example custom script template:
```python
from core.hierarchical_embedder import HierarchicalEmbedder
from core.semantic_searcher import AdvancedSemanticSearcher
class CustomDocumentAnalyzer:
def init(self, custom_model=None):
self.embedder = HierarchicalEmbedder(custom_model)
def specialized_search(self, document_path, custom_queries)
#Implement your specialized search logic
pass
def generate_insights(self, search_results):
# Create custom insight generation
pass
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push and create pull request

## License
MIT License

## References 
@inproceedings{chalkidis-etal-2020-legal,
    title = "{LEGAL}-{BERT}: The Muppets straight out of Law School",
    author = "Chalkidis, Ilias  and
      Fergadiotis, Manos  and
      Malakasiotis, Prodromos  and
      Aletras, Nikolaos  and
      Androutsopoulos, Ion",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2020.findings-emnlp.261",
    pages = "2898--2904"
}



