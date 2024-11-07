import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extract text from PDF and save to a text file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text
    """
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            # Write basic document info
            out_file.write(f"Document: {pdf_path.name}\n")
            out_file.write(f"Number of pages: {len(pdf_document)}\n")
            out_file.write("-" * 80 + "\n\n")
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_document, 1):
                text = page.get_text()
                
                # Write page header and content
                out_file.write(f"\n{'='*40} Page {page_num} {'='*40}\n\n")
                out_file.write(text)
                out_file.write("\n")
                
        print(f"Text extracted and saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

if __name__ == "__main__":
    # Setup paths
    current_dir = Path.cwd()
    pdf_path = current_dir / "data" / "raw" / "LT:UR:A-1A:2 copy.pdf"
    output_path = current_dir / "data" / "processed" / "LT:UR:A-1A:2_extracted.txt"
    
    # Extract text
    extract_text_from_pdf(pdf_path, output_path) 