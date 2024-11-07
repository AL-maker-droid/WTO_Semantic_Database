import PyPDF2
from pathlib import Path

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extract text from PDF using PyPDF2 and save to a text file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            with open(output_path, 'w', encoding='utf-8') as out_file:
                # Write basic document info
                out_file.write(f"Document: {pdf_path.name}\n")
                out_file.write(f"Number of pages: {len(pdf_reader.pages)}\n")
                out_file.write("-" * 80 + "\n\n")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    # Write page header and content
                    out_file.write(f"\n{'='*40} Page {page_num} {'='*40}\n\n")
                    out_file.write(text)
                    out_file.write("\n")
        
        print(f"Text extracted and saved to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    # Setup paths
    current_dir = Path.cwd()
    pdf_path = current_dir / "data" / "raw" / "Doc1.pdf"
    output_path = current_dir / "data" / "processed" / "Doc1_extracted.txt"
    
    # Extract text
    extract_text_from_pdf(pdf_path, output_path) 
