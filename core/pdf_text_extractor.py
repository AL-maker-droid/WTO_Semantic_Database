import PyPDF2
from pathlib import Path

def extract_text_from_pdf(input_path, output_path):
    """
    Extract text from a PDF file
    
    Args:
        input_path (Path or str): Path to the input PDF
        output_path (Path or str): Path to save extracted text
    """
    # Convert to Path objects if they aren't already
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Open the PDF
    with open(input_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages
        full_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            full_text.append(page_text)
        
        # Join the text from all pages
        extracted_text = "\n\n".join(full_text)

    # Write extracted text to file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(extracted_text)

    print(f"Text extracted to {output_path}")
    return output_path

if __name__ == "__main__":
    # Setup paths
    current_dir = Path.cwd()
    pdf_path = current_dir / "data" / "raw" / "Doc1.pdf"
    output_path = current_dir / "data" / "processed" / "Doc1_extracted.txt"
    
    # Extract text
    extract_text_from_pdf(pdf_path, output_path) 
