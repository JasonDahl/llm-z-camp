import os
import fitz  # PyMuPDF
from PyPDF2 import PdfMerger

def extract_first_page(pdf_folder, output_pdf):
    """
    Extracts the first page from each PDF file in the given folder and combines them into a single PDF.
    """
    merger = PdfMerger()
    temp_files = []
    
    for filename in sorted(os.listdir(pdf_folder)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            
            if len(doc) > 0:
                first_page = doc[0]
                temp_pdf_path = os.path.join(pdf_folder, f"temp_{filename}")
                temp_files.append(temp_pdf_path)
                
                # Save the first page to a temporary file
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=0, to_page=0)
                new_doc.save(temp_pdf_path)
                new_doc.close()
                
                merger.append(temp_pdf_path)
    
    # Save the final merged PDF
    merger.write(output_pdf)
    merger.close()
    
    # Cleanup temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    
    print(f"Bare Essentials pages saved to {output_pdf}")

