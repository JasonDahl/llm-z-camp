import os
import json
import re
import logging
import requests
import pdfplumber
import fitz  # PyMuPDF for image extraction
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from unstructured.partition.pdf import partition_pdf

# Ensure NLTK tokenization resources are available
nltk.download("punkt")

# Logging setup
logging.basicConfig(
    filename="pdf_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Output folders
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ================================
# PASS 1: METADATA EXTRACTION
# ================================
import requests
import re

def extract_metadata(pdf_path):
    """
    Extracts unit title, section headers, and page numbers using Unstructured.
    Falls back on extracting unit title from the filename if necessary.
    """
    elements = partition_pdf(filename=pdf_path, extract_images=False)
    
    unit_title = None
    section_headers = []

    # Try to extract from document text
    for element in elements:
        text = element.text.strip()
        
        match = re.match(r"^Unit\s*(\d+)\s*[-–]\s*(.+)", text)
        if match and not unit_title:
            unit_title = f"Unit {match.group(1)} - {match.group(2)}"

        section_match = re.match(r"^(\d+\.\d+)\s*[-–]\s*(.+)", text)
        if section_match:
            section_headers.append({"section": section_match.group(2), "page": element.metadata.page_number})

    # **Fallback: Extract from filename if document parsing fails**
    if not unit_title:
        filename = os.path.basename(pdf_path)
        filename_match = re.search(r"Unit\s*(\d+)\s*[-–]\s*(.+)\.pdf", filename, re.IGNORECASE)
        if filename_match:
            unit_title = f"Unit {filename_match.group(1)} - {filename_match.group(2)}"
            print(f"⚠️ Warning: Extracted unit title from filename -> {unit_title}")
        else:
            print("⚠️ Warning: Could not extract unit title from text or filename.")

    metadata = {"unit": unit_title, "sections": section_headers}
    return metadata


# ================================
# PASS 2: TEXT CHUNKING
# ================================
def extract_text_chunks(pdf_path, chunk_size=500, overlap=75):
    """
    Extracts and chunks text into LLM-friendly chunks (500-600 tokens).
    Uses pdfplumber for reliable text extraction.
    """
    logging.info(f"Extracting text from {pdf_path}")
    
    extracted_chunks = []
    buffer = []
    buffer_token_count = 0
    current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing Text")):
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")[1:]  # Skip first line (header)
            for line in lines:
                line = line.strip()

                # Detect section headers
                section_match = re.match(r"^(\d+\.\d+)\s*[-–]\s*(.+)", line)
                if section_match:
                    current_section = section_match.group(2)
                    continue
                
                tokens = word_tokenize(line)
                token_count = len(tokens)

                if buffer_token_count + token_count > chunk_size:
                    extracted_chunks.append({
                        "section": current_section,
                        "chunk_type": "paragraph",
                        "content": " ".join(buffer),
                        "page": page_num + 1,
                    })
                    buffer = buffer[-overlap:] + tokens
                    buffer_token_count = len(buffer)
                else:
                    buffer.extend(tokens)
                    buffer_token_count += token_count

            if buffer:
                extracted_chunks.append({
                    "section": current_section,
                    "chunk_type": "paragraph",
                    "content": " ".join(buffer),
                    "page": page_num + 1,
                })

    return extracted_chunks


# ================================
# PASS 3: EQUATION EXTRACTION
# ================================
def extract_equations(pdf_path):
    """
    Extracts LaTeX equations and replaces them with <<EQUATION_X>> placeholders.
    """
    logging.info(f"Extracting equations from {pdf_path}")
    
    equations = []
    equation_counter = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing Equations")):
            text = page.extract_text()
            if not text:
                continue

            matches = re.findall(r"(\$.*?\$)", text)
            for eq in matches:
                eq_placeholder = f"<<EQUATION_{equation_counter}>>"
                equations.append({
                    "equation_id": eq_placeholder,
                    "content": eq,
                    "page": page_num + 1,
                })
                text = text.replace(eq, eq_placeholder)
                equation_counter += 1

    return equations


# ================================
# PASS 4: FIGURE EXTRACTION
# ================================
def extract_figures(pdf_path):
    """
    Extracts figures and captions, saving images separately.
    """
    logging.info(f"Extracting figures from {pdf_path}")
    
    figures = []
    doc = fitz.open(pdf_path)

    for page_num in tqdm(range(len(doc)), desc="Processing Figures"):
        page = doc[page_num]

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_data = doc.extract_image(xref)
            img_ext = img_data["ext"]
            img_bytes = img_data["image"]

            img_filename = f"{FIGURE_DIR}/figure_{page_num+1}_{img_index}.{img_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(img_bytes)

            figures.append({
                "figure_id": f"FIGURE_{page_num+1}_{img_index}",
                "image_path": img_filename,
                "page": page_num + 1,
            })

    return figures


# ================================
# PASS 5: POST-PROCESSING & VALIDATION
# ================================
def validate_and_merge(metadata, text_chunks, equations, figures):
    """
    Merges all extracted components, ensuring consistency and linking elements.
    """
    logging.info("Validating and merging extracted data.")

    for chunk in text_chunks:
        chunk["unit"] = metadata.get("unit", "Unknown Unit")

    final_data = {
        "metadata": metadata,
        "text_chunks": text_chunks,
        "equations": equations,
        "figures": figures,
    }

    return final_data


# ================================
# MAIN PIPELINE
# ================================
def process_pdf(pdf_path, output_json="final_output.json"):
    """
    Executes the full multi-pass PDF extraction pipeline.
    """
    metadata = extract_metadata(pdf_path)
    text_chunks = extract_text_chunks(pdf_path)
    equations = extract_equations(pdf_path)
    figures = extract_figures(pdf_path)

    final_data = validate_and_merge(metadata, text_chunks, equations, figures)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)

    print(f"Extraction complete! Data saved to {output_json}")


# ================================
# RUN THE PIPELINE ON A TEST PDF
# ================================
if __name__ == "__main__":
    test_pdf = "data/chapters/Unit 105 - Motion and Kinetic Energy.pdf"
    process_pdf(test_pdf, output_json="data/json/unit_105_output.json")
