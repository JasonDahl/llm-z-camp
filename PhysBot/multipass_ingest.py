import os
import re
import json
import logging
import shutil
import fitz  # PyMuPDF
import nltk
import subprocess
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import pymupdf4llm
import unicodedata

nltk.download("punkt")

# Ensure logging is properly set up
logging.basicConfig(
    filename="multipass_ingest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ensure necessary directories exist
OUTPUT_DIR = "data/json"
FIGURE_DIR = "data/figures"
MARKDOWN_DIR = "data/markdown"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)

### PASS 1: CONVERT PDF TO MARKDOWN
def convert_pdf_to_md(pdf_path, output_dir="data/markdown/", reuse_existing=False):
    """
    Converts a PDF to Markdown using PyMuPDF4LLM and saves it to a specified output directory.

    Parameters:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where the Markdown file should be saved.
        reuse_existing (bool): If True, loads an existing Markdown file instead of re-processing.

    Returns:
        str: The extracted Markdown text.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename without extension
    pdf_filename = os.path.basename(pdf_path)
    md_filename = os.path.splitext(pdf_filename)[0] + ".md"
    output_md_path = os.path.join(output_dir, md_filename)

    # If the Markdown file exists and reuse_existing is True, load it
    if reuse_existing and os.path.exists(output_md_path):
        logging.info(f"Loading existing Markdown from {output_md_path}")
        with open(output_md_path, "r", encoding="utf-8") as md_file:
            return md_file.read()

    # Otherwise, convert the PDF to Markdown
    logging.info(f"Converting {pdf_path} to Markdown")
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Save Markdown to a file
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_text)

    logging.info(f"Saved Markdown to {output_md_path}")
    return md_text

### PASS 2: METADATA EXTRACTION
def extract_metadata(md_text, pdf_path):
    """
    Extracts unit title and section headers from Markdown.
    Falls back to PDF filename if necessary.
    """
    logging.info(f"Extracting metadata from {pdf_path}")
    
    unit_title = None
    section_headers = []
    
    lines = md_text.split("\n")
    
    for line in lines:
        line = line.strip()
        
        # Detect unit title
        match = re.match(r"^#\s*Unit\s*(\d+)\s*[-–]\s*(.+)", line)
        if match and not unit_title:
            unit_title = f"Unit {match.group(1)} - {match.group(2)}"
        
        # Detect section headers
        section_match = re.match(r"^##\s*(.+)", line)
        if section_match:
            section_headers.append(section_match.group(1))
    
    # Fallback to PDF filename if needed
    if not unit_title:
        filename = os.path.basename(pdf_path)
        match = re.match(r"(Unit\s*\d+\s*-\s*.+)\.pdf", filename)
        if match:
            unit_title = match.group(1)
        else:
            unit_title = "Unknown Unit"
            logging.warning("Could not extract unit title, falling back to 'Unknown Unit'")
    
    metadata = {"unit": unit_title, "sections": section_headers}
    return metadata

### PASS 3: CHUNKING TEXT FOR LLM
def chunk_text(md_text, chunk_size=500, overlap=75):
    """
    Tokenizes and chunks Markdown text into structured JSON.
    """
    tokens = nltk.word_tokenize(md_text)
    chunks = []
    buffer = []

    for token in tokens:
        buffer.append(token)
        if len(buffer) >= chunk_size:
            chunks.append(" ".join(buffer))
            buffer = buffer[-overlap:]  # Maintain overlap

    return [{"chunk_type": "text", "content": chunk} for chunk in chunks]

### PASS 4: EXTRACT EQUATIONS USING PANDOC
import subprocess
import json
import logging
import re

def extract_equations_with_pandoc(md_text):
    """
    Extracts LaTeX equations from Markdown using Pandoc's JSON AST format.
    Ensures proper equation formatting and replaces them with placeholders.
    """
    logging.info("Extracting equations using Pandoc")
    equations = []
    
    try:
        # Convert Markdown to JSON AST with Pandoc
        result = subprocess.run(
            ["pandoc", "--from=markdown", "--to=json"],
            input=md_text,
            text=True,
            capture_output=True,
            check=True,
        )
        ast = json.loads(result.stdout)
        
        # Traverse AST to extract equations
        def find_equations(obj):
            if isinstance(obj, dict):
                if obj.get("t") in ["Math", "RawInline"]:
                    eq_text = obj["c"][-1]
                    
                    # Normalize Unicode to prevent garbled characters
                    eq_text = unicodedata.normalize("NFKC", eq_text)

                    # Ensure valid LaTeX formatting (block vs inline)
                    if obj["t"] == "Math":
                        if obj["c"][0]["t"] == "DisplayMath":
                            eq_text = f"$$ {eq_text} $$"
                        else:
                            eq_text = f"$ {eq_text} $"

                    equations.append(eq_text)

                for v in obj.values():
                    find_equations(v)
            elif isinstance(obj, list):
                for v in obj:
                    find_equations(v)

        find_equations(ast)
        
        # Replace extracted equations in text with placeholders
        for i, eq in enumerate(equations):
            placeholder = f"<<EQUATION_{i+1}>>"
            md_text = re.sub(re.escape(eq), placeholder, md_text, count=1)

    except subprocess.CalledProcessError as e:
        logging.error(f"Pandoc equation extraction failed: {e}")
    
    return md_text, [{"id": f"<<EQUATION_{i+1}>>", "content": eq} for i, eq in enumerate(equations)]

### PASS 5: FIGURE EXTRACTION
def extract_figures(pdf_path, image_output_folder):
    """
    Extracts figures from PDF pages and saves them as separate image files.
    """
    logging.info(f"Extracting figures from {pdf_path}")
    doc = fitz.open(pdf_path)
    figures = []
    
    for page_num in tqdm(range(len(doc)), desc="Processing Figures"):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]  # Image reference number
            img_data = doc.extract_image(xref)
            img_bytes = img_data["image"]
            img_ext = img_data["ext"]
            
            img_filename = f"{image_output_folder}/figure_{page_num+1}_{img_index}.{img_ext}"
            with open(img_filename, "wb") as img_file:
                img_file.write(img_bytes)
            
            figures.append({
                "page": page_num + 1,
                "image_path": img_filename,
            })

    return figures

def validate_extraction(output_data):
    """
    Validates the extracted data to ensure completeness and correctness.
    Logs warnings if any issues are detected.
    """
    logging.info("Validating extracted data...")

    # Check for missing metadata
    if not output_data.get("metadata") or not output_data["metadata"].get("unit"):
        logging.warning("⚠️ Missing unit metadata!")

    if not output_data.get("metadata") or not output_data["metadata"].get("sections"):
        logging.warning("⚠️ No sections found in metadata!")

    # Check for empty text chunks
    if not output_data.get("text_chunks"):
        logging.warning("⚠️ No text chunks extracted!")
    else:
        empty_chunks = [chunk for chunk in output_data["text_chunks"] if not chunk["content"].strip()]
        if empty_chunks:
            logging.warning(f"⚠️ Found {len(empty_chunks)} empty text chunks!")

    # Check if equations were successfully extracted
    if not output_data.get("equations"):
        logging.warning("⚠️ No equations extracted!")

    # Ensure figure references exist
    if not output_data.get("figures"):
        logging.warning("⚠️ No figures extracted!")
    else:
        missing_images = [fig for fig in output_data["figures"] if not os.path.exists(fig["image_path"])]
        if missing_images:
            logging.warning(f"⚠️ {len(missing_images)} extracted figures have missing image files!")

    logging.info("Validation complete ✅")

### MAIN PROCESSING FUNCTION
def process_pdf(pdf_path, output_json):
    """
    Full multi-pass processing of a PDF file.
    """
    logging.info(f"Processing {pdf_path}")

    # PASS 1: Convert to Markdown
    md_text = convert_pdf_to_md(pdf_path, output_dir=MARKDOWN_DIR)

    # PASS 2: Extract Metadata
    metadata = extract_metadata(md_text, pdf_path)

    # PASS 3: Chunk Text
    text_chunks = chunk_text(md_text)

    # PASS 4: Extract Equations with Pandoc
    md_text, equations = extract_equations_with_pandoc(md_text)

    # PASS 5: Extract Figures
    figures = extract_figures(pdf_path, FIGURE_DIR)

    
    
    # Combine all extracted elements
    final_output = {
        "metadata": metadata,
        "text_chunks": text_chunks,
        "equations": equations,
        "figures": figures,
    }

    # ✅ **Validation step before saving**
    validate_extraction(final_output)

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    logging.info(f"Processing complete! Data saved to {output_json}")
    print(f"Processing complete! Data saved to {output_json}")

### RUN SCRIPT
if __name__ == "__main__":
    test_pdf = "data/chapters/Unit 105 - Motion and Kinetic Energy.pdf"
    output_json = "data/json/unit_105_output.json"
    process_pdf(test_pdf, output_json)