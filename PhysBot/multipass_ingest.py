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
import time

nltk.download("punkt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def extract_unit_number_from_filename(pdf_path):
    match = re.search(r"Unit\s*(\d+)", os.path.basename(pdf_path))
    return match.group(1) if match else "Unknown"

def call_openai_with_retry(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an AI assistant extracting LaTeX equations."},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}. Retrying {attempt + 1}/{retries}...")
            time.sleep(delay)
    return None

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

### PASS 4: EXTRACT EQUATIONS USING OpenAI
import subprocess
import json
import logging
import re

def extract_equations_with_openai(text_chunks, unit_number):
    """
    Sends extracted raw text from PyMuPDF to OpenAI API for equation extraction.

    Parameters:
    - text_chunks (list of dicts): List of extracted text chunks, each with 'content' and 'page' keys.
    - unit_number (str): Chapter or unit identifier (e.g., "105").

    Returns:
    - extracted_equations (list of dicts): Extracted equations and their metadata.
    """
    structured_equations = []
    client = OpenAI()

    for idx, chunk in enumerate(tqdm(text_chunks, desc=f"Extracting Equations for Unit {unit_number}", unit="chunk")):
        prompt = f"""
        Identify and extract any mathematical equations from the following text.
        Return:
        - The equation in LaTeX format
        - A placeholder identifier for reintegration (format: <<UNIT_{unit_number}_EQ_X>>)
        - The page number from which it was extracted

        Text: {chunk['content']}

        Format the output as a JSON array:
        [{{"placeholder": "<<UNIT_{unit_number}_EQ_X>>", "equation": "", "page": X}}]
        """

        try:
            response_text = call_openai_with_retry(prompt)
            if not response_text:
                logging.error(f"Failed to extract equations for chunk {idx}.")
                continue

            #response_text = response.choices[0].message.content.strip()

            # Clean response of markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            # Validate and parse JSON
            try:
                try:
                    extracted = json.loads(response_text)
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON from OpenAI: {response_text}")
                    extracted = []

                for eq in extracted:
                    eq["page"] = chunk["page"]
                    eq["placeholder"] = eq["placeholder"].replace("X", str(len(structured_equations) + 1))  # Ensure numbering
                structured_equations.extend(extracted)

            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response for page {chunk['page']}: {response_text}")

        except Exception as e:
            logging.error(f"OpenAI API error on page {chunk['page']}: {str(e)}")

    return structured_equations

### PASS 5: Extract figures
def extract_figures(pdf_path, output_dir):
    """
    Extracts figures and captions, saving images separately.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    figures = []

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            image = doc.extract_image(xref)
            img_path = os.path.join(output_dir, f"figure_{page_num+1}_{img_index}.png")
            with open(img_path, "wb") as f:
                f.write(image["image"])
            figures.append({"image_path": img_path, "page": page_num + 1})

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

def reintegrate_equations(md_text, equations):
    for eq in equations:
        placeholder = eq["placeholder"]
        formatted_eq = f"$$ {eq['content']} $$" if "\n" in eq["content"] else f"$ {eq['content']} $"
        md_text = md_text.replace(placeholder, formatted_eq)
    return md_text

### MAIN PROCESSING FUNCTION
def process_pdf(pdf_path, output_json):
    """
    Full pipeline for processing a chapter PDF with multi-pass extraction.
    """
    logging.info(f"Processing {pdf_path}")

    # Step 1: Convert PDF to Markdown
    md_text = convert_pdf_to_md(pdf_path)

    # Step 2: Extract metadata
    metadata = extract_metadata_from_md(md_text)

    # Step 3: Extract text chunks
    text_chunks = chunk_text(md_text)

    # Step 4: Extract figures
    figures = extract_figures(pdf_path, FIGURE_DIR)

    # Step 5: Extract equations via OpenAI API
    equations = extract_equations_with_openai(text_chunks, unit_number)

    # 🔥🔥🔥 Step 6: Reintegration of equations 🔥🔥🔥
    md_text = reintegrate_equations(md_text, equations)

    # Step 7: Save JSON Output
    final_output = {
        "unit": metadata.get("unit", "Unknown"),
        "sections": metadata.get("sections", []),
        "text_chunks": text_chunks,
        "equations": equations,
        "figures": figures,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    logging.info(f"Processing complete. Data saved to {output_json}")

### RUN SCRIPT
if __name__ == "__main__":
    test_pdf = "data/chapters/Unit 105 - Motion and Kinetic Energy.pdf"
    output_json = "data/json/unit_105_output.json"
    process_pdf(test_pdf, output_json)