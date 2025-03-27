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
from dotenv import load_dotenv
from openai import OpenAI

nltk.download("punkt")

# Load environment variables
load_dotenv(".env")  # Update with correct path

# Retrieve OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is not set in utils.py. Check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

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

    return [{"chunk_type": "text", "content": chunk, "page": 0} for chunk in chunks]

### PASS 4: EXTRACT EQUATIONS USING OpenAI
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

            # Remove Markdown fencing if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            try:
                extracted = json.loads(response_text)
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON from OpenAI: {response_text}")
                extracted = []

            eq_counter = len(structured_equations) + 1
            for eq in extracted:
                eq["page"] = chunk["page"]
                eq["placeholder"] = f"<<UNIT_{unit_number}_EQ_{eq_counter}>>"
                eq_counter += 1
                structured_equations.append(eq)

        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")

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
    logging.info("🔍 Validating extracted data...")

    # Check for missing unit title
    if not output_data.get("unit"):
        logging.warning("⚠️ Missing unit metadata!")

    # Check for missing section headers
    if not output_data.get("sections"):
        logging.warning("⚠️ No sections found in metadata!")

    # Check for empty or missing text chunks
    text_chunks = output_data.get("text_chunks", [])
    if not text_chunks:
        logging.warning("⚠️ No text chunks extracted!")
    else:
        empty_chunks = [chunk for chunk in text_chunks if not chunk["content"].strip()]
        if empty_chunks:
            logging.warning(f"⚠️ Found {len(empty_chunks)} empty text chunks!")

    # Check if equations were successfully extracted
    if not output_data.get("equations"):
        logging.warning("⚠️ No equations extracted!")

    # Ensure figure image paths are valid
    figures = output_data.get("figures", [])
    if not figures:
        logging.warning("⚠️ No figures extracted!")
    else:
        missing_images = [fig for fig in figures if not os.path.exists(fig.get("image_path", ""))]
        if missing_images:
            logging.warning(f"⚠️ {len(missing_images)} extracted figures have missing image files!")

    # ✅ Summary
    logging.info(
        f"✅ Extraction Summary: {len(text_chunks)} text chunks, "
        f"{len(output_data.get('equations', []))} equations, "
        f"{len(figures)} figures, "
        f"{len(output_data.get('sections', []))} sections."
    )

    logging.info("✅ Validation complete.")

def reintegrate_equations(md_text, equations):
    """
    Replace placeholders in Markdown with LaTeX-formatted equations.
    """
    for eq in equations:
        latex = eq.get("equation") or eq.get("content", "")
        if not latex:
            continue  # skip malformed entries
        placeholder = eq["placeholder"]
        formatted_eq = f"$$ {latex} $$" if "\n" in latex else f"$ {latex} $"
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
    metadata = extract_metadata(md_text, pdf_path)

    # Step 3: Extract text chunks
    text_chunks = chunk_text(md_text)

    # Step 4: Extract figures
    figures = extract_figures(pdf_path, FIGURE_DIR)

    # Step 5: Extract equations via OpenAI API
    unit_number = extract_unit_number_from_filename(pdf_path)
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

    validate_extraction(final_output)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    logging.info(f"Processing complete. Data saved to {output_json}")

### RUN SCRIPT
if __name__ == "__main__":
    test_pdf = "data/chapters/Unit 105 - Motion and Kinetic Energy.pdf"
    output_json = "data/json/unit_105_output.json"
    process_pdf(test_pdf, output_json)