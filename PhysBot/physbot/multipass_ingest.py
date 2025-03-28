import os
import re
import json
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

import pymupdf4llm
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import word_tokenize

from physbot.utils import (
    setup_logging,
    get_openai_client
)

nltk.download("punkt")

# Load environment variables
load_dotenv("../.env")  

from physbot.utils import get_openai_client
client = get_openai_client()

# Ensure logging is properly set up
#logging.basicConfig(
#    filename="multipass_ingest.log",
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s",
#)

from physbot.constants import CHAPTER_DIR, JSON_DIR, MARKDOWN_DIR, FIGURE_DIR, LOG_DIR

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
    Retries up to 3 times if invalid JSON is returned.
    """
    structured_equations = []
    max_retries = 3

    for idx, chunk in enumerate(tqdm(text_chunks, desc=f"Extracting Equations for Unit {unit_number}", unit="chunk")):
        for attempt in range(1, max_retries + 1):
            try:
                prompt = f"""
                Identify and extract any mathematical equations from the following text.
                Return:
                - The equation in LaTeX format
                - A placeholder identifier for reintegration (format: <<UNIT_{unit_number}_EQ_X>>)
                - The page number from which it was extracted

                Text: {chunk.get('content', '')}

                Format the output as a JSON array:
                [{{"placeholder": "<<UNIT_{unit_number}_EQ_X>>", "equation": "", "page": X}}]
                """

                response = call_openai_with_retry(prompt)
                if not response:
                    logging.error(f"[Retry {attempt}/{max_retries}] No response for chunk {idx}")
                    continue

                response_text = response.strip()

                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                try:
                    extracted = json.loads(response_text)
                except json.JSONDecodeError:
                    logging.warning(f"[Retry {attempt}/{max_retries}] Invalid JSON from OpenAI for chunk {idx}: {response_text}")
                    if attempt == max_retries:
                        logging.error(f"❌ Giving up after {max_retries} failed attempts for chunk {idx}")
                    continue  # Retry
                except Exception as parse_err:
                    logging.error(f"Unexpected parse error for chunk {idx}: {str(parse_err)}")
                    break  # Abort this chunk

                # Assign consistent placeholders and store
                eq_counter = len(structured_equations) + 1
                for eq in extracted:
                    eq["page"] = chunk.get("page", None)
                    eq["placeholder"] = f"<<UNIT_{unit_number}_EQ_{eq_counter}>>"
                    eq_counter += 1
                    structured_equations.append(eq)

                break  # ✅ Success – break retry loop

            except Exception as e:
                logging.error(f"OpenAI API error on chunk {idx} (page {chunk.get('page', '?')}): {str(e)}")
                break  # Do not retry on API-level exception

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
def process_pdf(pdf_path, output_json, output_markdown_dir=None, output_figure_dir=None):
    """
    Full pipeline for processing a chapter PDF with multi-pass extraction.
    """
    logging.info(f"Processing {pdf_path}")

    # Step 1: Convert PDF to Markdown
    md_text = convert_pdf_to_md(pdf_path)

    if output_markdown_dir:
        markdown_path = Path(output_markdown_dir) / f"{Path(pdf_path).stem}.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    # Step 2: Extract metadata
    metadata = extract_metadata(md_text, pdf_path)

    # Step 3: Extract text chunks
    text_chunks = chunk_text(md_text)

    # Step 4: Extract figures
    unit_number = extract_unit_number_from_filename(pdf_path)
    unit_figure_dir = os.path.join(FIGURE_DIR, f"unit_{unit_number}")
    figures = extract_figures(pdf_path, unit_figure_dir)

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
    setup_logging(log_name_prefix="multipass_ingest")

    # Set the PDF you want to process
    test_pdf = "data/chapters/units/Unit APP1 - Appendix A - Derivatives.pdf"
    pdf_name = Path(test_pdf).stem  # e.g., "Unit APP1 - Appendix A - Derivatives"

    # Prepare output paths
    output_json = JSON_DIR / f"{pdf_name.replace(' ', '_').lower()}.json"
    output_md_dir = MARKDOWN_DIR
    output_fig_dir = FIGURE_DIR / pdf_name.replace(' ', '_').lower()

    # Ensure figure subdir exists
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    # Run processing pipeline
    process_pdf(
        pdf_path=test_pdf,
        output_json=str(output_json),
        output_markdown_dir=str(output_md_dir),
        output_figure_dir=str(output_fig_dir)
    )