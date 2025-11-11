import os
import fitz  # PyMuPDF
from PyPDF2 import PdfMerger
import re
from openai import OpenAI
import json
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import logging
from pathlib import Path
import pdfplumber
import nltk
from datetime import datetime
from services.ingest_phys.constants import CHAPTER_DIR, JSON_DIR, MARKDOWN_DIR, FIGURE_DIR, LOG_DIR as DATASET_LOG_DIR

load_dotenv(find_dotenv())

def setup_logging(log_dir=None, log_name_prefix="physbot"):
    """
    Sets up logging for the entire pipeline.
    Creates a timestamped log file under the specified log_dir.
    """
    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Use datasets/physbot/metadata/logs by default
    if log_dir is None:
        log_dir = DATASET_LOG_DIR   # <— NEW (replaces project_root/"logs")
    else:
        log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name_prefix}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging initialized")
    logging.info(f"Log file: {log_file}")
    print(f"📄 Logging to: {log_file}")



nltk.download("punkt")
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Load environment variables
# Load .env from the project root, regardless of working directory
project_root = Path(__file__).resolve().parents[1]  # Adjust as needed
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

# Retrieve OpenAI API key
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    return OpenAI(api_key=api_key)



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

    return output_pdf  # Now returning the output file path


def clean_equation(equation):
    """
    Cleans extracted equations by:
    - Applying predefined replacements for common physics symbols.
    - Standardizing LaTeX formatting.
    - Converting vectors to bold symbols.
    - Handling fractions, exponents, matrices, and scientific notation.
    - Logging unrecognized formats for debugging.
    - Ensuring proper LaTeX formatting.
    """

    if equation is None:
        return None  # Return None if no equation is provided

    # **1. Handle dictionary-based equations (multiple forms like "momentum" and "kinetic energy")**
    if isinstance(equation, dict):
        return {key: clean_equation(value) for key, value in equation.items()}

    # **2. Handle list-based multiple equations (convert to LaTeX sequence)**
    if isinstance(equation, list):
        return " \\quad ".join([clean_equation(eq) for eq in equation])

    # **3. Ensure equation is a string before proceeding**
    if not isinstance(equation, str):
        logging.warning(f"Unexpected equation format (not str/dict/list): {equation}")
        return str(equation)  # Convert to string to prevent crashes

    equation = equation.strip()  # Ensure clean strings

    # **4. Dictionary of predefined replacements for LaTeX formatting**
    replacements = {
        "𝐅⃗": "\\mathbf{F}", "𝒗⃗": "\\mathbf{v}", "𝒂⃗": "\\mathbf{a}", "𝒑⃗": "\\mathbf{p}",
        "𝑭⃗𝒈": "\\mathbf{F}_g", "𝒎𝒈⃗": "mg", "𝒅𝒑⃗ 𝒅𝒕": "\\frac{d\\mathbf{p}}{dt}",
        "𝑑𝑝⃗/𝑑𝑡": "\\frac{d\\mathbf{p}}{dt}", "𝑚𝑎⃗": "m\\mathbf{a}", "𝜸−𝟏": "\\gamma - 1",
        "𝒗𝟐/𝒄𝟐": "\\frac{v^2}{c^2}", "𝑭𝑛𝑒𝑡": "\\mathbf{F}_{net}", "𝑭⃗𝑛𝑒𝑡": "\\mathbf{F}_{net}",
        "𝑷⃗": "\\mathbf{P}", "𝑲𝑻": "K_T", "𝑲𝒓": "K_R", "𝝉⃗": "\\mathbf{\\tau}", "𝒓⃗": "\\mathbf{r}",
        "𝑳⃗": "\\mathbf{L}", "𝜽": "\\theta", "𝜌": "\\rho", "𝑮": "G", "𝒈": "g", "𝑻": "T",
        "𝑼": "U", "𝒔𝒊𝒏": "\\sin", "𝒄𝒐𝒔": "\\cos", "𝒕𝒂𝒏": "\\tan",
        "𝑭⃗ = −𝒌(𝒙 − 𝒙𝟎)": "\\mathbf{F} = -k(x - x_0)", "𝑭⃗ = 𝒎𝒂⃗": "\\mathbf{F} = m\\mathbf{a}",
        "𝑷⃗ = 𝑭⃗ ⋅ 𝒗⃗": "\\mathbf{P} = \\mathbf{F} \\cdot \\mathbf{v}"
    }

    # **5. Apply replacements first**
    for old, new in replacements.items():
        equation = equation.replace(old, new)

    # **6. Convert generic vector notation (any variable marked with ⃗ to bold)**
    equation = re.sub(r"(\b[a-zA-Z_]+)⃗", r"\\mathbf{\1}", equation)

    # **7. Convert subscripts and superscripts correctly**
    equation = re.sub(r"([a-zA-Z])_([0-9]+)", r"\1_{\2}", equation)  # Subscripts
    equation = re.sub(r"([a-zA-Z])\^\(([^)]+)\)", r"\1^{\2}", equation)  # Parentheses in exponents
    equation = re.sub(r"([a-zA-Z])\^([0-9]+)", r"\1^{\2}", equation)  # Simple superscripts

    # **8. Handle fractions**
    equation = re.sub(r"([\dA-Za-z_+]+)\s*/\s*([\dA-Za-z_+]+)", r"\\frac{\1}{\2}", equation)

    # **9. Fix scientific notation (e.g., 3.5e-5 → 3.5 × 10^{-5})**
    equation = re.sub(r"(\d+(\.\d+)?)\s*[eE×]\s*([-+]?\d+)", r"\1 \\times 10^{\3}", equation)

    # **10. Convert matrices (only if they contain commas/semicolons)**
    equation = re.sub(r"\[(.*?[,;].*?)\]", r"\\begin{bmatrix} \1 \\end{bmatrix}", equation)
    equation = equation.replace(",", " \\\\ ")  # Format rows in matrices

    # **11. Log equations that were not recognized**
    unrecognized_chars = "𝒗𝑭𝒂𝒑𝒎𝑻𝑼𝝉𝒓𝑳𝜽𝜌𝑮𝒈"  # List of potential unrecognized symbols
    if any(char in equation for char in unrecognized_chars):
        logging.warning(f"Potential unrecognized equation format: {equation}")

    # **12. Ensure LaTeX formatting is applied**
    if not equation.startswith("$"):
        equation = f"${equation}$"  # Wrap in LaTeX math mode

    return equation.strip()

def post_clean_equation(equation):
    """Standardizes LaTeX formatting, corrects vector notation, subscripts, superscripts, and fractions."""

    if equation is None or equation.strip() == "" or equation == "$$":
        return None  # Ensure empty equations are replaced with None

    # Handle dictionary-based equations (multiple forms of an equation)
    if isinstance(equation, dict):
        return {key: clean_equation(value) for key, value in equation.items()}

    # Handle list-based multiple equations
    if isinstance(equation, list):
        return " \\quad ".join([clean_equation(eq) for eq in equation if eq.strip() != "$$"])

    # Ensure equation is a string before processing
    if not isinstance(equation, str):
        logging.warning(f"Unexpected equation format (not str/dict/list): {equation}")
        return str(equation)

    equation = equation.strip()

    # Predefined replacements for common physics symbols
    replacements = {
        "𝐅⃗": "\\mathbf{F}", "𝒗⃗": "\\mathbf{v}", "𝒂⃗": "\\mathbf{a}", "𝒑⃗": "\\mathbf{p}",
        "𝑭⃗𝒈": "\\mathbf{F}_g", "𝒎𝒈⃗": "mg", "𝒅𝒑⃗ 𝒅𝒕": "\\frac{d\\mathbf{p}}{dt}",
        "𝑑𝑝⃗/𝑑𝑡": "\\frac{d\\mathbf{p}}{dt}", "𝑚𝑎⃗": "m\\mathbf{a}", "𝜸−𝟏": "\\gamma - 1",
        "𝒗𝟐/𝒄𝟐": "\\frac{v^2}{c^2}", "𝑭𝑛𝑒𝑡": "\\mathbf{F}_{net}", "𝑭⃗𝑛𝑒𝑡": "\\mathbf{F}_{net}",
        "𝑷⃗": "\\mathbf{P}", "𝑲𝑻": "K_T", "𝑲𝒓": "K_R", "𝝉⃗": "\\mathbf{\\tau}", "𝒓⃗": "\\mathbf{r}",
        "𝑳⃗": "\\mathbf{L}", "𝜽": "\\theta", "𝜌": "\\rho", "𝑮": "G", "𝒈": "g", "𝑻": "T",
        "𝑼": "U", "𝒔𝒊𝒏": "\\sin", "𝒄𝒐𝒔": "\\cos", "𝒕𝒂𝒏": "\\tan",
        "𝑭⃗ = −𝒌(𝒙 − 𝒙𝟎)": "\\mathbf{F} = -k(x - x_0)", "𝑭⃗ = 𝒎𝒂⃗": "\\mathbf{F} = m\\mathbf{a}",
        "𝑷⃗ = 𝑭⃗ ⋅ 𝒗⃗": "\\mathbf{P} = \\mathbf{F} \\cdot \\mathbf{v}"
    }

    # Apply replacements
    for old, new in replacements.items():
        equation = equation.replace(old, new)

    # Convert vector notation to LaTeX (e.g., "x⃗" → "\mathbf{x}")
    equation = re.sub(r"([a-zA-Z])⃗", r"\\mathbf{\1}", equation)

    # Convert subscripts and superscripts
    equation = re.sub(r"([a-zA-Z])_([0-9]+)", r"\1_{\2}", equation)  # Subscripts
    equation = re.sub(r"([a-zA-Z])\^([0-9]+)", r"\1^{\2}", equation)  # Superscripts

    # Convert fractions
    equation = re.sub(r"([\dA-Za-z_+]+)\s*/\s*([\dA-Za-z_+]+)", r"\\frac{\1}{\2}", equation)

    # Convert scientific notation (e.g., 3.5e-5 → 3.5 × 10^{-5})
    equation = re.sub(r"(\d+(\.\d+)?)\s*[eE×]\s*([-+]?\d+)", r"\1 \\times 10^{\3}", equation)

    # Ensure LaTeX formatting
    if not equation.startswith("$"):
        equation = f"${equation}$"  # Wrap in math mode

    return equation.strip()

# Function to extract Bare Essentials sections
def extract_bare_essentials_pages(pdf_path):
    """
    Extracts the 'Bare Essentials' text from a compiled PDF of all chapters.
    Stops extraction when the first numbered section (e.g., '105.1') appears.
    """
    doc = fitz.open(pdf_path)
    extracted_sections = []
    current_unit = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")
        extracted_text = []
        in_bare_essentials = False

        for line in lines:
            line = line.strip()

            # Detect Unit Headers
            if re.match(r"^Unit \d+", line):
                current_unit = line.strip()

            # Start extraction when "Bare Essentials" is found
            if "Bare Essentials" in line:
                in_bare_essentials = True
                continue  # Skip the heading itself

            # Stop extraction when first numbered section is detected
            if in_bare_essentials and re.match(r"^\d+\.\d+\s*–", line):
                in_bare_essentials = False
                break

            if in_bare_essentials:
                extracted_text.append(line)

        if extracted_text:
            extracted_sections.append({
                "unit": current_unit,
                "section": "Bare Essentials",
                "text": " ".join(extracted_text)
            })
    
    return extracted_sections

# Function to send extracted text to OpenAI LLM
def send_to_openai_for_extraction(text):
    """
    Sends extracted Bare Essentials text to OpenAI's GPT model for structured extraction.
    """
    prompt = f"""
    Extract key physics concepts from the following text. For each concept, return:
    - The concept name
    - Its definition
    - Any relevant equations
    - Key notes (if present)

    Text: {text}

    Format output as JSON **without markdown formatting or triple backticks**:
    [{{"concept": "", "definition": "", "equation": "", "notes": []}}]
    """

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert physics assistant that extracts structured data."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content.strip()

    # Log raw response for debugging
    logging.info(f"Raw OpenAI response: {response_text}")

    # Fix: Strip ```json and ``` from response
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove trailing ```

    # Validate and parse JSON
    try:
        parsed_json = json.loads(response_text)
        return parsed_json
    except json.JSONDecodeError:
        logging.error("Invalid JSON format in OpenAI response. Response was:")
        logging.error(response_text)
        return []

# Process the Bare Essentials PDF
def process_bare_essentials(pdf_path, output_json_path):
    extracted_sections = extract_bare_essentials_pages(pdf_path)
    structured_concepts = []
    
    for section in tqdm(extracted_sections, desc="Extracting Concepts", unit="section"):
        concepts = send_to_openai_for_extraction(section["text"])
        
        for concept in concepts:
            concept["equation"] = clean_equation(concept["equation"])  # Clean equations

            structured_concepts.append({
                "unit": section["unit"],
                "section": "Bare Essentials",
                **concept  # Merging extracted fields
            })
    
    with open(output_json_path, "w") as f:
        json.dump(structured_concepts, f, indent=4)
    
    print(f"Extracted concepts saved to {output_json_path}")

    # return output_json  # Now returning the output file path

def extract_text_chunks(pdf_path, output_json="chapter_text.json", chunk_size=500, overlap=75):
    """
    Extracts structured text chunks from a chapter PDF with adaptive chunking for optimal LLM context retention.
    
    Parameters:
        pdf_path (str): Path to the chapter PDF.
        output_json (str): Path to save the structured JSON output.
        chunk_size (int): Target token count per chunk.
        overlap (int): Token overlap between adjacent chunks to maintain context continuity.
    """
    logging.basicConfig(
        filename="pdf_processing.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extracted_chunks = []
    current_unit = None
    current_section = None
    buffer = []
    buffer_token_count = 0
    
    section_pattern = re.compile(r"^(\d+)\s*[-–]\s*(.+)")  # Detect "105 – Motion and Kinetic Energy"
    subsection_pattern = re.compile(r"^(\d+\.\d+)\s*[-–]\s*(.+)")  # Detect "105.1 – Coordinate Systems"
    page_header_pattern = re.compile(r"^\d{3}[-–]\d+\s*\|\s*P\s*h\s*y\s*i\s*c\s*s", re.IGNORECASE)
    cid_pattern = re.compile(r"\( cid:\d+ \)")  # Detect ( cid:XXXX ) artifacts
    
    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = pdf.pages[0].extract_text()
        if first_page_text:
            first_line = first_page_text.strip().split("\n")[0]
            match = section_pattern.match(first_line)
            if match:
                unit_num, unit_title = match.groups()
                current_unit = f"Unit {unit_num}"
                current_section = unit_title
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split("\n")
            if len(lines) > 1:
                lines = lines[1:]  # Skip the first line to avoid page headers
            
            for line in lines:
                line = cid_pattern.sub("", line).strip()  # Remove cid artifacts
                
                # Detect and extract Section titles
                section_match = subsection_pattern.match(line)
                if section_match:
                    if buffer:
                        extracted_chunks.append({
                            "unit": current_unit,
                            "section": current_section,
                            "chunk_type": "paragraph",
                            "content": " ".join(buffer),
                            "page": page_num + 1,
                        })
                        buffer, buffer_token_count = [], 0  # Reset buffer for new section
                    current_section = section_match.group(2)
                    continue
                
                # Tokenize content and add to buffer
                tokens = word_tokenize(line)
                token_count = len(tokens)

                if buffer_token_count + token_count > chunk_size:
                    extracted_chunks.append({
                        "unit": current_unit,
                        "section": current_section,
                        "chunk_type": "paragraph",
                        "content": " ".join(buffer),
                        "page": page_num + 1,
                    })
                    
                    # Start new buffer with overlap from previous chunk
                    buffer = buffer[-overlap:] + tokens
                    buffer_token_count = len(buffer)
                else:
                    buffer.extend(tokens)
                    buffer_token_count += token_count
            
            # Store remaining buffer at the end of the document
            if buffer:
                extracted_chunks.append({
                    "unit": current_unit,
                    "section": current_section,
                    "chunk_type": "paragraph",
                    "content": " ".join(buffer),
                    "page": page_num + 1,
                })
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_chunks, f, indent=4)

    print(f"Text extraction complete! Data saved to {output_json}")
    return extracted_chunks
