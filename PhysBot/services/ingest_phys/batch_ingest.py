import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from services.ingest_phys.multipass_ingest import process_pdf
from services.ingest_phys.utils import setup_logging
import re

# === CONSTANTS ===
from services.ingest_phys.constants import CHAPTER_DIR, JSON_DIR, MARKDOWN_DIR, FIGURE_DIR, LOG_DIR

# === LOGGING CONFIGURATION ===
#logging.basicConfig(
#    filename=LOG_FILE,
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s"
#)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(console)

# === CLI ARGUMENTS ===
def parse_args():
    parser = argparse.ArgumentParser(description="Batch process physics chapter PDFs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip chapters already processed.")
    parser.add_argument("--units", nargs="*", help="Specific unit numbers to process (e.g. 101 102 105).")
    return parser.parse_args()

def extract_unit_number(filename):
    """
    Extracts the numeric unit ID from a filename like 'Unit 105 - Something.pdf'
    Returns the string '105' (same format as passed in unit_filter).
    """
    match = re.match(r"Unit\s*(\d+)", filename)
    return match.group(1) if match else None

# === MAIN BATCH FUNCTION ===
def batch_process_chapters(overwrite=False, skip_existing=False, unit_filter=None):
    chapter_pdfs = sorted([
        f for f in os.listdir(CHAPTER_DIR)
        if f.endswith(".pdf") and f.startswith("Unit")
        and (not unit_filter or extract_unit_number(f) in unit_filter)
    ])

    # Log the selected units and how many matched
    logging.info(f"Units selected: {unit_filter if unit_filter else 'All'}")
    logging.info(f"Found {len(chapter_pdfs)} matching PDFs")
    
    total = 0
    success = 0
    skipped = 0
    failed = 0

    logging.info(f"🔍 Found {len(chapter_pdfs)} chapter PDFs.")
    for dir_path in [JSON_DIR, MARKDOWN_DIR, FIGURE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    for filename in tqdm(chapter_pdfs, desc="📘 Processing Chapters"):
        try:
            unit_number = filename.split()[1]
            if unit_filter and unit_number not in unit_filter:
                continue

            total += 1
            pdf_path = os.path.join(CHAPTER_DIR, filename)
            output_json_path = os.path.join(JSON_DIR, f"unit_{unit_number}_output.json")

            # Skip or overwrite
            if os.path.exists(output_json_path):
                if skip_existing:
                    logging.info(f"⚠️ Skipping Unit {unit_number} (already processed)")
                    skipped += 1
                    continue
                elif not overwrite:
                    logging.info(f"❌ Skipping Unit {unit_number} (use --overwrite to replace)")
                    skipped += 1
                    continue

            logging.info(f"▶️ Processing {filename}")

            process_pdf(
                pdf_path,
                output_json=output_json_path,
                output_markdown_dir=MARKDOWN_DIR,
                output_figure_dir=os.path.join(FIGURE_DIR, f"unit_{unit_number}")
            )

            logging.info(f"✅ Completed Unit {unit_number}")
            success += 1

        except Exception as e:
            logging.error(f"❌ Failed to process {filename}: {e}")
            failed += 1

    # === Summary ===
    logging.info("\n📊 Batch Summary")
    logging.info(f"   → Total attempted:  {total}")
    logging.info(f"   → Successful:       {success}")
    logging.info(f"   → Skipped:          {skipped}")
    logging.info(f"   → Failed:           {failed}")

    print("\n📊 Batch Summary")
    print(f"   → Total attempted:  {total}")
    print(f"   → Successful:       {success}")
    print(f"   → Skipped:          {skipped}")
    print(f"   → Failed:           {failed}")

# === ENTRY POINT ===
if __name__ == "__main__":
    setup_logging(log_name_prefix="batch_ingest")
    args = parse_args()
    batch_process_chapters(
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        unit_filter=args.units
    )