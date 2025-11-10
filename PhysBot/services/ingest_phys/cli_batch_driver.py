import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ingest_phys.batch_ingest import batch_process_chapters
from services.ingest_phys.utils import setup_logging
from services.ingest_phys.constants import ensure_directories

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="CLI Driver for Batch Chapter Ingestion")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip chapters already processed.")
    parser.add_argument("--units", nargs="*", help="Specific unit numbers to process (e.g. 101 102 105).")
    return parser.parse_args()

def main():
    ensure_directories()
    setup_logging(log_dir="logs", log_name_prefix="physbot_batch")
    args = parse_args()
    batch_process_chapters(
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        unit_filter=args.units
    )

if __name__ == "__main__":
    main()