import argparse
from physbot.batch_ingest import batch_process_chapters
from physbot.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="CLI Driver for Batch Chapter Ingestion")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip chapters already processed.")
    parser.add_argument("--units", nargs="*", help="Specific unit numbers to process (e.g. 101 102 105).")
    return parser.parse_args()

def main():
    setup_logging(log_name_prefix="physbot_batch")
    args = parse_args()
    batch_process_chapters(
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        unit_filter=args.units
    )

if __name__ == "__main__":
    main()
