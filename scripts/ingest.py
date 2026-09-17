#!/usr/bin/env python
"""CLI: ingest a single CSV/PDF/Markdown/TXT source into the vector store.

Usage:
    python scripts/ingest.py --source data/sample/sample_tickets.csv
    python scripts/ingest.py --source docs/runbook.pdf
    python scripts/ingest.py --source docs/troubleshooting.md
    python scripts/ingest.py --source docs/faq.txt
    python scripts/ingest.py --source data/mesos_tickets.csv --csv-mapping config/mesos_mapping.json --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.mapping import CSVFieldMapping  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to a CSV/PDF/Markdown/TXT file")
    parser.add_argument(
        "--csv-mapping",
        help="Optional path to a JSON file overriding the default CSV column mapping (CSV sources only)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing chunks for this source_file before ingesting (use after changing chunking/mapping)",
    )
    args = parser.parse_args()

    csv_mapping = CSVFieldMapping.from_json(args.csv_mapping) if args.csv_mapping else None
    pipeline = IngestionPipeline(csv_mapping=csv_mapping)
    summary = pipeline.ingest(args.source, reset_source=args.reset)

    print(
        f"Loaded {summary.documents_loaded} document(s), "
        f"wrote {summary.chunks_written} chunk(s) from {summary.source}"
    )


if __name__ == "__main__":
    main()
