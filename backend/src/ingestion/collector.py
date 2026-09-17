"""PMC Open Access article collector.

Downloads PMC JATS XML files from the public ``pmc-oa-opendata`` S3 bucket.

The list of PMC IDs is obtained from NCBI E-utilities through the local
edirect binaries (``~/edirect/esearch`` + ``~/edirect/efetch``), then files
are downloaded concurrently via ``boto3`` with an unsigned (public) S3
client. Downloads are resumable by file existence.

Usage:
    python -m src.ingestion.collector --topic cardiology \
        --output-dir data/raw/cardiology --workers 20
    python -m src.ingestion.collector --topic cardiology \
        --output-dir data/raw/cardiology --min-year 2015 --max-year 2024
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


class Collector(Protocol):
    """Anything that discovers and downloads articles for a topic."""

    topic: str
    output_dir: Path
    max_workers: int

    def collect(self) -> None: ...


class PMCCollector(Collector):

    def __init__(
        self,
        topic: str,
        output_dir: str,
        max_workers: int = 20,
        min_year: int = 0,
        max_year: int = 0,
    ):
        self.topic = topic
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.min_year = min_year
        self.max_year = max_year
        self.bucket = "pmc-oa-opendata"

        self.s3 = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                retries={
                    "max_attempts": 10,
                    "mode": "standard",
                },
            ),
            region_name="us-east-1",
        )

    def collect(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        edirect = Path.home() / "edirect"
        esearch = edirect / "esearch"
        efetch = edirect / "efetch"

        query = (
            f"{self.topic} AND "
            "(open_access[filter] OR author_manuscript[filter])"
        )

        if self.min_year or self.max_year:
            lo = str(self.min_year or 1900)
            hi = str(self.max_year or 9999)
            # publication date range: E-utilities [pdat] interval syntax
            query = (
                f"{query} AND ({lo}/01/01[pdat] : {hi}/12/31[pdat])"
            )

        print(f"Searching PMC for: {self.topic}", flush=True)

        try:
            print("Running ESearch...", flush=True)

            result = subprocess.run(
                [
                    str(esearch),
                    "-db",
                    "pmc",
                    "-query",
                    query,
                ],
                stdout=subprocess.PIPE,
                stderr=None,  # show stderr directly in terminal
                text=True,
                check=True,
            )

            print("ESearch complete.", flush=True)
            print("Running EFetch...", flush=True)

            fetch_result = subprocess.run(
                [
                    str(efetch),
                    "-db",
                    "pmc",
                    "-format",
                    "uid",
                ],
                input=result.stdout,
                stdout=subprocess.PIPE,
                stderr=None,  # show stderr directly in terminal
                text=True,
                check=True,
            )

            print("EFetch complete.", flush=True)

            ids = {
                f"PMC{uid.strip()}"
                for uid in fetch_result.stdout.splitlines()
                if uid.strip()
            }

        except Exception as exc:
            print(f"PMC search failed: {exc}", flush=True)
            return

        print(f"Found {len(ids):,} unique PMC IDs.", flush=True)

        pending = [
            pmc_id
            for pmc_id in ids
            if not (self.output_dir / f"{pmc_id}.xml").exists()
        ]

        print(
            f"Already downloaded: {len(ids) - len(pending):,}",
            flush=True,
        )
        print(
            f"Pending: {len(pending):,}",
            flush=True,
        )

        def download(pmc_id):
            try:
                key = f"{pmc_id}.1/{pmc_id}.1.xml"

                self.s3.download_file(
                    self.bucket,
                    key,
                    str(self.output_dir / f"{pmc_id}.xml"),
                )

                return True

            except Exception as exc:
                print(
                    f"FAILED {pmc_id}: {exc}",
                    flush=True,
                )
                return False

        print(
            f"Starting {self.max_workers} download workers...",
            flush=True,
        )

        with ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:

            results = executor.map(download, pending)

            downloaded = sum(
                tqdm(
                    results,
                    total=len(pending),
                    desc="Downloading",
                    unit="article",
                )
            )

        print(f"Downloaded: {downloaded:,}", flush=True)


# ======================================================================
# CLI (repo entry point: python -m src.ingestion.collector)
# ======================================================================

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download PMC OA articles for a topic.")
    parser.add_argument("--topic", required=True,
                        help="Topic passed to the PMC query.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where downloaded XML files are stored.")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of concurrent download workers.")
    parser.add_argument("--min-year", type=int, default=0,
                        help="First publication year to include (0 = any).")
    parser.add_argument("--max-year", type=int, default=0,
                        help="Last publication year to include (0 = any).")
    args = parser.parse_args(argv)

    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 1
    if args.min_year < 0 or args.max_year < 0:
        print("years must be >= 0", file=sys.stderr)
        return 1
    if args.min_year and args.max_year and args.min_year > args.max_year:
        print("--min-year must be <= --max-year", file=sys.stderr)
        return 1

    PMCCollector(
        topic=args.topic,
        output_dir=args.output_dir,
        max_workers=args.workers,
        min_year=args.min_year,
        max_year=args.max_year,
    ).collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())