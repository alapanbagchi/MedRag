"""PMC Open Access article collector.

Downloads PMC JATS XML files from the public ``pmc-oa-opendata`` S3 bucket.

The list of PMC IDs is obtained from NCBI E-utilities (``esearch`` /
``efetch``), then files are downloaded concurrently via ``boto3`` with an
unsigned (public) S3 client. Downloads are resumable by file existence.

Usage:
    python -m medrag.collector --topic cardiology \\
        --output-dir data/raw/cardiology --workers 20 --limit 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


class PMCCollector:
    """Discover and download PMC OA articles for a topic."""

    def __init__(
        self,
        topic: str,
        output_dir: str | Path,
        max_workers: int = 20,
    ) -> None:
        self.topic = topic
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
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

    def collect(self, limit: int = 0) -> None:
        """Run discovery + download.

        ``limit`` caps the number of *new* files downloaded (0 = unlimited).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ids = self._discover_ids()
        if ids is None:
            return

        pending = [
            pmc_id
            for pmc_id in ids
            if not (self.output_dir / f"{pmc_id}.xml").exists()
        ]

        print(f"Already downloaded: {len(ids) - len(pending):,}", flush=True)
        print(f"Pending: {len(pending):,}", flush=True)

        if limit > 0:
            pending = pending[:limit]
            print(f"Limited to: {len(pending):,}", flush=True)

        if not pending:
            print("Nothing to download.", flush=True)
            return

        print(f"Starting {self.max_workers} download workers...", flush=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._download, pending)
            downloaded = sum(
                tqdm(
                    results,
                    total=len(pending),
                    desc="Downloading",
                    unit="article",
                )
            )

        print(f"Downloaded: {downloaded:,}", flush=True)

    # ==================================================================
    # Discovery
    # ==================================================================

    def _discover_ids(self) -> Optional[Set[str]]:
        edirect = Path.home() / "edirect"
        esearch = edirect / "esearch"
        efetch = edirect / "efetch"

        query = (
            f"{self.topic} AND "
            "(open_access[filter] OR author_manuscript[filter])"
        )

        print(f"Searching PMC for: {self.topic}", flush=True)

        try:
            print("Running ESearch...", flush=True)
            result = subprocess.run(
                [str(esearch), "-db", "pmc", "-query", query],
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                check=True,
            )

            print("ESearch complete.", flush=True)
            print("Running EFetch...", flush=True)

            fetch_result = subprocess.run(
                [str(efetch), "-db", "pmc", "-format", "uid"],
                input=result.stdout,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                check=True,
            )

            print("EFetch complete.", flush=True)

            ids = {
                f"PMC{uid.strip()}"
                for uid in fetch_result.stdout.splitlines()
                if uid.strip()
            }
        except Exception as exc:  # noqa: BLE001 - discovery failure is not fatal
            print(f"PMC search failed: {exc}", flush=True)
            return None

        print(f"Found {len(ids):,} unique PMC IDs.", flush=True)
        return ids

    # ==================================================================
    # Download
    # ==================================================================

    def _download(self, pmc_id: str) -> bool:
        try:
            key = f"{pmc_id}.1/{pmc_id}.1.xml"
            self.s3.download_file(
                self.bucket,
                key,
                str(self.output_dir / f"{pmc_id}.xml"),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - continue past per-file failures
            print(f"FAILED {pmc_id}: {exc}", flush=True)
            return False


# ======================================================================
# CLI
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download PMC OA articles for a topic.")
    parser.add_argument("--topic", required=True, help="Topic passed to the PMC query.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where downloaded XML files are stored.")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of concurrent download workers.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum number of new files to download. 0 = all.")
    args = parser.parse_args(argv)

    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 1
    if args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        return 1

    collector = PMCCollector(
        topic=args.topic,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )
    collector.collect(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
