import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from core.protocols import Collector


class PMCCollector(Collector):

    def __init__(
        self,
        topic: str,
        output_dir: str,
        max_workers: int = 20,
    ):
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

    def collect(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

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