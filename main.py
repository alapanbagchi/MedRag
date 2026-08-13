import os
import subprocess
from pathlib import Path

from plugins.collectors.pmc import PMCCollector

def main():
    cardiology = PMCCollector('"Cardiac Arrest" AND 2025[dp]', 'data/raw/heart-attack-25',1_000)
    cardiology.collect()


if __name__ == "__main__":
    main()