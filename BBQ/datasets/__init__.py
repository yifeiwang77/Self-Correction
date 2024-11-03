from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).parent
BBQ_DATASET_DIR = CURRENT_DIR / "BBQ" / "data"
BBQ_SUPPLEMENTAL_DIR = CURRENT_DIR / "BBQ" / "supplemental"
LAW_DATASET_DIR = CURRENT_DIR
WINOGENDER_DATASET_DIR = CURRENT_DIR / "winogender-schemas" / "data"


def find_bbq_dataset(search_dir: Path = BBQ_DATASET_DIR) -> Iterable[Path]:
    return search_dir.glob("*.jsonl")


def find_bbq_metadata(search_dir: Path = BBQ_SUPPLEMENTAL_DIR) -> Path:
    return search_dir / "additional_metadata.csv"


def find_law_dataset(search_dir: Path = LAW_DATASET_DIR) -> Iterable[Path]:
    return [search_dir / "law_data.csv"]


def find_winogender_dataset(
    search_dir: Path = WINOGENDER_DATASET_DIR,
) -> Iterable[Path]:
    return [search_dir / "all_sentences.tsv"]


def find_winogender_stats(search_dir: Path = WINOGENDER_DATASET_DIR) -> Path:
    return search_dir / "occupations-stats.tsv"
