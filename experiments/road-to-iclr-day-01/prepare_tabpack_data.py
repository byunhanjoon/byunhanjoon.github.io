"""Download the public, preprocessed datasets released with TabPack.

By default, all nine real datasets in the release are extracted. The archive
remains in the user cache and the datasets remain untracked by Git.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URL = (
    "https://huggingface.co/datasets/Yura52/tabpack-data/resolve/main/"
    "tabpack-data.tar.gz"
)
DEFAULT_DATASETS = (
    "adult",
    "black-friday",
    "california",
    "churn",
    "diamond",
    "higgs-small",
    "house",
    "microsoft",
    "otto",
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def extract(archive: Path, output: Path, datasets: set[str]) -> None:
    output = output.resolve()
    wanted_prefixes = tuple(f"data/{name}/" for name in datasets)
    with tarfile.open(archive, "r:gz") as handle:
        members = []
        for member in handle.getmembers():
            if not member.name.startswith(wanted_prefixes):
                continue
            destination = (output / member.name).resolve()
            if output not in destination.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
            if Path(member.name).name.startswith("._"):
                continue
            members.append(member)
        # Paths and link types are checked above. Avoid ``filter=`` here so the
        # helper also works on Python versions before 3.12.
        handle.extractall(output, members=members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="Directory that will contain one subdirectory per dataset.",
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / ".cache" / "day1-tabpack" / "tabpack-data.tar.gz",
    )
    args = parser.parse_args()

    if not args.archive.exists():
        download(ARCHIVE_URL, args.archive)

    staging = args.output.parent / f".{args.output.name}-extract"
    staging.mkdir(parents=True, exist_ok=True)
    extract(args.archive, staging, set(args.datasets))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        source = staging / "data" / name
        destination = args.output / name
        if destination.exists():
            print(f"Keeping existing {destination}")
            continue
        shutil.move(str(source), destination)
        print(f"Prepared {destination}")
    shutil.rmtree(staging)


if __name__ == "__main__":
    main()
