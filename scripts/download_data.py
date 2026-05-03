"""Download the DETER-B datacubes from Hugging Face.

Usage
-----
    python -m scripts.download_data --out data/

This pulls every file from `HF_REPO_ID` into the output directory and
verifies that the six files the rest of the pipeline expects are
present:

    deter_area.npz
    deter_count.npz
    deter_cummul.npz
    deter_seasonal.npz
    5km_biomemask.tif
    5km_biomemask_padded.tif
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# >>> EDIT THIS LINE if you publish the dataset under a different repo. <<<
HF_REPO_ID = "dudushi/SoftRank4Deforestation-DETERB"

REQUIRED_FILES = (
    "deter_area.npz",
    "deter_count.npz",
    "deter_cummul.npz",
    "deter_seasonal.npz",
    "5km_biomemask.tif",
    "5km_biomemask_padded.tif",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"),
                        help="output directory (default: ./data)")
    parser.add_argument("--repo-id", default=HF_REPO_ID,
                        help=f"HF dataset repo (default: {HF_REPO_ID})")
    parser.add_argument("--revision", default="main",
                        help="HF revision (branch, tag, or commit)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: `huggingface_hub` is not installed.\n"
              "       run: pip install huggingface_hub", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"→ Downloading {args.repo_id}@{args.revision} into {args.out} …")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.out),
        local_dir_use_symlinks=False,
    )

    missing = [f for f in REQUIRED_FILES if not (args.out / f).is_file()]
    if missing:
        print(f"\nwarning: the following expected files are missing:\n  - " +
              "\n  - ".join(missing), file=sys.stderr)
        return 2

    print("\n✓ Done. Six required files are present.")
    print(f"  Pass `--data-root {args.out}` to the training scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
