from __future__ import annotations

"""
splatpack CLI

What this tool does:
1) Find .dng files recursively under a dataset directory.
2) Group those .dng files by their parent directory.
   Each parent directory is treated as a "sequence" (common with CinemaDNG).
3) Export every DNG to a JPEG using darktable-cli.
4) Write a job bundle folder containing:
   - images/ (JPEG exports)
   - manifest.txt (list of exported outputs)
   - job.json (metadata about the run)
5) Optionally zip the job folder to make transport easy (for Drive sync or other handoff).

This script is intentionally "glue code":
- darktable does the image processing
- Python handles file discovery, safety checks, naming, and packaging

Design goals:
- Safe by default
- Easy to understand and modify
- Works from any directory via global command
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


# -----------------------------
# Constants and defaults
# -----------------------------

# We only look for .dng files right now. CinemaDNG is typically a folder full of .dng frames.
DNG_EXTS = {".dng"}

# Output folder created inside the dataset folder.
DEFAULT_OUT_DIRNAME = "_splatpack"

# Subfolder inside the job bundle where exported images are placed.
DEFAULT_IMAGES_DIRNAME = "images"


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class ExportResult:
    """
    A small results container you could return from a higher level function
    if you later refactor main() into a library-style API.
    """
    inputs_found: int
    outputs_written: int
    outputs_dir: Path
    job_dir: Path
    zip_path: Optional[Path]


# -----------------------------
# Small helpers
# -----------------------------

def eprint(*args: object) -> None:
    """Print to stderr so progress logs do not mix with normal output."""
    print(*args, file=sys.stderr)


def which_or_die(tool: str) -> str:
    """
    Find a tool on PATH or exit with a helpful message.
    We use this for darktable-cli.
    """
    found = shutil.which(tool)
    if not found:
        raise SystemExit(
            f"Missing required tool: {tool}\n"
            f"Install darktable (for darktable-cli) and ensure it is on PATH.\n"
            f"macOS: brew install darktable"
        )
    return found


def is_dangerous_directory(path: Path) -> bool:
    """
    Safety guard to prevent accidental recursion in huge or system directories.

    We refuse to run in locations where a recursive search would likely be:
    - enormous (slow and surprising)
    - risky (you might export from a folder that contains unrelated files)
    - destructive if you later add cleanup features

    This is intentionally conservative.
    """
    p = path.resolve()

    # Common "do not do this" roots
    dangerous = {
        Path("/"),
        Path("/System"),
        Path("/Library"),
        Path("/Applications"),
        Path("/Users"),
    }
    if p in dangerous:
        return True

    # Avoid running at the home directory root by default.
    # You can still run on a subfolder inside home, which is the normal use case.
    try:
        if p == Path.home().resolve():
            return True
    except Exception:
        # If anything weird happens resolving home, do not block.
        pass

    return False


# -----------------------------
# File discovery and grouping
# -----------------------------

def iter_dngs(root: Path) -> List[Path]:
    """
    Recursively find all DNG files under the given root directory.

    Returns:
      A sorted list of Paths.

    Notes:
      This does a full recursive walk. That is why we use is_dangerous_directory()
      to avoid accidentally running in giant folders.
    """
    dngs: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in DNG_EXTS:
            dngs.append(p)

    # Sorting provides stability:
    # - makes exports deterministic
    # - makes output file naming consistent across runs
    dngs.sort()
    return dngs


def group_by_parent_dir(dngs: List[Path]) -> List[Tuple[Path, List[Path]]]:
    """
    Group DNGs by their parent folder.

    Why group by parent?
      CinemaDNG sequences are often stored like:
        take001/000001.dng
        take001/000002.dng
        take002/000001.dng
      Grouping by parent lets us preserve "sequence identity" in output names.

    Returns:
      A list of (parent_directory, files_in_that_directory), sorted by directory path.
    """
    groups: dict[Path, List[Path]] = {}
    for p in dngs:
        groups.setdefault(p.parent, []).append(p)

    # Sort groups by the directory path string for deterministic ordering.
    grouped = sorted(groups.items(), key=lambda kv: str(kv[0]))

    # Within each group, sort files as well (rglob already sorted, but safe anyway).
    for parent, files in grouped:
        files.sort()

    return grouped


# -----------------------------
# Output folders and naming
# -----------------------------

def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.

    We do not automatically delete existing data. If you rerun, you will get a new
    timestamped job folder anyway, so collisions are unlikely.
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    """
    Convert a string into something safe-ish for filenames.
    Keeps alphanumerics, underscore, dot, and space. Everything else becomes underscore.

    This is used for:
      - job names
      - sequence folder names embedded into export filenames
    """
    text = (text or "").strip()
    if not text:
        return "dataset"

    out: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", ".", " "):
            out.append(ch)
        else:
            out.append("_")

    # Basic cleanup
    safe = "".join(out).strip()
    while "  " in safe:
        safe = safe.replace("  ", " ")
    return safe


def default_job_name(dataset_dir: Path) -> str:
    """Default job name derived from the dataset folder name."""
    return safe_name(dataset_dir.name)


# -----------------------------
# darktable-cli export
# -----------------------------

def run_darktable_export(
    darktable_cli: str,
    input_file: Path,
    output_file: Path,
    style: Optional[str],
    jpg_quality: int,
    max_long_edge: Optional[int],
) -> None:
    """
    Export one DNG to one JPG using darktable-cli.

    darktable-cli takes an input file and output file:
      darktable-cli input.dng output.jpg

    It can also apply:
      --style "Style Name"

    And it can accept configuration values via:
      --conf key=value

    About resizing:
      darktable can resize via modules or export settings, but the exact configuration keys
      vary by version. The most reliable way to resize is to bake a resize module into a style.

      Here, we attempt a best effort resize by setting both width and height to max_long_edge.
      The intent is "max dimension", not "force exact size". Depending on your version,
      you may need to move resizing into the style.
    """
    cmd = [darktable_cli, str(input_file), str(output_file)]

    if style:
        cmd += ["--style", style]

    # Export settings. Quality key is widely used.
    conf_items: List[str] = []
    conf_items.append(f"plugins/imageio/format/jpeg/quality={jpg_quality}")

    if max_long_edge:
        # Best effort. Some builds ignore these keys or use different ones.
        # If you notice no resizing, the fix is to add a resize module in your style.
        conf_items.append(f"plugins/imageio/format/jpeg/width={max_long_edge}")
        conf_items.append(f"plugins/imageio/format/jpeg/height={max_long_edge}")

    for item in conf_items:
        cmd += ["--conf", item]

    # We silence stdout to keep the terminal readable.
    # If darktable errors, we capture stderr and show it to you.
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -----------------------------
# Job metadata and packaging
# -----------------------------

def write_job_json(job_dir: Path, payload: dict) -> None:
    """Write job.json metadata file."""
    p = job_dir / "job.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(job_dir: Path, image_paths: List[Path]) -> None:
    """
    Write a manifest file listing exported images.

    This is helpful for:
    - debugging
    - later automation steps (the PC can read exactly what images exist)
    """
    p = job_dir / "manifest.txt"
    p.write_text("\n".join(str(x) for x in image_paths) + "\n", encoding="utf-8")


def zip_job(job_dir: Path, zip_path: Path) -> Path:
    """
    Create a zip archive of job_dir.

    shutil.make_archive wants a base name without suffix.
    We accept a path that may or may not end in .zip and normalize it.
    """
    if zip_path.suffix.lower() != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    if zip_path.exists():
        zip_path.unlink()

    base_name = str(zip_path.with_suffix(""))
    archive = shutil.make_archive(base_name, "zip", root_dir=str(job_dir))
    return Path(archive)


# -----------------------------
# CLI entry point
# -----------------------------

def main() -> None:
    """
    Parse CLI args and execute the pipeline.

    The main steps are:
      - validate environment
      - discover inputs
      - create job directory
      - export with darktable-cli
      - write manifest and metadata
      - optionally zip
    """
    parser = argparse.ArgumentParser(
        prog="splatpack",
        description="Export JPGs from DNG sequences using darktable-cli and bundle them for downstream alignment.",
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Dataset folder to process. Default is current directory.",
    )

    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIRNAME,
        help="Output folder name created inside the dataset folder. Default: _splatpack",
    )

    parser.add_argument(
        "--style",
        default=None,
        help="Optional darktable style name to apply during export.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100. Default: 95",
    )

    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=None,
        help="Optional max long edge in pixels. Best practice is to bake resize into a darktable style.",
    )

    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip of the job folder for transport.",
    )

    parser.add_argument(
        "--job-name",
        default=None,
        help="Override job name. Default is derived from dataset folder name.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without exporting or writing files.",
    )

    args = parser.parse_args()

    # Resolve dataset directory
    dataset_dir = Path(args.path).expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {dataset_dir}")

    # Safety check
    if is_dangerous_directory(dataset_dir):
        raise SystemExit(
            f"Refusing to run in a high risk directory: {dataset_dir}\n"
            f"Copy your dataset into a project folder and run splatpack there."
        )

    # Validate external dependency
    darktable_cli = which_or_die("darktable-cli")

    # Find inputs
    dngs = iter_dngs(dataset_dir)
    if not dngs:
        raise SystemExit(f"No DNG files found under: {dataset_dir}")

    groups = group_by_parent_dir(dngs)

    # Create a unique job folder name using a timestamp.
    # This avoids collisions and makes it easy to keep multiple runs.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or default_job_name(dataset_dir)

    # Example output:
    #   dataset/_splatpack/myjob_20260113_165501/images/...
    job_dir = dataset_dir / args.out / f"{job_name}_{timestamp}"
    images_dir = job_dir / DEFAULT_IMAGES_DIRNAME

    eprint(f"Dataset: {dataset_dir}")
    eprint(f"DNG files found: {len(dngs)}")
    eprint(f"Sequence folders found: {len(groups)}")
    eprint(f"Job dir: {job_dir}")
    eprint(f"Images dir: {images_dir}")
    eprint(f"JPEG quality: {args.quality}")
    if args.style:
        eprint(f"Darktable style: {args.style}")
    if args.max_long_edge:
        eprint(f"Max long edge: {args.max_long_edge}px (best effort)")
    if args.zip:
        eprint("ZIP output: enabled")

    if args.dry_run:
        eprint("Dry run only, no files will be written.")
        return

    # Create output folders
    ensure_dir(images_dir)

    # Export loop
    exported: List[Path] = []
    export_index = 0

    # We include the source sequence folder name in the output filename,
    # so multiple sequences can coexist in one job.
    #
    # Output naming format:
    #   <sequence_name>_<global_index>.jpg
    # Example:
    #   take001_000001.jpg
    #   take001_000002.jpg
    #   take002_000003.jpg
    for parent_dir, files in groups:
        seq_name = safe_name(parent_dir.name)

        for f in files:
            export_index += 1
            out_name = f"{seq_name}_{export_index:06d}.jpg"
            out_path = images_dir / out_name

            try:
                run_darktable_export(
                    darktable_cli=darktable_cli,
                    input_file=f,
                    output_file=out_path,
                    style=args.style,
                    jpg_quality=args.quality,
                    max_long_edge=args.max_long_edge,
                )
            except subprocess.CalledProcessError as ex:
                # If darktable fails, print stderr so you can see why.
                stderr = ex.stderr.decode("utf-8", errors="replace") if ex.stderr else ""
                raise SystemExit(f"darktable-cli failed on: {f}\n{stderr}")

            exported.append(out_path)

            # Lightweight progress indicator
            if export_index % 50 == 0:
                eprint(f"Exported {export_index} images...")

    # Write metadata files
    payload = {
        "tool": "splatpack",
        "version": "0.1.0",
        "dataset_dir": str(dataset_dir),
        "job_dir": str(job_dir),
        "images_dir": str(images_dir),
        "inputs_found": len(dngs),
        "outputs_written": len(exported),
        "darktable_style": args.style,
        "jpg_quality": args.quality,
        "max_long_edge": args.max_long_edge,
        "created_at": timestamp,
        "notes": "If you need reliable resizing, add it to your darktable style.",
    }
    write_job_json(job_dir, payload)
    write_manifest(job_dir, exported)

    # Optional zip bundle
    zip_path: Optional[Path] = None
    if args.zip:
        # Put zip next to job folder inside the out directory
        zip_path = dataset_dir / args.out / f"{job_name}_{timestamp}.zip"
        zip_path = zip_job(job_dir, zip_path)
        eprint(f"Zipped: {zip_path}")

    eprint("")
    eprint("Done.")
    eprint(f"Exported JPGs: {len(exported)}")
    eprint(f"Job folder: {job_dir}")
    if zip_path:
        eprint(f"Job zip: {zip_path}")


if __name__ == "__main__":
    main()
