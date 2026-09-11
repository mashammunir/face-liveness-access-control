"""
Small shared utilities used across scripts and src modules.
"""

import re
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def list_images(directory: Path):
    """Yield all image file paths under a directory (non-recursive by default caller choice)."""
    directory = Path(directory)
    if not directory.exists():
        return
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def guess_subject_id(filename: str) -> Optional[str]:
    """
    Best-effort extraction of a subject/client ID from a filename or path
    component, e.g. 'client001_real_frame_012.jpg' -> 'client001'.

    This is used only as a fallback when the dataset's own folder
    structure doesn't already separate subjects into their own folders.
    Datasets vary widely in naming, so this is intentionally permissive;
    always verify the result against the dataset's own documentation.
    """
    stem = Path(filename).stem
    match = re.match(r"([A-Za-z]*\d+)", stem)
    if match:
        return match.group(1)
    # Fallback: first underscore/dash-delimited token
    return stem.split("_")[0].split("-")[0]
