"""
scripts/preprocess_dataset.py

Converts a raw public face anti-spoofing dataset into the project's
standard structure:

    data/
        train/
            live/
            spoof/
        val/
            live/
            spoof/
        test/
            live/
            spoof/

WHY THIS SCRIPT EXISTS
-----------------------
Public datasets (MSU-MFSD, CASIA-FASD, Replay-Attack, OULU-NPU, SiW, ...)
each ship with their own folder layout and video formats. Rather than
hard-coding one dataset's quirks into the training code, this script
is the single place that:

  1. Extracts face crops from raw videos/images using our own
     face_detection module (so train/inference use the exact same
     face-cropping logic).
  2. Splits data by SUBJECT (person), not by frame or video, so that
     no subject appears in more than one of train/val/test. Frames
     from the same video (or the same person's other videos) are
     highly correlated — splitting by individual frame/video would
     leak subject identity into the test set and inflate accuracy
     numbers in a way that would not generalize to a real deployment.

HOW TO POINT IT AT YOUR DATASET
--------------------------------
Because every public anti-spoofing dataset organizes raw files
differently, this script expects your raw data pre-sorted into two
top-level folders that YOU (or a small one-off script) create from
whatever the source dataset gives you:

    <raw_dir>/
        live/
            <subject_id>_<anything>.jpg   (or .mp4/.avi -> frames extracted)
        spoof/
            <subject_id>_<anything>.jpg

Subject IDs are inferred from the filename prefix (see
src/utils.guess_subject_id). If your dataset's raw files are organized
as one folder per subject instead, see --subject-folders below.

USAGE
-----
    python scripts/preprocess_dataset.py \\
        --raw-dir /path/to/raw_dataset \\
        --out-dir data \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

    # If raw data is organized as raw_dir/live/<subject_id>/*.jpg (one
    # folder per subject) instead of flat files:
    python scripts/preprocess_dataset.py --raw-dir ... --subject-folders

This script does NOT download any dataset. Per the project's privacy
rules, you must obtain the dataset yourself under its own license terms
(e.g. MSU-MFSD, CASIA-FASD) and never scrape faces from the internet.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_detection import FaceDetector, crop_face
from src.utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, guess_subject_id
import config


def extract_frames(video_path: Path, every_n: int = 10):
    """Yield BGR frames from a video, sampling every Nth frame to avoid
    near-duplicate frames dominating the dataset."""
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            yield frame
        idx += 1
    cap.release()


def collect_subject_files(class_dir: Path, subject_folders: bool):
    """
    Returns a dict: {subject_id: [file_path, ...]}

    subject_folders=True  -> class_dir/<subject_id>/*.{jpg,mp4,...}
    subject_folders=False -> class_dir/<subject_id>_*.{jpg,mp4,...} (flat)
    """
    subjects = {}

    if subject_folders:
        for sub_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            files = [
                f for f in sub_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            ]
            if files:
                subjects[sub_dir.name] = files
    else:
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                sid = guess_subject_id(f.name)
                subjects.setdefault(sid, []).append(f)

    return subjects


def split_subjects(subject_ids, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split a list of subject IDs into train/val/test so that every subject
    appears in exactly one split. This is the key anti-leakage step.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    ids = sorted(subject_ids)  # sort first for determinism across platforms
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(n * train_ratio)) if n >= 3 else n
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 0

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    # Guarantee no split ends up empty when we have >=3 subjects
    if n >= 3:
        for split in (val_ids, test_ids):
            if not split and train_ids:
                split.append(train_ids.pop())

    return train_ids, val_ids, test_ids


def process_class(
    class_name: str,
    raw_class_dir: Path,
    out_dir: Path,
    detector: FaceDetector,
    subject_folders: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    frame_stride: int,
    seed: int,
):
    if not raw_class_dir.exists():
        print(f"  [!] Skipping '{class_name}': directory not found: {raw_class_dir}")
        return {"train": 0, "val": 0, "test": 0}

    subjects = collect_subject_files(raw_class_dir, subject_folders)
    if not subjects:
        print(f"  [!] No files found for class '{class_name}' in {raw_class_dir}")
        return {"train": 0, "val": 0, "test": 0}

    train_ids, val_ids, test_ids = split_subjects(
        list(subjects.keys()), train_ratio, val_ratio, test_ratio, seed=seed
    )
    split_of = {}
    split_of.update({sid: "train" for sid in train_ids})
    split_of.update({sid: "val" for sid in val_ids})
    split_of.update({sid: "test" for sid in test_ids})

    print(
        f"  Class '{class_name}': {len(subjects)} subjects "
        f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})"
    )

    counts = {"train": 0, "val": 0, "test": 0}

    for subject_id, files in subjects.items():
        split = split_of[subject_id]
        dest_dir = out_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(file_path))
                if frame is None:
                    continue
                frames = [(0, frame)]
            else:
                frames = list(enumerate(extract_frames(file_path, every_n=frame_stride)))

            for frame_idx, frame in frames:
                boxes = detector.detect_faces(frame)
                if not boxes:
                    continue
                face = crop_face(frame, boxes[0])
                if face is None or face.size == 0:
                    continue
                face = cv2.resize(face, (config.LIVENESS_INPUT_SIZE, config.LIVENESS_INPUT_SIZE))

                out_name = f"{subject_id}_{file_path.stem}_{frame_idx:04d}.jpg"
                out_path = dest_dir / out_name
                cv2.imwrite(str(out_path), face)
                counts[split] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, type=Path,
                         help="Raw dataset directory containing live/ and spoof/ subfolders")
    parser.add_argument("--out-dir", default=Path("data"), type=Path,
                         help="Output directory (default: data/)")
    parser.add_argument("--subject-folders", action="store_true",
                         help="Raw data is organized as live/<subject_id>/*.jpg instead of flat files")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--frame-stride", type=int, default=10,
                         help="Sample every Nth frame from videos (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                         help="Delete existing contents of out-dir/{train,val,test} before writing")
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: raw dataset directory does not exist: {args.raw_dir}")
        sys.exit(1)

    if args.clean:
        for split in ("train", "val", "test"):
            split_dir = args.out_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    detector = FaceDetector()

    print(f"Preprocessing dataset from: {args.raw_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Subject-level split (ratios train/val/test = "
          f"{args.train_ratio}/{args.val_ratio}/{args.test_ratio})")
    print("No frame- or video-level splitting is used — this avoids data leakage "
          "where the same person appears in both train and test.")
    print()

    totals = {"train": 0, "val": 0, "test": 0}
    for class_name in ("live", "spoof"):
        counts = process_class(
            class_name=class_name,
            raw_class_dir=args.raw_dir / class_name,
            out_dir=args.out_dir,
            detector=detector,
            subject_folders=args.subject_folders,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            frame_stride=args.frame_stride,
            seed=args.seed,
        )
        for split in totals:
            totals[split] += counts[split]

    print()
    print("Done. Face crops written:")
    print(f"  train: {totals['train']}")
    print(f"  val:   {totals['val']}")
    print(f"  test:  {totals['test']}")
    if sum(totals.values()) == 0:
        print("\nWARNING: No face crops were written. Check that --raw-dir points to a "
              "directory containing live/ and spoof/ subfolders with images or videos, "
              "and that faces are detectable in them.")


if __name__ == "__main__":
    main()
