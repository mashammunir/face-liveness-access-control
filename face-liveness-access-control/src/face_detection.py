"""
Face detection module.

Uses OpenCV's built-in Haar Cascade classifier for face detection.
This is intentionally simple and dependency-light (no extra downloads,
no GPU needed) which makes it reliable for a hackathon demo — the
cascade XML ships inside the opencv-python package itself.

Public API:
    detect_faces(frame) -> list[FaceBox]
    crop_face(frame, box) -> np.ndarray | None
    draw_faces(frame, boxes) -> np.ndarray
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import config


@dataclass
class FaceBox:
    """A detected face's bounding box in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceDetector:
    """Thin wrapper around OpenCV's Haar Cascade face detector."""

    def __init__(self, min_face_size: int = None):
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_file)
        if self._cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from '{cascade_file}'. "
                "Your OpenCV installation may be corrupted — try "
                "reinstalling opencv-python."
            )
        self.min_face_size = min_face_size or config.MIN_FACE_SIZE

    def detect_faces(self, frame: np.ndarray) -> List[FaceBox]:
        """
        Detect faces in a BGR frame (as returned by cv2.VideoCapture).

        Returns a list of FaceBox, sorted largest-first (the largest face
        is almost always the one closest to the camera / intended subject).
        Returns an empty list if no face is found or the frame is invalid.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection under uneven lighting

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        boxes = [FaceBox(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in detections]
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes


def crop_face(frame: np.ndarray, box: FaceBox, margin: float = 0.2) -> Optional[np.ndarray]:
    """
    Crop a face region from the frame with a small margin around the box,
    clamped to the frame boundaries. Returns None if the crop is invalid.
    """
    if frame is None or box is None:
        return None

    h_frame, w_frame = frame.shape[:2]
    mx = int(box.w * margin)
    my = int(box.h * margin)

    x1 = max(0, box.x - mx)
    y1 = max(0, box.y - my)
    x2 = min(w_frame, box.x + box.w + mx)
    y2 = min(h_frame, box.y + box.h + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def draw_faces(frame: np.ndarray, boxes: List[FaceBox], label: str = "Detected") -> np.ndarray:
    """Draw bounding boxes + label on a copy of the frame for display purposes."""
    annotated = frame.copy()
    for box in boxes:
        cv2.rectangle(annotated, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 200, 0), 2)
        cv2.putText(
            annotated, label, (box.x, max(0, box.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
        )
    return annotated
