"""
src/liveness.py

Defines the liveness detection CNN and the real-time inference wrapper
used by the Streamlit app. Two lightweight backbones are supported,
both fine-tuned from torchvision's ImageNet-pretrained weights so the
model converges quickly even on a small anti-spoofing dataset:

    - mobilenet_v3_small (default): smallest/fastest, good for real-time
      webcam inference on a laptop CPU.
    - resnet18: slightly heavier, sometimes a bit more accurate.

Output: 2-class classification — LIVE (1) vs SPOOF (0).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import config

CLASS_NAMES = ["spoof", "live"]  # index 0 = spoof, index 1 = live


def build_model(architecture: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    """
    Build a 2-class liveness classifier from a torchvision backbone.

    Using a pretrained ImageNet backbone (rather than training from
    scratch) lets a small anti-spoofing dataset still produce a
    reasonably accurate model — this is standard transfer learning and
    is appropriate for a hackathon timeline.
    """
    if architecture == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
    elif architecture == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}. "
                          "Use 'mobilenet_v3_small' or 'resnet18'.")
    return model


def get_eval_transform(input_size: int = None) -> transforms.Compose:
    """Preprocessing used at inference time (and for val/test sets)."""
    size = input_size or config.LIVENESS_INPUT_SIZE
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transform(input_size: int = None) -> transforms.Compose:
    """Preprocessing + light augmentation used only during training."""
    size = input_size or config.LIVENESS_INPUT_SIZE
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class LivenessDetector:
    """
    Loads a trained liveness model and runs inference on a single face
    crop (BGR numpy array, as produced by src.face_detection.crop_face).

    If no trained model file exists at config.MODEL_PATH, this class
    enters DEMO MODE: it still runs (so the rest of the app doesn't
    crash), but every prediction is clearly labeled as a demo/fallback
    result rather than a real AI prediction, per project requirements.
    """

    def __init__(self, model_path: Optional[Path] = None, architecture: str = "mobilenet_v3_small"):
        self.model_path = Path(model_path) if model_path else config.MODEL_PATH
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_eval_transform()
        self.demo_mode = True
        self.model = None

        if self.model_path.exists():
            try:
                self.model = build_model(architecture, pretrained=False)
                state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                self.demo_mode = False
            except Exception as e:
                # Corrupt/incompatible model file: fall back to demo mode
                # rather than crashing the whole application.
                print(f"[LivenessDetector] Failed to load model at {self.model_path}: {e}. "
                      f"Falling back to DEMO MODE.")
                self.model = None
                self.demo_mode = True

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, confidence) where label is 'LIVE' or 'SPOOF' and
        confidence is a float in [0, 1].

        In DEMO MODE (no trained model available), this returns a fixed
        neutral placeholder rather than a random or fabricated result —
        callers MUST check `.demo_mode` and display a DEMO MODE indicator
        in the UI; this method never pretends a demo result is a real
        prediction.

        If face_bgr is None, empty, or not a valid 3-channel image, this
        raises ValueError rather than crashing with an opaque exception
        from deep inside the transform/model pipeline — callers (e.g.
        the Streamlit UI) should validate their crop before calling, but
        this method does not trust that they did.
        """
        if face_bgr is None:
            raise ValueError("predict() received face_bgr=None; a valid face crop is required.")
        if not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
            raise ValueError("predict() received an empty or invalid face crop.")
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError(
                f"predict() expects a 3-channel BGR image (H, W, 3); got shape {face_bgr.shape}."
            )

        if self.demo_mode or self.model is None:
            # Explicit, non-random placeholder — never presented as a real
            # AI result. UI is responsible for showing "DEMO MODE".
            return "LIVE", 0.0

        rgb = face_bgr[:, :, ::-1]  # BGR -> RGB
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            live_prob = probs[CLASS_NAMES.index("live")].item()

        label = "LIVE" if live_prob >= config.LIVENESS_THRESHOLD else "SPOOF"
        confidence = live_prob if label == "LIVE" else (1.0 - live_prob)
        return label, confidence
