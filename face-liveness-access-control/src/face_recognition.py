"""
src/face_recognition.py

Face embedding extraction and comparison for the registration/
recognition pipeline:

    Detected face -> face embedding -> compare against registered
    embeddings -> similarity/distance -> Authorized/Unknown

WHY A CUSTOM EMBEDDING MODEL INSTEAD OF THE `face_recognition` LIBRARY
------------------------------------------------------------------------
The popular `face_recognition` PyPI package depends on `dlib`, which
must be compiled from source on most systems (needs CMake and a C++
toolchain — on Windows this typically means Visual Studio build
tools). That's a common source of hackathon setup failures. Per the
project's own "prefer a simpler alternative if a dependency is hard to
install on Windows" rule, this module instead reuses the
ImageNet-pretrained ResNet-18 we already depend on (torchvision) with
its final classification layer removed, turning it into a general
512-dimensional feature extractor. This is a legitimate, standard
transfer-learning technique for face embeddings on a hackathon
timeline: it won't beat a model trained specifically for face
verification, but it requires zero extra installs beyond what Stage 3
already added, and produces embeddings that are consistent and
comparable for the same person across frames/lighting.

PUBLIC API
----------
    EmbeddingExtractor().embed(face_bgr) -> np.ndarray shape (512,)
    cosine_similarity(a, b) -> float in [-1, 1]
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import config


def _build_embedding_backbone(pretrained: bool = True) -> nn.Module:
    """ResNet-18 with the final FC layer removed -> 512-d feature vector."""
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Identity()  # strip classification head -> raw 512-d features
    model.eval()
    return model


class EmbeddingExtractor:
    """
    Extracts a 512-dimensional embedding vector from a face crop.

    Loaded once and reused (constructing it downloads/initializes the
    ResNet-18 backbone, which is comparatively expensive to do per call).
    """

    EMBEDDING_SIZE = 512

    def __init__(self, pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = _build_embedding_backbone(pretrained=pretrained).to(self.device)
            self.available = True
        except Exception as e:
            # E.g. no internet access to fetch pretrained weights. The rest
            # of the app must handle `available=False` gracefully rather
            # than crash — registration/recognition will be unavailable
            # but liveness detection and face detection still work.
            print(f"[EmbeddingExtractor] Failed to initialize backbone: {e}. "
                  f"Face recognition will be unavailable.")
            self.model = None
            self.available = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a unit-normalized 512-d embedding for a BGR face crop,
        or None if the extractor is unavailable or the input is invalid
        (None, empty, wrong number of dimensions, or not 3-channel).
        """
        if not self.available or face_bgr is None:
            return None
        if not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
            return None
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            return None

        rgb = face_bgr[:, :, ::-1]  # BGR -> RGB
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)[0]  # shape (512,)

        vec = features.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors, in [-1, 1].
    Assumes vectors may or may not already be unit-normalized.
    """
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def match_embedding(query_embedding: np.ndarray, known_embeddings: dict, threshold: float = None):
    """
    Compare a query embedding against a dict of {name: embedding} and
    return (best_match_name_or_None, best_similarity).

    Returns (None, best_similarity) if the best match is below the
    similarity threshold (config.FACE_RECOGNITION_THRESHOLD by default) —
    i.e. an "Unknown" result — even if some similarity score exists.
    """
    threshold = threshold if threshold is not None else config.FACE_RECOGNITION_THRESHOLD

    if query_embedding is None or not known_embeddings:
        return None, 0.0

    best_name = None
    best_score = -1.0
    for name, emb in known_embeddings.items():
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score
