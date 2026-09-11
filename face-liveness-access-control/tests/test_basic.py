"""
Basic sanity tests for the face-liveness-access-control project.

Run with:
    pytest tests/test_basic.py -v

These tests avoid requiring a physical webcam — they use synthetic
frames and (where available) a downloaded sample image so they can run
in CI or any machine without a camera attached.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.face_detection import FaceDetector, crop_face, draw_faces, FaceBox


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


def test_detector_loads_cascade(detector):
    """The Haar Cascade should load without error."""
    assert detector is not None
    assert not detector._cascade.empty()


def test_blank_frame_returns_no_faces(detector):
    """A blank/black frame should produce zero detections, not an error."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector.detect_faces(blank)
    assert boxes == []


def test_none_frame_does_not_crash(detector):
    """Passing None should return an empty list rather than raising."""
    boxes = detector.detect_faces(None)
    assert boxes == []


def test_crop_face_returns_none_for_invalid_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bad_box = FaceBox(x=1000, y=1000, w=50, h=50)  # entirely outside the frame
    crop = crop_face(frame, bad_box)
    assert crop is None


def test_crop_face_valid_box():
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 255
    box = FaceBox(x=50, y=50, w=50, h=50)
    crop = crop_face(frame, box, margin=0.0)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_draw_faces_returns_same_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FaceBox(x=10, y=10, w=30, h=30)
    annotated = draw_faces(frame, [box])
    assert annotated.shape == frame.shape


# --- Stage 2: dataset preprocessing tests ---

from scripts.preprocess_dataset import split_subjects


def test_split_subjects_no_overlap():
    """Core anti-leakage guarantee: every subject lands in exactly one split."""
    ids = [f"subj{i:03d}" for i in range(20)]
    train, val, test = split_subjects(ids, 0.7, 0.15, 0.15, seed=1)

    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(ids)


def test_split_subjects_deterministic():
    """Same seed should produce the same split, for reproducibility."""
    ids = [f"s{i}" for i in range(15)]
    split_a = split_subjects(ids, 0.7, 0.15, 0.15, seed=42)
    split_b = split_subjects(ids, 0.7, 0.15, 0.15, seed=42)
    assert split_a == split_b


def test_split_subjects_small_n_all_splits_nonempty():
    """With very few subjects, val/test should still get at least one each
    (rather than silently ending up empty)."""
    ids = [f"s{i}" for i in range(3)]
    train, val, test = split_subjects(ids, 0.7, 0.15, 0.15, seed=0)
    assert len(train) >= 1
    assert len(val) >= 1
    assert len(test) >= 1


# --- Stage 3: liveness model tests ---

from src.liveness import build_model, LivenessDetector, get_eval_transform, get_train_transform


def test_build_model_mobilenet_output_shape():
    """Model should output 2 logits (spoof, live) for a batch of 1."""
    model = build_model("mobilenet_v3_small", pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, config.LIVENESS_INPUT_SIZE, config.LIVENESS_INPUT_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 2)


def test_build_model_resnet18_output_shape():
    model = build_model("resnet18", pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, config.LIVENESS_INPUT_SIZE, config.LIVENESS_INPUT_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 2)


def test_build_model_invalid_architecture_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_architecture", pretrained=False)


def test_liveness_detector_demo_mode_when_no_model_file():
    """No model file at the given path -> must enter DEMO MODE, not crash."""
    det = LivenessDetector(model_path="/tmp/definitely_missing_model_xyz.pth")
    assert det.demo_mode is True
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    label, conf = det.predict(frame)
    assert label in ("LIVE", "SPOOF")
    assert 0.0 <= conf <= 1.0


def test_liveness_detector_corrupt_model_falls_back_to_demo_mode(tmp_path):
    """A corrupt/unreadable model file must not crash the app — it should
    fall back to DEMO MODE."""
    bad_model = tmp_path / "corrupt.pth"
    bad_model.write_bytes(b"not a valid checkpoint")
    det = LivenessDetector(model_path=bad_model)
    assert det.demo_mode is True


# --- Stage 4: evaluation metrics tests ---

from scripts.evaluate_liveness import compute_anti_spoofing_metrics


def test_anti_spoofing_metrics_perfect_classifier():
    """All predictions correct -> APCER=0, BPCER=0, ACER=0."""
    y_true = [0, 0, 0, 1, 1, 1]  # 0=spoof, 1=live
    y_pred = [0, 0, 0, 1, 1, 1]
    m = compute_anti_spoofing_metrics(y_true, y_pred, live_label=1, spoof_label=0)
    assert m["APCER"] == 0.0
    assert m["BPCER"] == 0.0
    assert m["ACER"] == 0.0
    assert m["n_spoof_samples"] == 3
    assert m["n_live_samples"] == 3


def test_anti_spoofing_metrics_all_spoof_classified_as_live():
    """Worst case for security: every spoof attack fools the system."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [1, 1, 1, 1, 1, 1]  # all predicted live
    m = compute_anti_spoofing_metrics(y_true, y_pred, live_label=1, spoof_label=0)
    assert m["APCER"] == 1.0  # every spoof got through
    assert m["BPCER"] == 0.0  # every real live user correctly accepted
    assert m["ACER"] == 0.5


def test_anti_spoofing_metrics_all_live_rejected():
    """Every genuine user wrongly rejected (BPCER=1), no spoofs get through."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0]  # all predicted spoof
    m = compute_anti_spoofing_metrics(y_true, y_pred, live_label=1, spoof_label=0)
    assert m["APCER"] == 0.0
    assert m["BPCER"] == 1.0
    assert m["ACER"] == 0.5


def test_anti_spoofing_metrics_known_mixed_case():
    """Hand-computed mixed case to verify the formula end-to-end."""
    # 4 spoof samples: 1 misclassified as live (FP_spoof=1) -> APCER=1/4=0.25
    # 4 live samples: 1 misclassified as spoof (FN_live=1) -> BPCER=1/4=0.25
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    m = compute_anti_spoofing_metrics(y_true, y_pred, live_label=1, spoof_label=0)
    assert m["APCER"] == 0.25
    assert m["BPCER"] == 0.25
    assert m["ACER"] == 0.25
    assert m["spoof_classified_as_live"] == 1
    assert m["live_classified_as_spoof"] == 1


# --- Stage 6: face recognition + database tests ---

import tempfile as _tempfile

from src.face_recognition import EmbeddingExtractor, cosine_similarity, match_embedding
from src.database import UserDatabase, AccessLogDatabase


@pytest.fixture(scope="module")
def embedding_extractor():
    # pretrained=False avoids requiring internet access to download
    # ImageNet weights in the test environment; the embedding logic
    # being tested (shape, normalization, comparison) is identical
    # regardless of which backbone weights are loaded.
    return EmbeddingExtractor(pretrained=False)


def test_embedding_shape_and_normalization(embedding_extractor):
    face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    emb = embedding_extractor.embed(face)
    assert emb is not None
    assert emb.shape == (EmbeddingExtractor.EMBEDDING_SIZE,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4


def test_embedding_deterministic(embedding_extractor):
    face = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    emb1 = embedding_extractor.embed(face)
    emb2 = embedding_extractor.embed(face)
    assert np.allclose(emb1, emb2)


def test_embedding_none_input_returns_none(embedding_extractor):
    assert embedding_extractor.embed(None) is None


def test_cosine_similarity_self_is_one():
    v = np.random.randn(512).astype(np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-4


def test_cosine_similarity_none_safe():
    v = np.random.randn(512).astype(np.float32)
    assert cosine_similarity(None, v) == -1.0
    assert cosine_similarity(v, None) == -1.0


def test_match_embedding_exact_match():
    rng = np.random.RandomState(1)
    alice = rng.randn(512).astype(np.float32)
    alice /= np.linalg.norm(alice)
    known = {"Alice": alice}
    name, score = match_embedding(alice, known, threshold=0.6)
    assert name == "Alice"
    assert score > 0.99


def test_match_embedding_unrelated_is_unknown():
    rng = np.random.RandomState(2)
    alice = rng.randn(512).astype(np.float32)
    alice /= np.linalg.norm(alice)
    unrelated = rng.randn(512).astype(np.float32)
    unrelated /= np.linalg.norm(unrelated)
    known = {"Alice": alice}
    name, score = match_embedding(unrelated, known, threshold=0.6)
    assert name is None  # Unknown


def test_match_embedding_empty_registry():
    v = np.random.randn(512).astype(np.float32)
    name, score = match_embedding(v, {}, threshold=0.6)
    assert name is None
    assert score == 0.0


def test_user_database_add_and_retrieve(tmp_path):
    db = UserDatabase(db_path=tmp_path / "users.db")
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    user_id = db.add_user("Alice", emb)
    assert user_id == 1

    embeddings = db.get_all_active_embeddings()
    assert "Alice" in embeddings
    assert np.allclose(embeddings["Alice"], emb, atol=1e-5)


def test_user_database_empty_name_raises(tmp_path):
    db = UserDatabase(db_path=tmp_path / "users.db")
    emb = np.random.randn(512).astype(np.float32)
    with pytest.raises(ValueError):
        db.add_user("", emb)


def test_user_database_deactivate_excludes_from_active(tmp_path):
    db = UserDatabase(db_path=tmp_path / "users.db")
    emb = np.random.randn(512).astype(np.float32)
    user_id = db.add_user("Bob", emb)
    db.deactivate_user(user_id)
    embeddings = db.get_all_active_embeddings()
    assert "Bob" not in embeddings


def test_access_log_records_and_orders_recent_first(tmp_path):
    log_db = AccessLogDatabase(db_path=tmp_path / "log.db")
    log_db.log_access("Ali", "LIVE", 0.97, "MATCH", 0.94, "GRANTED")
    log_db.log_access(None, "SPOOF", 0.91, "N/A", 0.0, "DENIED")

    recent = log_db.get_recent(limit=10)
    assert len(recent) == 2
    assert recent[0]["access_result"] == "DENIED"  # most recent first
    assert recent[1]["user_name"] == "Ali"


# --- Stage 7: recognition-gated-by-liveness tests (security-critical) ---

def _simulate_dashboard_recognition_step(label, embedding_extractor_available,
                                          face_crop_present, query_embedding, known_embeddings):
    """
    Mirrors the exact gating logic in app.py's Dashboard tab: recognition
    is attempted ONLY when liveness == LIVE. This function exists purely
    so the security-critical gate can be unit tested without booting
    Streamlit.
    """
    recognized_name, match_score = None, 0.0
    if label == "LIVE" and embedding_extractor_available and face_crop_present:
        recognized_name, match_score = match_embedding(query_embedding, known_embeddings)
    return recognized_name, match_score


def test_recognition_never_runs_on_spoof_label():
    """Security-critical: a SPOOF liveness result must never produce a
    recognized identity, regardless of how similar the face embedding is
    to a registered user."""
    rng = np.random.RandomState(7)
    registered = rng.randn(512).astype(np.float32)
    registered /= np.linalg.norm(registered)
    known = {"Alice": registered}

    # Even with an exact-match embedding available, SPOOF must short-circuit
    name, score = _simulate_dashboard_recognition_step(
        label="SPOOF",
        embedding_extractor_available=True,
        face_crop_present=True,
        query_embedding=registered,  # would be a perfect match if attempted
        known_embeddings=known,
    )
    assert name is None
    assert score == 0.0


def test_recognition_runs_on_live_label_with_match():
    rng = np.random.RandomState(8)
    registered = rng.randn(512).astype(np.float32)
    registered /= np.linalg.norm(registered)
    known = {"Bob": registered}

    name, score = _simulate_dashboard_recognition_step(
        label="LIVE",
        embedding_extractor_available=True,
        face_crop_present=True,
        query_embedding=registered,
        known_embeddings=known,
    )
    assert name == "Bob"
    assert score > 0.99


def test_recognition_runs_on_live_label_unregistered_face_is_unknown():
    rng = np.random.RandomState(9)
    registered = rng.randn(512).astype(np.float32)
    registered /= np.linalg.norm(registered)
    unrelated = rng.randn(512).astype(np.float32)
    unrelated /= np.linalg.norm(unrelated)
    known = {"Carol": registered}

    name, score = _simulate_dashboard_recognition_step(
        label="LIVE",
        embedding_extractor_available=True,
        face_crop_present=True,
        query_embedding=unrelated,
        known_embeddings=known,
    )
    assert name is None  # Unknown


def test_recognition_skipped_when_extractor_unavailable():
    name, score = _simulate_dashboard_recognition_step(
        label="LIVE",
        embedding_extractor_available=False,
        face_crop_present=True,
        query_embedding=None,
        known_embeddings={},
    )
    assert name is None
    assert score == 0.0


def test_box_label_display_logic():
    """Mirrors app.py's box-label computation for the annotated video frame."""
    def compute_box_label(label, recognized_name):
        box_label = label if label else "Detected"
        if label == "LIVE":
            box_label = recognized_name if recognized_name else "LIVE - Unknown"
        return box_label

    assert compute_box_label("LIVE", "Alice") == "Alice"
    assert compute_box_label("LIVE", None) == "LIVE - Unknown"
    assert compute_box_label("SPOOF", None) == "SPOOF"
    assert compute_box_label(None, None) == "Detected"


# --- Stage 8: access control decision logic tests (security-critical) ---

from src.access_control import decide_access, GRANTED, DENIED, AccessDecision


def test_access_granted_when_live_and_recognized():
    """Spec Test 1: real person, recognized -> ACCESS GRANTED."""
    d = decide_access("LIVE", 0.974, "Ahmed", 0.942)
    assert d.access_result == GRANTED
    assert d.user_name == "Ahmed"
    assert d.liveness_result == "LIVE"
    assert d.recognition_result == "MATCH"


def test_access_denied_when_spoof():
    """Spec Test 2/3: printed photo or phone replay -> SPOOF -> ACCESS DENIED."""
    d = decide_access("SPOOF", 0.941, None, 0.0)
    assert d.access_result == DENIED
    assert d.user_name is None
    assert d.recognition_result == "N/A"


def test_access_denied_spoof_even_if_a_name_was_passed_in():
    """Defense-in-depth: even if a caller mistakenly supplies a
    recognized_name alongside a SPOOF liveness label, decide_access must
    NOT grant access or attribute the attempt to that name. Liveness gates
    everything, structurally, inside this function — not just by
    convention in the caller."""
    d = decide_access("SPOOF", 0.85, "Ahmed", 0.99)
    assert d.access_result == DENIED
    assert d.user_name is None
    assert d.recognition_result == "N/A"
    assert d.recognition_confidence == 0.0


def test_access_denied_when_live_but_unrecognized():
    """Spec Test 4: unknown live person -> LIVE + UNKNOWN -> ACCESS DENIED."""
    d = decide_access("LIVE", 0.88, None, 0.3)
    assert d.access_result == DENIED
    assert d.liveness_result == "LIVE"
    assert d.recognition_result == "UNKNOWN"
    assert d.user_name is None


def test_access_denied_when_liveness_could_not_be_evaluated():
    """Fail closed: if liveness is None (e.g. no valid face crop that
    frame), access must be denied rather than assuming anything."""
    d = decide_access(None, 0.0, None, 0.0)
    assert d.access_result == DENIED
    assert d.user_name is None


def test_access_decision_is_dataclass_with_expected_fields():
    d = decide_access("LIVE", 0.9, "Bob", 0.8)
    assert isinstance(d, AccessDecision)
    assert hasattr(d, "liveness_result")
    assert hasattr(d, "access_result")


# --- Stage 8: access-control decision tests (spec's demo scenarios) ---
# (builds on the tests above with the spec's named Test 1-4 attack
# scenarios, plus full log round-trips; decide_access/GRANTED/DENIED
# are already imported above)


def test_access_empty_string_user_treated_as_unrecognized():
    d = decide_access("LIVE", 0.9, "", 0.9)
    assert d.access_result == DENIED


def test_access_log_full_round_trip_granted(tmp_path):
    """End-to-end: a GRANTED decision writes correctly to the access log
    and comes back out with matching fields."""
    log_db = AccessLogDatabase(db_path=tmp_path / "log.db")
    decision = decide_access("LIVE", 0.97, "Ahmed", 0.94)
    log_db.log_access(
        user_name=decision.user_name,
        liveness_result=decision.liveness_result,
        liveness_confidence=decision.liveness_confidence,
        recognition_result=decision.recognition_result,
        recognition_confidence=decision.recognition_confidence,
        access_result=decision.access_result,
    )
    entry = log_db.get_recent(1)[0]
    assert entry["access_result"] == "GRANTED"
    assert entry["user_name"] == "Ahmed"
    assert entry["liveness_result"] == "LIVE"


def test_access_log_full_round_trip_denied_spoof(tmp_path):
    log_db = AccessLogDatabase(db_path=tmp_path / "log.db")
    decision = decide_access("SPOOF", 0.94, None, 0.0)
    log_db.log_access(
        user_name=decision.user_name,
        liveness_result=decision.liveness_result,
        liveness_confidence=decision.liveness_confidence,
        recognition_result=decision.recognition_result,
        recognition_confidence=decision.recognition_confidence,
        access_result=decision.access_result,
    )
    entry = log_db.get_recent(1)[0]
    assert entry["access_result"] == "DENIED"
    assert entry["user_name"] is None
    assert entry["liveness_result"] == "SPOOF"


# --- Stage 11: error-handling audit regression tests ---

from src.database import DatabaseError


def test_corrupt_database_file_raises_database_error(tmp_path):
    """A file that exists but isn't a valid SQLite database must raise
    DatabaseError, not an unhandled sqlite3.Error."""
    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"not a valid sqlite database file at all")

    with pytest.raises(DatabaseError):
        db = UserDatabase(db_path=corrupt_path)
        emb = np.random.randn(512).astype(np.float32)
        db.add_user("Test", emb)


def test_corrupt_stored_embedding_raises_database_error(tmp_path):
    """A row with unparseable embedding JSON (e.g. from manual tampering
    or partial write) must raise DatabaseError rather than crash with a
    raw JSONDecodeError or silently return garbage."""
    import sqlite3

    db_path = tmp_path / "test.db"
    db = UserDatabase(db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO users (name, embedding, created_at, status) VALUES (?, ?, ?, 'active')",
        ("CorruptUser", "not valid json {{{", "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseError):
        db.get_all_active_embeddings()


def test_database_directory_creation_failure_raises_database_error(tmp_path):
    """If the database's parent directory can't be created (e.g. a path
    component is actually a file), this must raise DatabaseError."""
    blocker_file = tmp_path / "blocker"
    blocker_file.write_text("this is a file, not a directory")
    bad_path = blocker_file / "subdir" / "test.db"

    with pytest.raises(DatabaseError):
        UserDatabase(db_path=bad_path)


def test_deactivate_and_delete_nonexistent_user_do_not_crash(tmp_path):
    """Operating on a user_id that doesn't exist is a no-op, not an error."""
    db = UserDatabase(db_path=tmp_path / "test.db")
    db.deactivate_user(9999)  # should not raise
    db.delete_user(9999)  # should not raise


def test_liveness_predict_rejects_none():
    det = LivenessDetector(model_path="/tmp/nonexistent_for_validation_test.pth")
    with pytest.raises(ValueError):
        det.predict(None)


def test_liveness_predict_rejects_2d_grayscale():
    det = LivenessDetector(model_path="/tmp/nonexistent_for_validation_test.pth")
    with pytest.raises(ValueError):
        det.predict(np.zeros((100, 100), dtype=np.uint8))


def test_liveness_predict_rejects_wrong_channel_count():
    det = LivenessDetector(model_path="/tmp/nonexistent_for_validation_test.pth")
    with pytest.raises(ValueError):
        det.predict(np.zeros((100, 100, 4), dtype=np.uint8))  # RGBA, not BGR


def test_liveness_predict_rejects_empty_array():
    det = LivenessDetector(model_path="/tmp/nonexistent_for_validation_test.pth")
    with pytest.raises(ValueError):
        det.predict(np.array([]))


def test_face_detector_rejects_frame_smaller_than_min_size():
    """A frame smaller than min_face_size can never contain a detectable
    face at that minimum size — must return an empty list, not error."""
    detector = FaceDetector(min_face_size=80)
    tiny_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    boxes = detector.detect_faces(tiny_frame)
    assert boxes == []


def test_registration_handles_undecodable_image_bytes():
    """Mirrors app.py's registration flow: cv2.imdecode on garbage bytes
    returns None, and detect_faces(None) must handle that gracefully
    rather than crash — this is the exact call sequence app.py makes."""
    import cv2

    garbage = np.frombuffer(b"not a real image, just garbage bytes here", dtype=np.uint8)
    frame_bgr = cv2.imdecode(garbage, cv2.IMREAD_COLOR)
    assert frame_bgr is None  # confirms the precondition this test is checking

    detector = FaceDetector()
    boxes = detector.detect_faces(frame_bgr)
    assert boxes == []
