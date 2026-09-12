"""
AI Access Control System — Streamlit dashboard.

STAGE 9: UI polish. Same pipeline as Stage 8 (detect -> liveness ->
recognition-if-LIVE -> decide_access -> log), restyled to match the
spec's dashboard mockup: a bordered status card next to the camera
feed, larger metric displays for liveness/identity confidence, and a
large, impossible-to-miss centered GRANTED/DENIED banner. No decision
logic changed in this stage — only layout and presentation.
"""

import cv2
import numpy as np
import streamlit as st

import config
from src.face_detection import FaceDetector, crop_face, draw_faces
from src.liveness import LivenessDetector
from src.face_recognition import EmbeddingExtractor, match_embedding
from src.database import UserDatabase, AccessLogDatabase, DatabaseError
from src.access_control import decide_access, GRANTED

st.set_page_config(page_title="AI Access Control System", layout="wide")

st.title("🔐 AI Access Control System")
st.caption("Liveness Detection → Face Recognition → Access Control → Logging")


@st.cache_resource
def get_face_detector():
    return FaceDetector()


@st.cache_resource
def get_liveness_detector():
    return LivenessDetector()


@st.cache_resource
def get_embedding_extractor():
    return EmbeddingExtractor()


@st.cache_resource
def get_user_db():
    return UserDatabase()


@st.cache_resource
def get_access_log_db():
    return AccessLogDatabase()


detector = get_face_detector()
liveness_detector = get_liveness_detector()
embedding_extractor = get_embedding_extractor()

try:
    user_db = get_user_db()
    access_log_db = get_access_log_db()
except DatabaseError as e:
    st.error(
        f"⚠️ Could not initialize the database: {e}\n\n"
        f"Check that `{config.DATABASE_PATH.parent}` is writable, and "
        "that no other process has an exclusive lock on the database "
        "files, then reload this page."
    )
    st.stop()

tab_dashboard, tab_register, tab_logs = st.tabs(["📹 Dashboard", "➕ Register User", "📋 Access Logs"])

# --------------------------------------------------------------------------
# DASHBOARD TAB — live webcam feed with liveness detection + recognition
# --------------------------------------------------------------------------
with tab_dashboard:
    if liveness_detector.demo_mode:
        st.warning(
            "⚠️ **DEMO MODE** — no trained liveness model found at "
            f"`{config.MODEL_PATH}`. Liveness results below are placeholders, "
            "not real AI predictions. Run `scripts/train_liveness.py` on a "
            "real dataset to enable actual liveness detection."
        )

    if not embedding_extractor.available:
        st.info(
            "ℹ️ Face recognition is unavailable (embedding model failed to "
            "initialize). Liveness detection will still run, but identity "
            "will always show as unavailable."
        )

    if st.session_state.get("db_error"):
        st.error(
            "⚠️ Database error — some recognition or logging may be "
            f"incomplete: {st.session_state['db_error']}"
        )

    st.markdown("Take a photo to run it through the access control pipeline.")
    dashboard_photo = st.camera_input("Take a photo to verify", key="dashboard_photo")

    cam_col, status_col = st.columns([3, 2], gap="large")

    if dashboard_photo is None:
        with status_col:
            card = st.container(border=True)
            with card:
                st.markdown("#### System Status")
                st.metric("Face", "—")
                st.metric("Liveness", "—")
                st.metric("Identity", "—")
                st.info("Waiting for a photo...")
    else:
        file_bytes = np.frombuffer(dashboard_photo.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not read the captured photo. Please try again.")
        else:
            boxes = detector.detect_faces(frame)

            with cam_col:
                if boxes:
                    annotated = draw_faces(frame, boxes, label="Detected")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            with status_col:
                card = st.container(border=True)
                with card:
                    st.markdown("#### System Status")

                    if not boxes:
                        st.metric("Face", "Not detected")
                        st.metric("Liveness", "—")
                        st.metric("Identity", "—")
                        st.info("No face in frame.")
                    else:
                        st.metric("Face", f"Detected ({len(boxes)})")

                        primary_box = boxes[0]
                        face_crop = crop_face(frame, primary_box)

                        label, confidence = None, None
                        if face_crop is not None and face_crop.size > 0:
                            try:
                                label, confidence = liveness_detector.predict(face_crop)
                            except ValueError as e:
                                st.session_state["db_error"] = f"Liveness check skipped: {e}"

                        recognized_name, match_score = None, 0.0
                        if label == "LIVE" and embedding_extractor.available and face_crop is not None:
                            query_embedding = embedding_extractor.embed(face_crop)
                            try:
                                known_embeddings = user_db.get_all_active_embeddings()
                                recognized_name, match_score = match_embedding(query_embedding, known_embeddings)
                            except DatabaseError as e:
                                st.session_state["db_error"] = str(e)

                        decision = decide_access(
                            liveness_label=label,
                            liveness_confidence=confidence or 0.0,
                            recognized_name=recognized_name,
                            recognition_score=match_score,
                        )
                        try:
                            access_log_db.log_access(
                                user_name=decision.user_name,
                                liveness_result=decision.liveness_result,
                                liveness_confidence=decision.liveness_confidence,
                                recognition_result=decision.recognition_result,
                                recognition_confidence=decision.recognition_confidence,
                                access_result=decision.access_result,
                            )
                        except DatabaseError as e:
                            st.session_state["db_error"] = str(e)

                        if label == "LIVE":
                            st.metric("Liveness", "LIVE", delta="passed", delta_color="normal")
                            st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence * 100:.1f}%")
                            if not embedding_extractor.available:
                                st.metric("Identity", "Unavailable")
                            elif recognized_name:
                                st.metric("Identity", recognized_name, delta="match", delta_color="normal")
                                st.progress(min(1.0, max(0.0, match_score)), text=f"Match: {match_score * 100:.1f}%")
                            else:
                                st.metric("Identity", "Unknown", delta="no match", delta_color="inverse")
                        elif label == "SPOOF":
                            st.metric("Liveness", "SPOOF", delta="failed", delta_color="inverse")
                            st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence * 100:.1f}%")
                            st.metric("Identity", "--")
                            st.caption("Skipped — liveness check failed")
                        else:
                            st.metric("Liveness", "Inconclusive")
                            st.metric("Identity", "—")

                        if decision.access_result == GRANTED:
                            st.success(f"## ✅ ACCESS GRANTED\n**Welcome, {decision.user_name}**")
                        else:
                            st.error("## ⛔ ACCESS DENIED")
# --------------------------------------------------------------------------
# REGISTER USER TAB — capture a face snapshot, extract embedding, store it
# --------------------------------------------------------------------------
with tab_register:
    st.markdown(
        """
        Register a new user so the system can recognize them later.
        Take a clear, well-lit photo of your face using the camera
        below, enter a name, then click **Register**.
        """
    )

    if not embedding_extractor.available:
        st.error(
            "Face recognition is currently unavailable (the embedding "
            "model failed to initialize — check your internet connection "
            "for downloading pretrained weights, or see the terminal logs "
            "for details). Registration cannot proceed until this is resolved."
        )
    else:
        name_input = st.text_input("Name / ID", key="register_name")
        photo = st.camera_input("Capture your face", key="register_photo")

        if photo is not None:
            file_bytes = np.frombuffer(photo.getvalue(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                st.error(
                    "Could not read the captured photo (invalid or corrupted "
                    "image data). Please try capturing the photo again."
                )
                boxes = []
            else:
                boxes = detector.detect_faces(frame_bgr)

            if frame_bgr is not None and not boxes:
                st.error("No face detected in the captured photo. Please try again with better lighting.")
            elif len(boxes) > 1:
                st.warning(
                    f"{len(boxes)} faces detected in the photo. Using the largest "
                    "one — for best results, make sure only one person is in frame."
                )

            if boxes:
                primary_box = boxes[0]
                face_crop = crop_face(frame_bgr, primary_box)
                annotated = draw_faces(frame_bgr, boxes[:1], label="Face")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected face", width=300)

                if st.button("Register", key="register_button"):
                    if not name_input or not name_input.strip():
                        st.error("Please enter a name before registering.")
                    elif face_crop is None or face_crop.size == 0:
                        st.error("Could not crop the detected face. Please try again.")
                    else:
                        embedding = embedding_extractor.embed(face_crop)
                        if embedding is None:
                            st.error("Failed to generate a face embedding. Please try again.")
                        else:
                            try:
                                user_id = user_db.add_user(name_input.strip(), embedding)
                                st.success(f"✅ Registered **{name_input.strip()}** (user id {user_id}).")
                            except DatabaseError as e:
                                st.error(f"Could not save this registration due to a database error: {e}")

    st.divider()
    st.subheader("Registered Users")
    try:
        users = user_db.list_users()
    except DatabaseError as e:
        users = []
        st.error(f"Could not load registered users due to a database error: {e}")
    if users:
        st.table(
            [{"ID": u["id"], "Name": u["name"], "Status": u["status"], "Registered": u["created_at"][:19]}
             for u in users]
        )
    else:
        st.caption("No users registered yet.")

# --------------------------------------------------------------------------
# ACCESS LOGS TAB — view recent access attempts
# --------------------------------------------------------------------------
with tab_logs:
    st.markdown("Every access attempt from the Dashboard tab is recorded here, granted or denied.")

    log_limit = st.slider("Number of recent entries to show", min_value=5, max_value=200, value=50, step=5)
    try:
        entries = access_log_db.get_recent(limit=log_limit)
    except DatabaseError as e:
        entries = []
        st.error(f"Could not load the access log due to a database error: {e}")

    if not entries:
        st.caption("No access attempts logged yet. Use the Dashboard tab with the camera running to generate log entries.")
    else:
        display_rows = []
        for e in entries:
            display_rows.append({
                "Timestamp": e["timestamp"][:19],
                "User": e["user_name"] or "--",
                "Liveness": e["liveness_result"],
                "Liveness Conf.": f"{e['liveness_confidence'] * 100:.1f}%" if e["liveness_confidence"] is not None else "--",
                "Recognition": e["recognition_result"],
                "Recognition Conf.": f"{e['recognition_confidence'] * 100:.1f}%" if e["recognition_confidence"] is not None else "--",
                "Access": e["access_result"],
            })
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        granted_count = sum(1 for e in entries if e["access_result"] == "GRANTED")
        denied_count = len(entries) - granted_count
        col1, col2 = st.columns(2)
        col1.metric("Granted", granted_count)
        col2.metric("Denied", denied_count)
