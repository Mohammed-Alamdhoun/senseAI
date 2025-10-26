# app.py
# ------------------------------------------------------------
# Flask + MediaPipe Pose -> COCO-17 output (xy pixels + normalized)
# - /video: MJPEG stream with skeleton overlay
# - /pose.json: latest pose (COCO-17 order) + scores
# Comments are in English (as requested)
# ------------------------------------------------------------

import os
import time
import json
import threading
import joblib
from datetime import datetime
from typing import Dict, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp


# -----------------------------
# MediaPipe Pose Backend
# -----------------------------

mp_pose = mp.solutions.pose

# Map MediaPipe(33) -> COCO-17 indices
# COCO-17 order:
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
# 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
# 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
# 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
MP_TO_COCO17 = np.array([
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
], dtype=int)

# COCO-17 skeleton edges
COCO17_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14),
    (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 11), (6, 12)
]

MODEL_PATH = os.environ.get("AQ10_MODEL", "model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load {MODEL_PATH}: {e}. Falling back to heuristic.")
else:
    print("[INFO] No model.pkl found; using heuristic scoring.")

QUESTIONS = [
    "I often notice small sounds when others do not.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I find it easy to do more than one thing at once.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I find it easy to 'read between the lines' when someone is talking to me.",
    "I know how to tell if someone listening to me is getting bored.",
    "When I'm reading a story, I find it difficult to work out the characters' intentions.",
    "I like to collect information about categories of things.",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "I find it difficult to work out people's intentions."
]

def draw_skeleton(img: np.ndarray, kp: np.ndarray, scores: np.ndarray, thr: float = 0.2):
    for i, (x, y) in enumerate(kp.astype(int)):
        if scores[i] >= thr:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    for a, b in COCO17_EDGES:
        if scores[a] >= thr and scores[b] >= thr:
            pa = tuple(kp[a].astype(int))
            pb = tuple(kp[b].astype(int))
            cv2.line(img, pa, pb, (255, 0, 0), 2)


def normalize_xy(kp_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    out = kp_xy.copy().astype(np.float32)
    out[:, 0] /= max(1, w)
    out[:, 1] /= max(1, h)
    return out


# -----------------------------
# App + State
# -----------------------------

app = Flask(__name__, template_folder="templates")

latest_lock = threading.Lock()
latest_pose: Dict[str, object] = {
    "kp": None,   # (17,2)
    "sc": None,   # (17,)
    "wh": None,   # (w,h)
    "ts": 0.0,    # timestamp
}


def camera_loop(src=0, model_complexity=1):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        ema_kp = None
        alpha = 0.6  # smoothing factor

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                mp_xy = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
                mp_sc = np.array([l.visibility for l in lm], dtype=np.float32)

                kp17 = mp_xy[MP_TO_COCO17]
                sc17 = mp_sc[MP_TO_COCO17]

                if ema_kp is None:
                    ema_kp = kp17.copy()
                else:
                    ema_kp = alpha * kp17 + (1 - alpha) * ema_kp

                now = time.time()
                with latest_lock:
                    latest_pose["kp"] = ema_kp.copy()
                    latest_pose["sc"] = sc17.copy()
                    latest_pose["wh"] = (w, h)
                    latest_pose["ts"] = now

                with rec_lock:
                    if active_rec["section_id"] is not None:
                        kp_norm = normalize_xy(ema_kp, w, h)
                        frame_entry = {
                            "t": now,
                            "xy": ema_kp.tolist(),
                            "xy_norm": kp_norm.tolist(),
                            "scores": sc17.tolist(),
                        }
                        active_rec["buffer"].append(frame_entry)

            overlay = frame.copy()
            with latest_lock:
                if latest_pose["kp"] is not None and latest_pose["sc"] is not None:
                    draw_skeleton(overlay, latest_pose["kp"], latest_pose["sc"])

            ret, jpeg = cv2.imencode(".jpg", overlay)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )


# -----------------------------
# Recording manager
# -----------------------------

SAVE_ROOT = os.environ.get("SAVE_ROOT", "sessions")
os.makedirs(SAVE_ROOT, exist_ok=True)

rec_lock = threading.Lock()
active_rec = {
    "section_id": None,
    "activity_id": None,
    "started_at": None,
    "buffer": []
}


def sanitize_id(s: str) -> str:
    return "".join(c for c in s if (c.isalnum() or c in "-_")).strip() or "unknown"


def ensure_dirs(section_id: str, activity_id: str) -> str:
    sec = sanitize_id(section_id)
    act = sanitize_id(activity_id)
    path = os.path.join(SAVE_ROOT, sec, act)
    os.makedirs(path, exist_ok=True)
    return path

# put near your other helpers
def count_saved_records(section_id: str) -> int:
    sec = sanitize_id(section_id)
    base = os.path.join(SAVE_ROOT, sec)
    total = 0
    if not os.path.isdir(base):
        return 0
    for act in os.listdir(base):
        d = os.path.join(base, act)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                total += 1
    return total


def dump_recording_to_file(section_id: str, activity_id: str, started_at: float, buffer: list) -> str:
    payload = {
        "section_id": section_id,
        "activity_id": activity_id,
        "started_at": started_at,
        "ended_at": time.time(),
        "num_frames": len(buffer),
        "frames": buffer,
    }

    out_dir = ensure_dirs(section_id, activity_id)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}_{len(buffer)}f.json"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return out_path


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    cam_idx = int(os.environ.get("CAMERA_IDX", "0"))
    mcomp = int(os.environ.get("MP_MODEL_COMPLEXITY", "1"))
    return Response(
        camera_loop(cam_idx, mcomp),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/record/start", methods=["POST"])
def record_start():
    data = request.get_json(silent=True) or {}
    section_id = data.get("section_id")
    activity_id = data.get("activity_id")

    if not section_id or not activity_id:
        return jsonify({"ok": False, "error": "section_id and activity_id are required"}), 400

    with rec_lock:
        if active_rec["section_id"] is not None:
            return jsonify({"ok": False, "error": "another recording is already active"}), 409

        active_rec["section_id"] = sanitize_id(section_id)
        active_rec["activity_id"] = sanitize_id(activity_id)
        active_rec["started_at"] = time.time()
        active_rec["buffer"] = []

    return jsonify({"ok": True, "message": "recording started"}), 200


@app.route("/record/stop", methods=["POST"])
def record_stop():
    with rec_lock:
        if active_rec["section_id"] is None:
            return jsonify({"ok": False, "error": "no active recording"}), 409

        section_id = active_rec["section_id"]
        activity_id = active_rec["activity_id"]
        started_at = active_rec["started_at"]
        buffer = active_rec["buffer"][:]  # copy

        # reset
        active_rec["section_id"] = None
        active_rec["activity_id"] = None
        active_rec["started_at"] = None
        active_rec["buffer"] = []

    # write file outside lock
    out_path = dump_recording_to_file(section_id, activity_id, started_at, buffer)
    rel_path = os.path.relpath(out_path, start=SAVE_ROOT)

    # NEW: compute updated count for this section
    n = count_saved_records(section_id)

    return jsonify({
        "ok": True,
        "saved": out_path,
        "relative": rel_path,
        "section_id": section_id,
        "section_record_count": n,
        "min_required": 6,
        "eligible": n >= 6
    }), 200

@app.route("/session/status")
def session_status():
    section_id = request.args.get("section_id") or "session_001"
    n = count_saved_records(section_id)
    return jsonify({
        "ok": True,
        "section_id": sanitize_id(section_id),
        "section_record_count": n,
        "min_required": 6,
        "eligible": n >= 6
    }), 200

@app.route("/questionnaire")
def questionnaire():
    # Optional: carry section_id from querystring to stitch results with recordings
    section_id = request.args.get("section_id", "session_001")
    return render_template("questionnaire.html", questions=QUESTIONS, section_id=section_id)

@app.route("/questionnaire/submit", methods=["POST"])
def questionnaire_submit():
    try:
        section_id = request.form.get("section_id", "session_001")

        # Collect 10 radio answers: '1' or '0'
        answers = []
        for i in range(1, 11):
            v = request.form.get(f"q{i}")
            if v not in ("0", "1"):
                return render_template(
                    "questionnaire_result.html",
                    result={"error": f"Missing answer for question {i}.", "answers": []},
                    section_id=section_id
                )
            answers.append(int(v))

        aq10_score = sum(answers)

        # Use model if present; otherwise heuristic
        if model is not None:
            try:
                pred = int(model.predict([answers])[0])
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba([answers])[0][1])
                else:
                    probability = aq10_score / 10.0
            except Exception as e:
                print(f"[WARN] Model inference failed: {e}. Using heuristic.")
                pred = 1 if aq10_score >= 7 else 0
                probability = aq10_score / 10.0
        else:
            pred = 1 if aq10_score >= 7 else 0
            probability = aq10_score / 10.0

        # Risk label
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Binary recommendation text for the user
        if pred == 1:
            user_message = (
                "Your responses suggest traits that may be consistent with autism. "
                "We recommend talking with a qualified clinician or specialist for a full evaluation."
            )
        else:
            user_message = (
                "Your responses do not strongly indicate autistic traits. "
                "If you still have concerns about development, communication, or behavior, "
                "you can still speak with a clinician for peace of mind."
            )

        # Build result object for the template
        result = {
            "section_id": section_id,
            "aq10_score": aq10_score,
            "risk_level": risk_level,
            "probability": probability,   # 0..1
            "answers": answers,
            "prediction": pred,           # 0 or 1 (binary)
            "user_message": user_message, # <— NEW
            "error": None
        }

        return render_template("questionnaire_result.html", result=result, section_id=section_id)

    except Exception as e:
        return render_template(
            "questionnaire_result.html",
            result={"error": f"Unexpected error: {e}", "answers": []},
            section_id=request.form.get("section_id", "session_001")
        )


@app.route("/record/reset", methods=["POST"])
def record_reset():
    with rec_lock:
        if active_rec["section_id"] is None:
            return jsonify({"ok": False, "error": "no active recording to reset"}), 409

        active_rec["buffer"] = []
        active_rec["started_at"] = time.time()

    return jsonify({"ok": True, "message": "recording buffer reset"}), 200


@app.route("/pose.json")
def pose_json():
    with latest_lock:
        if latest_pose["kp"] is None:
            return jsonify({"ok": False, "message": "no pose yet"}), 200

        w, h = latest_pose["wh"]
        kp_xy = np.asarray(latest_pose["kp"]).astype(float)
        kp_norm = normalize_xy(kp_xy, w, h).astype(float)
        scores = np.asarray(latest_pose["sc"]).astype(float)

    return jsonify({
        "ok": True,
        "timestamp": latest_pose["ts"],
        "width": w,
        "height": h,
        "coco17_keypoints_xy": kp_xy.tolist(),
        "coco17_keypoints_xy_norm": kp_norm.tolist(),
        "coco17_scores": scores.tolist()
    }), 200


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True
    )
