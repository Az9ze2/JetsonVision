"""
Student Enrollment Tool
=======================
Captures face embeddings from the RealSense camera using the same
detector + recognizer pipeline as visual_jetson_async.py.

Run:
    python enroll_student.py
    python enroll_student.py --rec-model models/buffalo_l/w600k_r50.onnx

Controls (in the camera window):
    SPACE  – capture current face for this angle
    R      – retake the last capture
    Q      – quit without saving
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from vision.detector_factory import create_scrfd_detector
from vision.recognizer import FaceRecognizer
from vision.database import EnrollmentDatabase

# ── Angles to capture (in order) ─────────────────────────────────────────────
# Only poses within yaw < 25° and pitch < 15° — matching the recognition trigger.
# UP/DOWN are excluded because recognition never fires when pitch > 15°.
ANGLES = [
    ("STRAIGHT",       "Look straight at the camera",          "0°"),
    ("SLIGHT LEFT",    "Turn your head slightly to the LEFT",   "~15°"),
    ("LEFT",           "Turn your head more to the LEFT",       "~25°"),
    ("SLIGHT RIGHT",   "Turn your head slightly to the RIGHT",  "~15°"),
    ("RIGHT",          "Turn your head more to the RIGHT",      "~25°"),
]

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
ORANGE = (0, 165, 255)
RED    = (0, 0, 220)
CYAN   = (255, 220, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Student face enrollment tool")
    p.add_argument("--det-model", default="models/buffalo_l/det_10g.onnx")
    p.add_argument("--rec-model", default="models/buffalo_l/w600k_r50.onnx")
    p.add_argument("--db-path",   default="data/enrollments.json")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Student info prompt (terminal)
# ─────────────────────────────────────────────────────────────────────────────

def prompt_student_info() -> dict:
    print("\n" + "="*55)
    print("  STUDENT ENROLLMENT – fill in student details")
    print("="*55)

    def ask(label, required=True):
        while True:
            val = input(f"  {label}: ").strip()
            if val or not required:
                return val
            print("  (required – please enter a value)")

    student_id    = ask("Student ID (e.g. 65011386)")
    fullname_eng  = ask("Full name English (e.g. Natcha Rungruang)")
    fullname_thai = ask("Full name Thai   (e.g. ณัชชา รุ่งเรือง)", required=False)
    nickname_eng  = ask("Nickname English (e.g. Fern)")
    nickname_thai = ask("Nickname Thai    (e.g. เฟิร์น)", required=False)

    print()
    print("  ── Summary ──────────────────────────────")
    print(f"  ID            : {student_id}")
    print(f"  Full name ENG : {fullname_eng}")
    print(f"  Full name TH  : {fullname_thai or '(none)'}")
    print(f"  Nickname ENG  : {nickname_eng}")
    print(f"  Nickname TH   : {nickname_thai or '(none)'}")
    print("  ─────────────────────────────────────────")
    confirm = input("  Confirm? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("  Restarting input...")
        return prompt_student_info()

    return {
        "student_id":    student_id,
        "fullname_eng":  fullname_eng,
        "fullname_thai": fullname_thai,
        "nickname_eng":  nickname_eng,
        "nickname_thai": nickname_thai,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text(frame, msg, pos, color=WHITE, scale=0.6, thickness=2):
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_DUPLEX, scale, BLACK, thickness + 2)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness)


def _draw_ui(frame, angle_name, instruction, degree, captured, total, feedback, face_ok):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    _text(frame, f"ENROLLMENT  –  Angle {captured + 1}/{total}: {angle_name}", (10, 38), CYAN, 0.7, 2)

    # Instruction bar
    cv2.rectangle(frame, (0, h - 90), (w, h - 55), (20, 20, 20), -1)
    _text(frame, instruction, (10, h - 65), WHITE, 0.55, 1)

    # Controls bar
    cv2.rectangle(frame, (0, h - 52), (w, h), (30, 30, 30), -1)
    _text(frame, "SPACE = capture    R = retake    Q = quit", (10, h - 22), ORANGE, 0.5, 1)

    # Face feedback
    if feedback:
        color = GREEN if face_ok else RED
        cv2.rectangle(frame, (0, 62), (w, 100), (0, 0, 0), -1)
        _text(frame, feedback, (10, 90), color, 0.55, 1)

    # Prominent degree display — centre of screen top area
    deg_size = cv2.getTextSize(degree, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)[0]
    deg_x = (w - deg_size[0]) // 2
    _text(frame, degree, (deg_x, 130), CYAN, 2.5, 3)

    # Captured count dots
    for i in range(total):
        cx = w - total * 22 + i * 22 - 10
        cy = 38
        col = GREEN if i < captured else (80, 80, 80)
        cv2.circle(frame, (cx, cy), 8, col, -1)


def _draw_bbox(frame, bbox, landmarks, face_ok):
    x1, y1, x2, y2 = map(int, bbox)
    color = GREEN if face_ok else ORANGE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if landmarks:
        for lx, ly in landmarks:
            cv2.circle(frame, (int(lx), int(ly)), 3, RED, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Main enrollment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_enrollment(info: dict, args):
    # ── Init models ──────────────────────────────────────────────────────────
    logger.info("Loading face detector…")
    detector = create_scrfd_detector(
        model_path=args.det_model,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_size=(640, 640),
        device="cpu",
        use_tensorrt=False,
    )

    logger.info(f"Loading face recognizer ({args.rec_model})…")
    recognizer = FaceRecognizer(
        model_path=args.rec_model,
        device="cpu",
        use_tensorrt=False,
    )

    db = EnrollmentDatabase(args.db_path)

    # ── RealSense camera ──────────────────────────────────────────────────────
    logger.info("Starting RealSense camera…")
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)

    embeddings = []
    angle_idx  = 0
    feedback   = ""
    face_ok    = False

    print(f"\n  Camera ready. Follow the on-screen instructions.")
    print(f"  Press SPACE to capture each angle, R to retake.\n")

    try:
        while angle_idx < len(ANGLES):
            angle_name, instruction, degree = ANGLES[angle_idx]

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            display = frame.copy()

            # Detect faces
            detections = detector.detect(frame)

            best_det = None
            if detections:
                # Pick largest face
                best_det = max(detections, key=lambda d: (
                    (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                ))
                lm = best_det.get("landmarks")
                x1, y1, x2, y2 = best_det["bbox"]
                face_w = x2 - x1
                face_h = y2 - y1
                face_ok = face_w >= 80 and face_h >= 80
                feedback = "Face detected – good position!" if face_ok else "Move closer (face too small)"
                _draw_bbox(display, best_det["bbox"], lm, face_ok)
            else:
                feedback = "No face detected – position yourself in view"
                face_ok  = False

            _draw_ui(display, angle_name, instruction, degree, len(embeddings), len(ANGLES), feedback, face_ok)
            cv2.imshow("Enrollment", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n  Enrollment cancelled – no data saved.")
                return False

            elif key == ord("r") and embeddings:
                embeddings.pop()
                angle_idx -= 1
                print(f"  Retaking angle: {ANGLES[angle_idx][0]} ({ANGLES[angle_idx][2]})")
                feedback = ""

            elif key == ord(" "):
                if not best_det:
                    print("  No face detected – cannot capture.")
                    continue
                if not face_ok:
                    print("  Face too small – move closer and try again.")
                    continue

                # Extract embedding using the same aligned pipeline
                try:
                    emb = recognizer.extract_embedding(
                        frame,
                        best_det["bbox"],
                        best_det.get("landmarks"),
                    )
                    embeddings.append(emb)
                    print(f"  ✓ Captured [{angle_name}]  ({len(embeddings)}/{len(ANGLES)})")
                    angle_idx += 1
                    feedback = ""
                except Exception as e:
                    print(f"  Embedding failed: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(embeddings) < len(ANGLES):
        print(f"\n  Only {len(embeddings)}/{len(ANGLES)} angles captured – not saving.")
        return False

    # ── Save to database ──────────────────────────────────────────────────────
    success = db.enroll_student(
        student_id    = info["student_id"],
        embeddings    = embeddings,
        fullname_eng  = info["fullname_eng"],
        fullname_thai = info["fullname_thai"],
        nickname_eng  = info["nickname_eng"],
        nickname_thai = info["nickname_thai"],
    )

    if success:
        print("\n" + "="*55)
        print(f"  ENROLLED: {info['nickname_eng']} ({info['fullname_eng']})")
        print(f"  ID      : {info['student_id']}")
        print(f"  Angles  : {len(embeddings)} embeddings saved")
        print(f"  DB      : {args.db_path}")
        print("="*55 + "\n")
    else:
        print("\n  ERROR: failed to save to database.")

    return success


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               colorize=True)

    args = _parse_args()

    while True:
        info = prompt_student_info()
        run_enrollment(info, args)

        again = input("  Enroll another student? [Y/n]: ").strip().lower()
        if again not in ("", "y", "yes"):
            break

    print("  Done.\n")


if __name__ == "__main__":
    main()
