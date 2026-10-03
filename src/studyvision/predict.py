import time
from collections import deque
from pathlib import Path

import cv2
import joblib

from studyvision.face_mesh import FaceMesh
from studyvision.features.eye import calculate_eye_features
from studyvision.session_report import SessionLogger


# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------

MODEL_FILE = "models/studyvision_random_forest.joblib"

# Phone detector
PHONE_CFG = "models/phone_detector/yolov4-tiny.cfg"
PHONE_WEIGHTS = "models/phone_detector/yolov4-tiny.weights"

PHONE_CLASS_ID = 67
PHONE_CONFIDENCE_THRESHOLD = 0.40
PHONE_DETECTION_DELAY_SECONDS = 1.5

# Camera / detection settings
WINDOW_SIZE = 30

DISTRACTION_DELAY_SECONDS = 3.0

# Drowsiness settings
EYE_CLOSED_EAR_THRESHOLD = 0.20
DROWSINESS_DELAY_SECONDS = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def draw_text(
    frame,
    text,
    position,
    scale=0.8,
    color=(255, 255, 255),
    thickness=2,
):
    """Draw readable text on the camera frame."""

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_ready_screen(frame):
    """Display the START SESSION screen."""

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        (25, 20, 60),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
    )

    draw_text(
        frame,
        "STUDYVISION",
        (40, 70),
        1.4,
        (255, 255, 255),
        3,
    )

    draw_text(
        frame,
        "Student Study Tracker",
        (43, 105),
        0.7,
        (220, 220, 255),
        2,
    )

    draw_text(
        frame,
        "READY",
        (40, 190),
        1.5,
        (100, 255, 180),
        3,
    )

    draw_text(
        frame,
        "Press S to START SESSION",
        (40, 245),
        0.9,
        (255, 255, 255),
        2,
    )

    draw_text(
        frame,
        "Press Q to EXIT",
        (40, 285),
        0.75,
        (220, 220, 220),
        2,
    )


def draw_break_screen(frame, break_duration):
    """Display the break screen."""

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        (45, 30, 10),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
    )

    draw_text(
        frame,
        "STUDYVISION",
        (40, 65),
        1.2,
        (255, 255, 255),
        3,
    )

    draw_text(
        frame,
        "BREAK TIME",
        (40, 145),
        1.4,
        (80, 210, 255),
        3,
    )

    draw_text(
        frame,
        f"Break duration: {break_duration:.0f}s",
        (40, 195),
        0.85,
        (255, 255, 255),
        2,
    )

    draw_text(
        frame,
        "Press B to RESUME SESSION",
        (40, 250),
        0.8,
        (255, 255, 255),
        2,
    )

    draw_text(
        frame,
        "Press Q to END SESSION",
        (40, 290),
        0.75,
        (220, 220, 220),
        2,
    )


def detect_phone(net, output_layers, frame):
    """Detect a visible mobile phone using YOLOv4-tiny."""

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    best_confidence = 0.0
    best_box = None

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = scores.argmax()

            confidence = float(
                scores[class_id]
            )

            if class_id != PHONE_CLASS_ID:
                continue

            if confidence < PHONE_CONFIDENCE_THRESHOLD:
                continue

            center_x = int(
                detection[0] * width
            )

            center_y = int(
                detection[1] * height
            )

            box_width = int(
                detection[2] * width
            )

            box_height = int(
                detection[3] * height
            )

            x = int(
                center_x - box_width / 2
            )

            y = int(
                center_y - box_height / 2
            )

            if confidence > best_confidence:

                best_confidence = confidence

                best_box = (
                    x,
                    y,
                    box_width,
                    box_height,
                )

    return (
        best_box is not None,
        best_confidence,
        best_box,
    )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("StudyVision - Student Study Tracker")
    print("=" * 60)
    print()

    print("Loading phone detector...")

    phone_net = cv2.dnn.readNetFromDarknet(
        PHONE_CFG,
        PHONE_WEIGHTS,
    )

    phone_layer_names = (
        phone_net.getLayerNames()
    )

    phone_output_layers = [
        phone_layer_names[i - 1]
        for i in
        phone_net.getUnconnectedOutLayers()
        .flatten()
    ]

    print("Phone detector loaded successfully.")

    print("Phone class: Cell phone")

    print(
        f"Phone confidence threshold: "
        f"{PHONE_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Phone confirmation delay: "
        f"{PHONE_DETECTION_DELAY_SECONDS:.1f} seconds"
    )

    print()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    face_mesh = FaceMesh()

    # -----------------------------------------------------
    # SESSION VARIABLES
    # -----------------------------------------------------

    session_started = False

    session_logger = None

    feature_history = deque(
        maxlen=WINDOW_SIZE
    )

    face_missing_since = None

    eyes_closed_since = None

    phone_detected_since = None

    break_started_at = None

    on_break = False

    previous_left_ear = None
    previous_right_ear = None

    print("Camera started.")

    print()

    print("=" * 60)
    print("READY")
    print("=" * 60)

    print()

    print("Press S to START SESSION")
    print("Press Q to EXIT")

    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read camera frame."
                )

                break

            now = time.time()

            # -------------------------------------------------
            # READY SCREEN
            # -------------------------------------------------

            if not session_started:

                draw_ready_screen(frame)

                cv2.imshow(
                    "StudyVision - Student Study Tracker",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("s"):

                    session_started = True

                    session_logger = SessionLogger(
                        started_at=time.time()
                    )

                    feature_history.clear()

                    face_missing_since = None

                    eyes_closed_since = None

                    phone_detected_since = None

                    break_started_at = None

                    on_break = False

                    previous_left_ear = None

                    previous_right_ear = None

                    print()

                    print("=" * 60)
                    print("STUDY SESSION STARTED")
                    print("=" * 60)

                    print()

                    print(
                        "B = Take Break"
                    )

                    print(
                        "Q = End Session"
                    )

                    print()

                elif key == ord("q"):

                    break

                continue

            # -------------------------------------------------
            # PHONE DETECTION
            # -------------------------------------------------

            phone_found, phone_confidence, phone_box = (
                detect_phone(
                    phone_net,
                    phone_output_layers,
                    frame,
                )
            )

            if phone_found:

                if phone_detected_since is None:

                    phone_detected_since = now

                phone_duration = (
                    now
                    - phone_detected_since
                )

            else:

                phone_detected_since = None

                phone_duration = 0.0

            phone_use = (
                phone_found
                and phone_duration
                >= PHONE_DETECTION_DELAY_SECONDS
            )

            # Draw phone box
            if (
                phone_found
                and phone_box is not None
            ):

                x, y, box_width, box_height = (
                    phone_box
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (
                        x + box_width,
                        y + box_height,
                    ),
                    (0, 255, 0),
                    2,
                )

                draw_text(
                    frame,
                    f"PHONE {phone_confidence:.2f}",
                    (
                        x,
                        max(y - 10, 20),
                    ),
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # -------------------------------------------------
            # BREAK MODE
            # -------------------------------------------------

            if on_break:

                break_duration = (
                    now
                    - break_started_at
                )

                session_logger.record(
                    now,
                    state="on_break",
                )

                draw_break_screen(
                    frame,
                    break_duration,
                )

                cv2.imshow(
                    "StudyVision - Student Study Tracker",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("b"):

                    on_break = False

                    break_started_at = None

                    face_missing_since = None

                    eyes_closed_since = None

                    phone_detected_since = None

                    print(
                        "Break ended. "
                        "Study session resumed."
                    )

                elif key == ord("q"):

                    break

                continue

            # -------------------------------------------------
            # FACE DETECTION
            # -------------------------------------------------

            height, width, _ = frame.shape

            faces = face_mesh.process(frame)

            # -------------------------------------------------
            # FACE FOUND
            # -------------------------------------------------

            if faces:

                face_missing_since = None

                landmarks = (
                    faces[0].landmark
                )

                eye_features = (
                    calculate_eye_features(
                        landmarks,
                        width,
                        height,
                    )
                )

                left_ear = (
                    eye_features["left_ear"]
                )

                right_ear = (
                    eye_features["right_ear"]
                )

                average_ear = (
                    eye_features["average_ear"]
                )

                # ---------------------------------------------
                # EYE STATE
                # ---------------------------------------------

                if (
                    average_ear
                    < EYE_CLOSED_EAR_THRESHOLD
                ):

                    if eyes_closed_since is None:

                        eyes_closed_since = now

                    eyes_closed_duration = (
                        now
                        - eyes_closed_since
                    )

                else:

                    eyes_closed_since = None

                    eyes_closed_duration = 0.0

                drowsy = (
                    average_ear
                    < EYE_CLOSED_EAR_THRESHOLD
                    and eyes_closed_duration
                    >= DROWSINESS_DELAY_SECONDS
                )

                # ---------------------------------------------
                # PRIORITY 1: PHONE
                # ---------------------------------------------

                if phone_use:

                    session_logger.record(
                        now,
                        state="phone_use",
                    )

                    draw_text(
                        frame,
                        "Status: Phone Use",
                        (20, 45),
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        (
                            f"Phone detected: "
                            f"{phone_duration:.1f}s"
                        ),
                        (20, 82),
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        "Please put the phone away",
                        (20, 115),
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # ---------------------------------------------
                # PRIORITY 2: DROWSINESS
                # ---------------------------------------------

                elif drowsy:

                    session_logger.record(
                        now,
                        state="drowsy",
                    )

                    draw_text(
                        frame,
                        "Status: Drowsy / Eyes Closed",
                        (20, 45),
                        0.85,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        (
                            f"Eyes closed: "
                            f"{eyes_closed_duration:.1f}s"
                        ),
                        (20, 82),
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        "Open your eyes to resume focus",
                        (20, 115),
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # ---------------------------------------------
                # PRIORITY 3: FOCUSED
                # ---------------------------------------------

                else:

                    session_logger.record(
                        now,
                        state="focused",
                    )

                    draw_text(
                        frame,
                        "Status: Focused",
                        (20, 45),
                        0.9,
                        (0, 255, 120),
                        2,
                    )

                    draw_text(
                        frame,
                        "Face detected",
                        (20, 80),
                        0.65,
                        (0, 255, 120),
                        2,
                    )

                # ---------------------------------------------
                # EAR DISPLAY
                # ---------------------------------------------

                draw_text(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (20, height - 55),
                    0.65,
                    (255, 255, 255),
                    2,
                )

            # -------------------------------------------------
            # FACE NOT FOUND
            # -------------------------------------------------

            else:

                eyes_closed_since = None

                if face_missing_since is None:

                    face_missing_since = now

                missing_duration = (
                    now
                    - face_missing_since
                )

                distracted = (
                    missing_duration
                    >= DISTRACTION_DELAY_SECONDS
                )

                if phone_use:

                    session_logger.record(
                        now,
                        state="phone_use",
                    )

                    draw_text(
                        frame,
                        "Status: Phone Use",
                        (20, 45),
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                elif distracted:

                    session_logger.record(
                        now,
                        state="distracted",
                    )

                    draw_text(
                        frame,
                        "Status: Distracted",
                        (20, 45),
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        (
                            f"Face not visible: "
                            f"{missing_duration:.1f}s"
                        ),
                        (20, 82),
                        0.65,
                        (0, 0, 255),
                        2,
                    )

                else:

                    draw_text(
                        frame,
                        "Face not visible",
                        (20, 45),
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                    draw_text(
                        frame,
                        (
                            "Distraction recorded "
                            "after 3 seconds"
                        ),
                        (20, 82),
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # -------------------------------------------------
            # PHONE CONFIRMATION
            # -------------------------------------------------

            if (
                phone_found
                and not phone_use
            ):

                draw_text(
                    frame,
                    (
                        f"Phone confirmation: "
                        f"{phone_duration:.1f}/"
                        f"{PHONE_DETECTION_DELAY_SECONDS:.1f}s"
                    ),
                    (20, 150),
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # -------------------------------------------------
            # CONTROLS
            # -------------------------------------------------

            draw_text(
                frame,
                "B = Break    Q = End Session",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                0.55,
                (220, 220, 220),
                2,
            )

            cv2.imshow(
                "StudyVision - Student Study Tracker",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # -------------------------------------------------
            # START BREAK
            # -------------------------------------------------

            if key == ord("b"):

                on_break = True

                break_started_at = time.time()

                face_missing_since = None

                eyes_closed_since = None

                phone_detected_since = None

                print(
                    "Break started."
                )

                print(
                    "Press B to resume "
                    "or Q to finish."
                )

            # -------------------------------------------------
            # END SESSION
            # -------------------------------------------------

            elif key == ord("q"):

                break

    finally:

        face_mesh.close()

        camera.release()

        cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # SAVE SESSION
    # ---------------------------------------------------------

    if (
        session_started
        and session_logger is not None
    ):

        csv_path, report_path = (
            session_logger.save(
                PROJECT_ROOT
            )
        )

        print()

        print("=" * 60)
        print(
            "STUDYVISION - SESSION COMPLETE"
        )
        print("=" * 60)

        print()

        print(
            "Session observations saved to:"
        )

        print(csv_path)

        print()

        print(
            "Dashboard saved to:"
        )

        print(report_path)

        print()

        print(
            "Great work! Your session "
            "has been recorded."
        )


if __name__ == "__main__":
    main()