
import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os

# ============================================================
# SMOOTH VIRTUAL MOUSE
# MediaPipe 1.x Tasks API
# Designed for 640x480 webcams
# ============================================================

# ---------------- CAMERA ----------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------------- CONTROL AREA ----------------

# Keep the hand inside this area.
MARGIN_X = 70
MARGIN_Y = 55

# ---------------- CURSOR ----------------

# Larger = smoother but slower
SMOOTHING = 7.0

# Ignore tiny fingertip movement
DEAD_ZONE = 2.0

# ---------------- GESTURES ----------------

# Pinch thresholds
PINCH_START = 0.27
PINCH_END = 0.36

# Gesture must remain stable this long
GESTURE_CONFIRM_TIME = 0.10

# Click cooldown
CLICK_COOLDOWN = 0.40

# ---------------- SCROLL ----------------

SCROLL_INTERVAL = 0.12
SCROLL_THRESHOLD = 7

# ---------------- MODEL ----------------

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "hand_landmarker.task"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(a, b):

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def hand_scale(hand):

    return max(
        distance(hand[0], hand[9]),
        0.001
    )


def normalized_distance(hand, a, b):

    return (
        distance(hand[a], hand[b])
        / hand_scale(hand)
    )


def finger_extended(hand, tip, pip):

    return hand[tip].y < hand[pip].y


def index_extended(hand):

    return finger_extended(
        hand,
        8,
        6
    )


def middle_extended(hand):

    return finger_extended(
        hand,
        12,
        10
    )


def ring_extended(hand):

    return finger_extended(
        hand,
        16,
        14
    )


def pinky_extended(hand):

    return finger_extended(
        hand,
        20,
        18
    )


def is_open_palm(hand):

    return (
        index_extended(hand)
        and middle_extended(hand)
        and ring_extended(hand)
        and pinky_extended(hand)
    )


def is_fist(hand):

    return (
        not index_extended(hand)
        and not middle_extended(hand)
        and not ring_extended(hand)
        and not pinky_extended(hand)
    )


def is_index_only(hand):

    return (
        index_extended(hand)
        and not middle_extended(hand)
        and not ring_extended(hand)
        and not pinky_extended(hand)
    )


def is_index_pinch(hand):

    return normalized_distance(
        hand,
        4,
        8
    ) < PINCH_START


def is_middle_pinch(hand):

    return normalized_distance(
        hand,
        4,
        12
    ) < PINCH_START


def map_value(
    value,
    in_min,
    in_max,
    out_min,
    out_max
):

    value = max(
        in_min,
        min(value, in_max)
    )

    return out_min + (
        (value - in_min)
        / (in_max - in_min)
    ) * (
        out_max - out_min
    )


def draw_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255)
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


def draw_hand(frame, hand, width, height):

    points = []

    for landmark in hand:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17)
    ]

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 120, 0),
            2
        )

    return points


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: hand_landmarker.task not found.")
        print()
        print(
            "Expected location:"
        )
        print(MODEL_PATH)
        print()

        return

    print()
    print("Starting Smooth Virtual Mouse...")
    print()

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    BaseOptions = mp.tasks.BaseOptions

    Vision = mp.tasks.vision

    options = Vision.HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=Vision.RunningMode.VIDEO,

        num_hands=1,

        min_hand_detection_confidence=0.75,

        min_hand_presence_confidence=0.75,

        min_tracking_confidence=0.75
    )

    landmarker = (
        Vision.HandLandmarker
        .create_from_options(options)
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        print("ERROR: Cannot open camera.")

        landmarker.close()

        return

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Camera: {actual_width}x{actual_height}"
    )

    # --------------------------------------------------------
    # Screen
    # --------------------------------------------------------

    screen_width, screen_height = pyautogui.size()

    current_x = screen_width / 2
    current_y = screen_height / 2

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    current_gesture = "NONE"

    candidate_gesture = "NONE"

    candidate_start = time.time()

    dragging = False

    last_left_click = 0
    last_right_click = 0

    last_scroll = 0

    previous_wrist_y = None

    timestamp_ms = 0

    # Smoothed fingertip
    smooth_index_x = None
    smooth_index_y = None

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "ERROR: Camera frame failed."
                )

                break

            # Mirror image
            frame = cv2.flip(
                frame,
                1
            )

            height, width = frame.shape[:2]

            # ------------------------------------------------
            # Control area
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (MARGIN_X, MARGIN_Y),
                (
                    width - MARGIN_X,
                    height - MARGIN_Y
                ),
                (0, 255, 255),
                2
            )

            draw_text(
                frame,
                "CONTROL AREA",
                MARGIN_X,
                MARGIN_Y - 10,
                (0, 255, 255)
            )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                image,
                timestamp_ms
            )

            status = "NO HAND"

            # =================================================
            # HAND FOUND
            # =================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                points = draw_hand(
                    frame,
                    hand,
                    width,
                    height
                )

                # ------------------------------------------------
                # Important points
                # ------------------------------------------------

                index_tip = points[8]

                wrist = points[0]

                # Highlight index fingertip
                cv2.circle(
                    frame,
                    index_tip,
                    9,
                    (0, 255, 255),
                    -1
                )

                # =================================================
                # RAW GESTURE CLASSIFICATION
                # =================================================

                index_pinch = is_index_pinch(hand)

                middle_pinch = is_middle_pinch(hand)

                fist = is_fist(hand)

                open_palm = is_open_palm(hand)

                index_only = is_index_only(hand)

                # ------------------------------------------------
                # Strict gesture priority
                # ------------------------------------------------

                if index_pinch and not middle_pinch:

                    detected_gesture = "LEFT_CLICK"

                elif middle_pinch and not index_pinch:

                    detected_gesture = "RIGHT_CLICK"

                elif fist:

                    detected_gesture = "DRAG"

                elif open_palm:

                    detected_gesture = "SCROLL"

                elif index_only:

                    detected_gesture = "MOVE"

                else:

                    detected_gesture = "NONE"

                # =================================================
                # GESTURE STABILITY
                # =================================================

                if detected_gesture != candidate_gesture:

                    candidate_gesture = (
                        detected_gesture
                    )

                    candidate_start = (
                        time.time()
                    )

                elif (
                    time.time()
                    - candidate_start
                    >= GESTURE_CONFIRM_TIME
                ):

                    current_gesture = (
                        candidate_gesture
                    )

                # =================================================
                # CURRENT GESTURE
                # =================================================

                now = time.time()

                # =================================================
                # LEFT CLICK
                # =================================================

                if current_gesture == "LEFT_CLICK":

                    status = "LEFT CLICK"

                    if (
                        now - last_left_click
                        > CLICK_COOLDOWN
                    ):

                        pyautogui.click()

                        last_left_click = now

                    # Never move cursor
                    continue_frame = False

                # =================================================
                # RIGHT CLICK
                # =================================================

                elif current_gesture == "RIGHT_CLICK":

                    status = "RIGHT CLICK"

                    if (
                        now - last_right_click
                        > CLICK_COOLDOWN
                    ):

                        pyautogui.rightClick()

                        last_right_click = now

                    continue_frame = False

                # =================================================
                # DRAG
                # =================================================

                elif current_gesture == "DRAG":

                    status = "DRAG"

                    if not dragging:

                        pyautogui.mouseDown()

                        dragging = True

                    continue_frame = False

                else:

                    # Release drag
                    if dragging:

                        pyautogui.mouseUp()

                        dragging = False

                    continue_frame = True

                # =================================================
                # SCROLL
                # =================================================

                if current_gesture == "SCROLL":

                    status = "SCROLL"

                    wrist_y = wrist[1]

                    if previous_wrist_y is not None:

                        movement = (
                            wrist_y
                            - previous_wrist_y
                        )

                        if (
                            abs(movement)
                            >= SCROLL_THRESHOLD
                            and
                            now - last_scroll
                            >= SCROLL_INTERVAL
                        ):

                            if movement < 0:

                                pyautogui.scroll(
                                    2
                                )

                            else:

                                pyautogui.scroll(
                                    -2
                                )

                            last_scroll = now

                    previous_wrist_y = wrist_y

                else:

                    previous_wrist_y = None

                # =================================================
                # MOVE
                # =================================================

                if (
                    current_gesture == "MOVE"
                ):

                    status = "MOVE"

                    index_x = index_tip[0]

                    index_y = index_tip[1]

                    # ---------------------------------------------
                    # Smooth fingertip position
                    # ---------------------------------------------

                    if smooth_index_x is None:

                        smooth_index_x = (
                            index_x
                        )

                        smooth_index_y = (
                            index_y
                        )

                    else:

                        smooth_index_x += (
                            index_x
                            - smooth_index_x
                        ) / SMOOTHING

                        smooth_index_y += (
                            index_y
                            - smooth_index_y
                        ) / SMOOTHING

                    # ---------------------------------------------
                    # Map camera -> screen
                    # ---------------------------------------------

                    target_x = map_value(
                        smooth_index_x,
                        MARGIN_X,
                        width - MARGIN_X,
                        0,
                        screen_width - 1
                    )

                    target_y = map_value(
                        smooth_index_y,
                        MARGIN_Y,
                        height - MARGIN_Y,
                        0,
                        screen_height - 1
                    )

                    # ---------------------------------------------
                    # Smooth cursor
                    # ---------------------------------------------

                    dx = (
                        target_x
                        - current_x
                    )

                    dy = (
                        target_y
                        - current_y
                    )

                    movement = math.hypot(
                        dx,
                        dy
                    )

                    if movement > DEAD_ZONE:

                        # Adaptive smoothing
                        if movement > 150:

                            factor = 0.35

                        elif movement > 80:

                            factor = 0.25

                        else:

                            factor = 0.15

                        current_x += (
                            dx * factor
                        )

                        current_y += (
                            dy * factor
                        )

                        current_x = max(
                            0,
                            min(
                                screen_width - 1,
                                current_x
                            )
                        )

                        current_y = max(
                            0,
                            min(
                                screen_height - 1,
                                current_y
                            )
                        )

                        pyautogui.moveTo(
                            int(current_x),
                            int(current_y)
                        )

                # =================================================
                # RESET FINGER SMOOTHING WHEN NOT MOVING
                # =================================================

                if current_gesture != "MOVE":

                    smooth_index_x = None
                    smooth_index_y = None

                # =================================================
                # STATUS DISPLAY
                # =================================================

                cv2.rectangle(
                    frame,
                    (5, 5),
                    (245, 45),
                    (20, 20, 20),
                    -1
                )

                draw_text(
                    frame,
                    status,
                    15,
                    32,
                    (0, 255, 0)
                )

            else:

                # ------------------------------------------------
                # No hand
                # ------------------------------------------------

                if dragging:

                    pyautogui.mouseUp()

                    dragging = False

                current_gesture = "NONE"

                candidate_gesture = "NONE"

                previous_wrist_y = None

                smooth_index_x = None

                smooth_index_y = None

                draw_text(
                    frame,
                    "NO HAND",
                    15,
                    32,
                    (0, 0, 255)
                )

            # =================================================
            # INSTRUCTIONS
            # =================================================

            instructions = [
                "INDEX ONLY  = MOVE",
                "PINCH INDEX = LEFT CLICK",
                "PINCH MIDDLE = RIGHT CLICK",
                "FIST = DRAG",
                "OPEN PALM = SCROLL",
                "ESC = EXIT"
            ]

            y = 75

            for instruction in instructions:

                draw_text(
                    frame,
                    instruction,
                    8,
                    y
                )

                y += 23

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "Smooth Virtual Mouse",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:

                break

    finally:

        if dragging:

            pyautogui.mouseUp()

        cap.release()

        cv2.destroyAllWindows()

        landmarker.close()

        print()
        print("Virtual Mouse stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()

