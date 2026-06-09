import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import math
import joblib

# =========================================
# LOAD MACHINE LEARNING MODEL
# =========================================
model = joblib.load("gesture_model.pkl")

# =========================================
# ARDUINO SERIAL CONNECTION
# =========================================
arduino = serial.Serial("COM4", 9600)
time.sleep(3)

# =========================================
# MEDIAPIPE
# =========================================
mp_pose  = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# =========================================
# MOTION FILTER
# =========================================
SMOOTH_ALPHA = 0.88
DEADZONE = 2
MAX_STEP = 3
WRIST_ROT_GAIN = 1.2

# =========================================
# DEFAULT SAFE / HOME POSITION
# =========================================
DEFAULT = {
    "shoulder": 90,
    "elbow": 5,
    "wristTilt": 90,
    "wristRotate": 90,
    "gripper": 60
}

# =========================================
# JOINT LIMITS
# =========================================
LIMITS = {
    "shoulder": (20, 160),
    "elbow": (5, 20),
    "wristTilt": (20, 160),
    "wristRotate": (20, 160),
    "gripper": (30, 120),
}

state = DEFAULT.copy()
neutral_roll = None

# =========================================
# GESTURE LATCH SYSTEM
# =========================================
active_command = "THUMBS UP"
last_seen_gesture = "NONE"
gesture_counter = 0
STABLE_GESTURE_FRAMES = 8

VALID_GESTURES = [
    "OPEN PALM",
    "FIST",
    "POINT",
    "PEACE",
    "THREE",
    "ROCK",
    "PINKY",
    "THUMBS UP",
    "OK",
    "SPIDER"
]

# =========================================
# HELPERS
# =========================================
def clamp(x, a, b):
    return int(max(a, min(b, x)))

def clamp_joint(name, value):
    lo, hi = LIMITS[name]
    return clamp(value, lo, hi)

def vec2(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)

def angle_between(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cosang = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def smooth(old, target):
    if abs(target - old) < DEADZONE:
        target = old

    sm = old * SMOOTH_ALPHA + target * (1 - SMOOTH_ALPHA)

    diff = sm - old

    if diff > MAX_STEP:
        sm = old + MAX_STEP

    if diff < -MAX_STEP:
        sm = old - MAX_STEP

    return int(sm)

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# =========================================
# ML FEATURE EXTRACTION
# =========================================
def extract_features(hand_lm):
    features = []

    wrist_x = hand_lm[0].x
    wrist_y = hand_lm[0].y
    wrist_z = hand_lm[0].z

    for lm in hand_lm:
        features.append(lm.x - wrist_x)
        features.append(lm.y - wrist_y)
        features.append(lm.z - wrist_z)

    return features

# =========================================
# UPDATE ACTIVE COMMAND
# =========================================
def update_active_command(predicted_gesture):
    global active_command, last_seen_gesture, gesture_counter

    if predicted_gesture not in VALID_GESTURES:
        return active_command

    if predicted_gesture == last_seen_gesture:
        gesture_counter += 1
    else:
        last_seen_gesture = predicted_gesture
        gesture_counter = 1

    if gesture_counter >= STABLE_GESTURE_FRAMES:
        if predicted_gesture != active_command:
            active_command = predicted_gesture
            print("New active command:", active_command)

    return active_command

# =========================================
# APPLY GESTURE COMMANDS
# =========================================
def apply_gesture_command(command,
                          shoulder,
                          elbow,
                          wristTilt,
                          wristRotate,
                          gripper):

    if command == "OPEN PALM":
        gripper = 120

    elif command == "FIST":
        gripper = 30

    elif command == "POINT":
        shoulder = 135
        elbow = 5
        wristTilt = 90

    elif command == "PEACE":
        wristRotate = 40

    elif command == "THREE":
        wristRotate = 140

    elif command == "ROCK":
        shoulder = 120
        elbow = 20
        wristTilt = 60
        wristRotate = 90
        gripper = 30

    elif command == "PINKY":
        shoulder = 95
        elbow = 15
        wristTilt = 120
        wristRotate = 90
        gripper = 50

    elif command == "THUMBS UP":
        shoulder = DEFAULT["shoulder"]
        elbow = DEFAULT["elbow"]
        wristTilt = DEFAULT["wristTilt"]
        wristRotate = DEFAULT["wristRotate"]
        gripper = DEFAULT["gripper"]

    elif command == "OK":
        shoulder = 70
        elbow = 20
        wristTilt = 110
        wristRotate = 90
        gripper = 120

    elif command == "SPIDER":
        shoulder = 140
        elbow = 10
        wristTilt = 70
        wristRotate = 120
        gripper = 120

    return shoulder, elbow, wristTilt, wristRotate, gripper

# =========================================
# PALM ROLL FOR WRIST ROTATION
# =========================================
def palm_roll(hand_lm):
    W = np.array([hand_lm[0].x,  hand_lm[0].y,  hand_lm[0].z], dtype=np.float32)
    I = np.array([hand_lm[5].x,  hand_lm[5].y,  hand_lm[5].z], dtype=np.float32)
    M = np.array([hand_lm[9].x,  hand_lm[9].y,  hand_lm[9].z], dtype=np.float32)
    P = np.array([hand_lm[17].x, hand_lm[17].y, hand_lm[17].z], dtype=np.float32)

    a = normalize(M - W)
    b = normalize(I - P)

    b_proj = b - np.dot(b, a) * a
    b_proj = normalize(b_proj)

    ex = np.array([1, 0, 0], dtype=np.float32)
    ey = np.array([0, 1, 0], dtype=np.float32)

    ex_p = ex - np.dot(ex, a) * a
    ey_p = ey - np.dot(ey, a) * a

    if np.linalg.norm(ex_p) < 1e-6:
        ex_p = np.array([0, 0, 1], dtype=np.float32) - np.dot(
            np.array([0, 0, 1], dtype=np.float32), a
        ) * a

    if np.linalg.norm(ey_p) < 1e-6:
        ey_p = np.array([0, 0, 1], dtype=np.float32) - np.dot(
            np.array([0, 0, 1], dtype=np.float32), a
        ) * a

    ex_p = normalize(ex_p)
    ey_p = normalize(ey_p)

    return math.atan2(
        float(np.dot(b_proj, ey_p)),
        float(np.dot(b_proj, ex_p))
    )

# =========================================
# AUTO LEFT / RIGHT ARM SELECTION
# =========================================
def landmark_to_px(lm, index, w, h):
    return int(lm[index].x * w), int(lm[index].y * h)

def landmark_visibility(lm, index):
    if hasattr(lm[index], "visibility"):
        return float(lm[index].visibility)
    return 1.0

def get_arm_points(lm, side, w, h):
    if side == "RIGHT":
        shoulder_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        elbow_index    = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        wrist_index    = mp_pose.PoseLandmark.RIGHT_WRIST.value
    else:
        shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        elbow_index    = mp_pose.PoseLandmark.LEFT_ELBOW.value
        wrist_index    = mp_pose.PoseLandmark.LEFT_WRIST.value

    SHO = landmark_to_px(lm, shoulder_index, w, h)
    ELB = landmark_to_px(lm, elbow_index, w, h)
    WRI = landmark_to_px(lm, wrist_index, w, h)

    visibility = (
        landmark_visibility(lm, shoulder_index) +
        landmark_visibility(lm, elbow_index) +
        landmark_visibility(lm, wrist_index)
    ) / 3.0

    in_frame_score = 0

    for idx in [shoulder_index, elbow_index, wrist_index]:
        if -0.1 <= lm[idx].x <= 1.1 and -0.1 <= lm[idx].y <= 1.1:
            in_frame_score += 1

    in_frame_score = in_frame_score / 3.0

    return SHO, ELB, WRI, visibility, in_frame_score

def choose_best_arm(lm, w, h, hand_wrist_px=None):
    candidates = []

    for side in ["RIGHT", "LEFT"]:
        SHO, ELB, WRI, visibility, in_frame_score = get_arm_points(lm, side, w, h)

        score = visibility + (0.3 * in_frame_score)

        if hand_wrist_px is not None:
            dist = np.linalg.norm(np.array(WRI) - np.array(hand_wrist_px))
            max_dist = np.sqrt(w ** 2 + h ** 2)
            closeness = 1.0 - min(dist / max_dist, 1.0)
            score += 0.8 * closeness

        candidates.append({
            "side": side,
            "SHO": SHO,
            "ELB": ELB,
            "WRI": WRI,
            "visibility": visibility,
            "score": score
        })

    best = max(candidates, key=lambda x: x["score"])

    if best["visibility"] < 0.35:
        return None, None, None, None

    return best["side"], best["SHO"], best["ELB"], best["WRI"]

# =========================================
# DRAW TEXT WITH WHITE BACKGROUND
# =========================================
def draw_black_text(frame, text, position, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_w, text_h = text_size

    cv2.rectangle(
        frame,
        (x - 6, y - text_h - 8),
        (x + text_w + 6, y + baseline + 6),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

# =========================================
# MAIN PROGRAM
# =========================================
with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True
    ) as pose, \
     mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        pose_detected = pose_res.pose_landmarks is not None
        hand_detected = hand_res.multi_hand_landmarks is not None

        shoulder     = state["shoulder"]
        elbow        = state["elbow"]
        wristTilt    = state["wristTilt"]
        wristRotate  = state["wristRotate"]
        gripper      = state["gripper"]

        gesture_name = "NONE"
        hand_lm = None
        hand_wrist_px = None

        # =====================================
        # GET HAND FIRST
        # This helps the program choose left or right arm
        # =====================================
        if hand_detected:
            hand_lm = hand_res.multi_hand_landmarks[0].landmark
            hand_wrist_px = (
                int(hand_lm[0].x * w),
                int(hand_lm[0].y * h)
            )

            for p in hand_lm:
                cv2.circle(
                    frame,
                    (int(p.x * w), int(p.y * h)),
                    2,
                    (255, 0, 0),
                    -1
                )

            features = extract_features(hand_lm)
            gesture_name = model.predict([features])[0]

            update_active_command(gesture_name)

        # =====================================
        # AUTO LEFT OR RIGHT ARM TRACKING
        # =====================================
        if pose_detected:
            lm = pose_res.pose_landmarks.landmark

            tracked_arm, SHO, ELB, WRI = choose_best_arm(
                lm,
                w,
                h,
                hand_wrist_px
            )

            if tracked_arm is not None:
                upper_arm = vec2(SHO, ELB)
                forearm   = vec2(ELB, WRI)
                vertical  = np.array([0, -1], dtype=np.float32)

                shoulder = clamp_joint(
                    "shoulder",
                    int(angle_between(upper_arm, vertical) * 1.2)
                )

                elbow = clamp_joint(
                    "elbow",
                    int(angle_between(upper_arm, forearm))
                )

                wristTilt = clamp_joint(
                    "wristTilt",
                    int(90 + float(forearm[1]) * 0.25)
                )

                # Draw selected arm in green
                cv2.circle(frame, SHO, 7, (0, 255, 0), -1)
                cv2.circle(frame, ELB, 7, (0, 255, 0), -1)
                cv2.circle(frame, WRI, 7, (0, 255, 0), -1)

                cv2.line(frame, SHO, ELB, (0, 255, 0), 3)
                cv2.line(frame, ELB, WRI, (0, 255, 0), 3)

            else:
                shoulder     = state["shoulder"]
                elbow        = state["elbow"]
                wristTilt    = state["wristTilt"]
                wristRotate  = state["wristRotate"]
                gripper      = state["gripper"]

        else:
            shoulder     = state["shoulder"]
            elbow        = state["elbow"]
            wristTilt    = state["wristTilt"]
            wristRotate  = state["wristRotate"]
            gripper      = state["gripper"]

        # =====================================
        # HAND-BASED WRIST ROTATION AND GRIPPER
        # =====================================
        if hand_lm is not None:
            roll = palm_roll(hand_lm)

            if neutral_roll is None:
                neutral_roll = roll

            d = wrap_pi(roll - neutral_roll)
            d_deg = (d * 180.0 / np.pi) * WRIST_ROT_GAIN

            wristRotate = clamp_joint(
                "wristRotate",
                90 + int(d_deg)
            )

            tx, ty = hand_lm[4].x * w, hand_lm[4].y * h
            ix, iy = hand_lm[8].x * w, hand_lm[8].y * h

            dist = math.hypot(ix - tx, iy - ty)

            gripper = clamp_joint(
                "gripper",
                int(np.interp(dist, [20, 120], [30, 120]))
            )

        # =====================================
        # APPLY ACTIVE GESTURE COMMAND
        # =====================================
        shoulder, elbow, wristTilt, wristRotate, gripper = apply_gesture_command(
            active_command,
            shoulder,
            elbow,
            wristTilt,
            wristRotate,
            gripper
        )

        # =====================================
        # APPLY JOINT LIMITS
        # =====================================
        shoulder    = clamp_joint("shoulder", shoulder)
        elbow       = clamp_joint("elbow", elbow)
        wristTilt   = clamp_joint("wristTilt", wristTilt)
        wristRotate = clamp_joint("wristRotate", wristRotate)
        gripper     = clamp_joint("gripper", gripper)

        # =====================================
        # SMOOTHING
        # =====================================
        state["shoulder"]    = smooth(state["shoulder"], shoulder)
        state["elbow"]       = smooth(state["elbow"], elbow)
        state["wristTilt"]   = smooth(state["wristTilt"], wristTilt)
        state["wristRotate"] = smooth(state["wristRotate"], wristRotate)
        state["gripper"]     = smooth(state["gripper"], gripper)

        # =====================================
        # SEND DATA TO ARDUINO
        # Format:
        # shoulder servo 1,
        # shoulder servo 2,
        # elbow,
        # wrist tilt,
        # wrist rotate,
        # gripper
        # =====================================
        cmd = (
            f"{state['shoulder']},"
            f"{state['shoulder']},"
            f"{state['elbow']},"
            f"{state['wristTilt']},"
            f"{state['wristRotate']},"
            f"{state['gripper']}\n"
        )

        arduino.write(cmd.encode())
        arduino.flush()

        # =====================================
        # DISPLAY INFO
        # =====================================
        draw_black_text(
            frame,
            f"Predicted: {gesture_name}",
            (20, 50),
            0.75,
            2
        )

        draw_black_text(
            frame,
            f"Active Command: {active_command}",
            (20, 90),
            0.75,
            2
        )

        draw_black_text(
            frame,
            f"S:{state['shoulder']} E:{state['elbow']} WT:{state['wristTilt']} WR:{state['wristRotate']} G:{state['gripper']}",
            (20, 130),
            0.65,
            2
        )

        cv2.imshow("ARM CONTROL WITH AUTO LEFT/RIGHT ARM TRACKING", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        # Recalibrate palm neutral roll
        if key in (ord('c'), ord('C')) and hand_lm is not None:
            neutral_roll = palm_roll(hand_lm)

cap.release()
arduino.close()
cv2.destroyAllWindows()
