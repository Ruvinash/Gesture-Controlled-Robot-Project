import cv2
import mediapipe as mp
import joblib
import numpy as np

# =========================================
# SETTINGS
# =========================================
MODEL_FILE = "gesture_model.pkl"

# =========================================
# LOAD TRAINED ML MODEL
# =========================================
print("Loading trained model:", MODEL_FILE)

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

if hasattr(model, "classes_"):
    print("Model gesture classes:")
    print(model.classes_)

# =========================================
# MEDIAPIPE SETUP
# =========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# =========================================
# FEATURE EXTRACTION
# Must match collect_gesture_data.py
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
# DRAW BLACK TEXT WITH WHITE BACKGROUND
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

    # White background box
    cv2.rectangle(
        frame,
        (x - 6, y - text_h - 8),
        (x + text_w + 6, y + baseline + 6),
        (255, 255, 255),
        -1
    )

    # Black text
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
# MAIN WEBCAM TEST
# =========================================
print("\nStarting webcam gesture recognition...")
print("--------------------------------")
print("Using gesture_model.pkl")
print("Press Q = Quit")
print("--------------------------------")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_gesture = "NO HAND"
        confidence_text = "Confidence: N/A"

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            hand_lm = hand_landmarks.landmark

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract features
            features = extract_features(hand_lm)

            # Predict gesture using gesture_model.pkl
            predicted_gesture = model.predict([features])[0]

            # Get confidence if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([features])[0]
                confidence = np.max(probabilities) * 100
                confidence_text = f"Confidence: {confidence:.1f}%"

        # =========================================
        # DISPLAY INFORMATION
        # =========================================
        draw_black_text(
            frame,
            f"Predicted Gesture: {predicted_gesture}",
            (20, 50),
            0.8,
            2
        )

        draw_black_text(
            frame,
            confidence_text,
            (20, 90),
            0.75,
            2
        )

        draw_black_text(
            frame,
            "Press Q to quit",
            (20, h - 30),
            0.65,
            2
        )

        cv2.imshow("Gesture Recognition Using gesture_model.pkl", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

cap.release()
cv2.destroyAllWindows()
