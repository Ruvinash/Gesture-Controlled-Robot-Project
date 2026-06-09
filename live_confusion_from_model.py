import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================
# SETTINGS
# =========================================
MODEL_FILE = "gesture_model.pkl"

SAMPLES_PER_GESTURE = 100

RESULTS_CSV_FILE = "live_gesture_results.csv"
CONFUSION_MATRIX_IMAGE = "live_confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_IMAGE = "live_normalized_confusion_matrix.png"
CLASSIFICATION_REPORT_FILE = "live_classification_report.txt"

# =========================================
# GESTURE KEYS
# =========================================
GESTURES = {
    ord('0'): "OPEN PALM",
    ord('1'): "FIST",
    ord('2'): "POINT",
    ord('3'): "PEACE",
    ord('4'): "THREE",
    ord('5'): "ROCK",
    ord('6'): "PINKY",
    ord('7'): "THUMBS UP",
    ord('8'): "OK",
    ord('9'): "SPIDER",
}

GESTURE_LABELS = [
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
# LOAD TRAINED MODEL
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
# STORAGE FOR ACTUAL AND PREDICTED RESULTS
# =========================================
actual_results = []
predicted_results = []
confidence_results = []

# =========================================
# FEATURE EXTRACTION
# Must match your training / collect_gesture_data.py
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
# PREDICT GESTURE FROM ONE HAND FRAME
# No "Can't recognize gesture" is used here
# =========================================
def predict_gesture(hand_lm):
    features = extract_features(hand_lm)

    predicted_gesture = model.predict([features])[0]
    confidence = 0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        confidence = np.max(probabilities) * 100

    return predicted_gesture, confidence

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
# DRAW CONFUSION MATRIX
# =========================================
def save_confusion_matrix(y_true, y_pred):
    labels = GESTURE_LABELS

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(12, 9))
    plt.imshow(cm)
    plt.title("Live Gesture Recognition Confusion Matrix")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("Actual Gesture")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMAGE, dpi=300)
    plt.close()

    print("Saved:", CONFUSION_MATRIX_IMAGE)

# =========================================
# DRAW NORMALIZED CONFUSION MATRIX
# =========================================
def save_normalized_confusion_matrix(y_true, y_pred):
    labels = GESTURE_LABELS

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(12, 9))
    plt.imshow(cm_normalized)
    plt.title("Live Normalized Gesture Recognition Confusion Matrix")
    plt.xlabel("Predicted Gesture")
    plt.ylabel("Actual Gesture")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    plt.savefig(NORMALIZED_CONFUSION_MATRIX_IMAGE, dpi=300)
    plt.close()

    print("Saved:", NORMALIZED_CONFUSION_MATRIX_IMAGE)

# =========================================
# SAVE CLASSIFICATION REPORT
# =========================================
def save_classification_report(y_true, y_pred, confidence_values):
    labels = GESTURE_LABELS

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    avg_confidence = np.mean(confidence_values) if len(confidence_values) > 0 else 0

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    print("\n==============================")
    print("LIVE GESTURE RECOGNITION RESULTS")
    print("==============================")
    print(f"Total live samples: {len(y_true)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision_macro * 100:.2f}%")
    print(f"Macro Recall: {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score: {f1_macro * 100:.2f}%")
    print(f"Weighted Precision: {precision_weighted * 100:.2f}%")
    print(f"Weighted Recall: {recall_weighted * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print(f"Average Confidence: {avg_confidence:.2f}%")

    print("\nClassification Report:")
    print(report)

    with open(CLASSIFICATION_REPORT_FILE, "w") as f:
        f.write("LIVE GESTURE RECOGNITION CLASSIFICATION REPORT\n")
        f.write("==============================================\n\n")

        f.write(f"Model file used: {MODEL_FILE}\n")
        f.write(f"Total live samples: {len(y_true)}\n")
        f.write(f"Samples per gesture: {SAMPLES_PER_GESTURE}\n")
        f.write(f"Average confidence: {avg_confidence:.2f}%\n\n")

        f.write("Overall Multiclass Results:\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Macro Precision: {precision_macro * 100:.2f}%\n")
        f.write(f"Macro Recall: {recall_macro * 100:.2f}%\n")
        f.write(f"Macro F1-Score: {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted Precision: {precision_weighted * 100:.2f}%\n")
        f.write(f"Weighted Recall: {recall_weighted * 100:.2f}%\n")
        f.write(f"Weighted F1-Score: {f1_weighted * 100:.2f}%\n\n")

        f.write("Classification Report:\n")
        f.write(report)

    print("Saved:", CLASSIFICATION_REPORT_FILE)

# =========================================
# MAIN PROGRAM
# =========================================
print("\nLive Gesture Evaluation Started")
print("--------------------------------")
print("Press the number key for the gesture you are ACTUALLY doing.")
print("Then hold that gesture until 100 samples are collected.")
print("--------------------------------")
print("0 = OPEN PALM")
print("1 = FIST")
print("2 = POINT")
print("3 = PEACE")
print("4 = THREE")
print("5 = ROCK")
print("6 = PINKY")
print("7 = THUMBS UP")
print("8 = OK")
print("9 = SPIDER")
print("Q = Finish and generate results")
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

        live_prediction = "NO HAND"
        live_confidence = 0

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            hand_lm = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            live_prediction, live_confidence = predict_gesture(hand_lm)

        draw_black_text(
            frame,
            f"Live Prediction: {live_prediction}",
            (20, 50),
            0.8,
            2
        )

        draw_black_text(
            frame,
            f"Confidence: {live_confidence:.1f}%",
            (20, 90),
            0.7,
            2
        )

        draw_black_text(
            frame,
            f"Collected Samples: {len(actual_results)}",
            (20, 130),
            0.7,
            2
        )

        draw_black_text(
            frame,
            "Press 0-9 to collect actual gesture samples | Q to finish",
            (20, h - 30),
            0.65,
            2
        )

        cv2.imshow("Live Gesture Evaluation Using gesture_model.pkl", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        # =========================================
        # COLLECT LIVE TEST SAMPLES
        # =========================================
        if key in GESTURES:
            actual_label = GESTURES[key]
            count = 0

            print(f"\nCollecting {SAMPLES_PER_GESTURE} live samples for: {actual_label}")
            print("Hold the correct gesture and move your hand slightly.")

            while count < SAMPLES_PER_GESTURE:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                prediction = "NO HAND"
                confidence = 0

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    hand_lm = hand_landmarks.landmark

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    prediction, confidence = predict_gesture(hand_lm)

                    actual_results.append(actual_label)
                    predicted_results.append(prediction)
                    confidence_results.append(confidence)

                    count += 1

                draw_black_text(
                    frame,
                    f"Actual Gesture: {actual_label}",
                    (20, 50),
                    0.8,
                    2
                )

                draw_black_text(
                    frame,
                    f"Predicted Gesture: {prediction}",
                    (20, 90),
                    0.8,
                    2
                )

                draw_black_text(
                    frame,
                    f"Confidence: {confidence:.1f}%",
                    (20, 130),
                    0.7,
                    2
                )

                draw_black_text(
                    frame,
                    f"Collecting: {count}/{SAMPLES_PER_GESTURE}",
                    (20, 170),
                    0.7,
                    2
                )

                draw_black_text(
                    frame,
                    "Move hand slightly: left, right, near, far, tilted",
                    (20, 210),
                    0.65,
                    2
                )

                cv2.imshow("Live Gesture Evaluation Using gesture_model.pkl", frame)

                inner_key = cv2.waitKey(1) & 0xFF

                if inner_key in (ord('q'), ord('Q')):
                    count = SAMPLES_PER_GESTURE
                    break

            print(f"Finished collecting: {actual_label}")

cap.release()
cv2.destroyAllWindows()

# =========================================
# CREATE RESULTS AFTER TESTING
# =========================================
if len(actual_results) == 0:
    print("\nNo live test samples collected.")
    print("Run the program again and press 0-9 to collect gesture samples.")
else:
    results_df = pd.DataFrame({
        "actual_gesture": actual_results,
        "predicted_gesture": predicted_results,
        "confidence": confidence_results
    })

    results_df.to_csv(RESULTS_CSV_FILE, index=False)

    print("\nSaved:", RESULTS_CSV_FILE)

    print("\nSamples collected per actual gesture:")
    print(results_df["actual_gesture"].value_counts())

    print("\nPredicted gesture counts:")
    print(results_df["predicted_gesture"].value_counts())

    save_confusion_matrix(actual_results, predicted_results)
    save_normalized_confusion_matrix(actual_results, predicted_results)
    save_classification_report(
        actual_results,
        predicted_results,
        confidence_results
    )

    print("\n===================================")
    print("LIVE RESULT FILES CREATED")
    print("===================================")
    print("1.", RESULTS_CSV_FILE)
    print("2.", CONFUSION_MATRIX_IMAGE)
    print("3.", NORMALIZED_CONFUSION_MATRIX_IMAGE)
    print("4.", CLASSIFICATION_REPORT_FILE)
