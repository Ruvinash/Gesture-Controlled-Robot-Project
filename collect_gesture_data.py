import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

DATA_FILE = "gesture_data.csv"

SAMPLES_PER_GESTURE = 100

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

file_exists = os.path.exists(DATA_FILE)

with open(DATA_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        current_text = "Press 0-9 to collect 100 samples"

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_visible = result.multi_hand_landmarks is not None

            if hand_visible:
                hand_lm = result.multi_hand_landmarks[0].landmark

                for p in hand_lm:
                    cv2.circle(
                        frame,
                        (int(p.x * w), int(p.y * h)),
                        2,
                        (255, 0, 0),
                        -1
                    )

            cv2.putText(
                frame,
                current_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "0 Open | 1 Fist | 2 Point | 3 Peace | 4 Three",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "5 Rock | 6 Pinky | 7 Thumbs Up | 8 OK | 9 Spider | Q Quit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in GESTURES:
                label = GESTURES[key]
                count = 0

                print(f"Collecting {SAMPLES_PER_GESTURE} samples for {label}")
                print("Move your hand slightly while keeping the same gesture.")

                while count < SAMPLES_PER_GESTURE:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        hand_lm = result.multi_hand_landmarks[0].landmark

                        features = extract_features(hand_lm)
                        writer.writerow(features + [label])
                        count += 1

                        for p in hand_lm:
                            cv2.circle(
                                frame,
                                (int(p.x * w), int(p.y * h)),
                                2,
                                (255, 0, 0),
                                -1
                            )

                    cv2.putText(
                        frame,
                        f"Collecting {label}: {count}/{SAMPLES_PER_GESTURE}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "Move hand slowly: left, right, near, far, tilted",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow("Collect Gesture Data", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    time.sleep(0.03)

                current_text = f"Saved {count} samples for {label}"
                print(current_text)

cap.release()
cv2.destroyAllWindows()
