import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

# =========================================
# SETTINGS
# =========================================
RESULTS_FILE = "live_gesture_results.csv"

PRECISION_RECALL_F1_IMAGE = "precision_recall_f1_bar_chart.png"
PER_GESTURE_ACCURACY_IMAGE = "per_gesture_accuracy_bar_chart.png"
AVERAGE_CONFIDENCE_IMAGE = "average_confidence_per_gesture_chart.png"

SUMMARY_CSV = "gesture_chart_summary.csv"

GESTURE_ORDER = [
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
# LOAD LIVE RESULTS
# =========================================
df = pd.read_csv(RESULTS_FILE)

required_columns = ["actual_gesture", "predicted_gesture", "confidence"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV file: {col}")

print("Live gesture results loaded successfully.")
print("Total samples:", len(df))

# Keep only known gestures
df = df[df["actual_gesture"].isin(GESTURE_ORDER)]
df = df[df["predicted_gesture"].isin(GESTURE_ORDER)]

print("Samples after filtering:", len(df))

# =========================================
# 1. PRECISION / RECALL / F1-SCORE DATA
# =========================================
report_dict = classification_report(
    df["actual_gesture"],
    df["predicted_gesture"],
    labels=GESTURE_ORDER,
    output_dict=True,
    zero_division=0
)

metrics_data = []

for gesture in GESTURE_ORDER:
    metrics_data.append({
        "gesture": gesture,
        "precision": report_dict[gesture]["precision"] * 100,
        "recall": report_dict[gesture]["recall"] * 100,
        "f1_score": report_dict[gesture]["f1-score"] * 100,
        "support": report_dict[gesture]["support"]
    })

metrics_df = pd.DataFrame(metrics_data)

# =========================================
# 2. PER-GESTURE ACCURACY DATA
# =========================================
accuracy_data = []

for gesture in GESTURE_ORDER:
    gesture_df = df[df["actual_gesture"] == gesture]

    if len(gesture_df) == 0:
        accuracy = 0
        total_samples = 0
        correct_samples = 0
    else:
        correct_samples = (gesture_df["actual_gesture"] == gesture_df["predicted_gesture"]).sum()
        total_samples = len(gesture_df)
        accuracy = (correct_samples / total_samples) * 100

    accuracy_data.append({
        "gesture": gesture,
        "correct_samples": correct_samples,
        "total_samples": total_samples,
        "accuracy": accuracy
    })

accuracy_df = pd.DataFrame(accuracy_data)

# =========================================
# 3. AVERAGE CONFIDENCE DATA
# =========================================
confidence_df = (
    df.groupby("actual_gesture")["confidence"]
    .mean()
    .reindex(GESTURE_ORDER)
    .reset_index()
)

confidence_df.columns = ["gesture", "average_confidence"]
confidence_df["average_confidence"] = confidence_df["average_confidence"].fillna(0)

# =========================================
# SAVE SUMMARY CSV
# =========================================
summary_df = metrics_df.merge(
    accuracy_df,
    on="gesture",
    how="left"
).merge(
    confidence_df,
    on="gesture",
    how="left"
)

summary_df.to_csv(SUMMARY_CSV, index=False)

print("Saved summary CSV:", SUMMARY_CSV)

# =========================================
# CHART 1: PRECISION / RECALL / F1-SCORE
# =========================================
x = np.arange(len(GESTURE_ORDER))
bar_width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(
    x - bar_width,
    metrics_df["precision"],
    width=bar_width,
    label="Precision"
)

plt.bar(
    x,
    metrics_df["recall"],
    width=bar_width,
    label="Recall"
)

plt.bar(
    x + bar_width,
    metrics_df["f1_score"],
    width=bar_width,
    label="F1-score"
)

plt.title("Precision, Recall and F1-Score per Gesture")
plt.xlabel("Gesture")
plt.ylabel("Score (%)")
plt.xticks(x, GESTURE_ORDER, rotation=45, ha="right")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PRECISION_RECALL_F1_IMAGE, dpi=300)
plt.close()

print("Saved:", PRECISION_RECALL_F1_IMAGE)

# =========================================
# CHART 2: PER-GESTURE ACCURACY
# =========================================
plt.figure(figsize=(12, 7))

plt.bar(
    accuracy_df["gesture"],
    accuracy_df["accuracy"]
)

plt.title("Per-Gesture Accuracy")
plt.xlabel("Gesture")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show accuracy values above bars
for i, value in enumerate(accuracy_df["accuracy"]):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(PER_GESTURE_ACCURACY_IMAGE, dpi=300)
plt.close()

print("Saved:", PER_GESTURE_ACCURACY_IMAGE)

# =========================================
# CHART 3: AVERAGE CONFIDENCE PER GESTURE
# =========================================
plt.figure(figsize=(12, 7))

plt.bar(
    confidence_df["gesture"],
    confidence_df["average_confidence"]
)

plt.title("Average Confidence per Gesture")
plt.xlabel("Gesture")
plt.ylabel("Average Confidence (%)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show confidence values above bars
for i, value in enumerate(confidence_df["average_confidence"]):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(AVERAGE_CONFIDENCE_IMAGE, dpi=300)
plt.close()

print("Saved:", AVERAGE_CONFIDENCE_IMAGE)

# =========================================
# PRINT RESULTS IN TERMINAL
# =========================================
print("\n===================================")
print("GESTURE RESULT CHARTS CREATED")
print("===================================")
print("1.", PRECISION_RECALL_F1_IMAGE)
print("2.", PER_GESTURE_ACCURACY_IMAGE)
print("3.", AVERAGE_CONFIDENCE_IMAGE)
print("4.", SUMMARY_CSV)

print("\nSummary:")
print(summary_df)
