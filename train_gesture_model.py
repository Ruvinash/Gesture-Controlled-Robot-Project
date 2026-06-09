import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

df = pd.read_csv(DATA_FILE)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model accuracy:", acc)

joblib.dump(model, MODEL_FILE)

print("Saved model as:", MODEL_FILE)
