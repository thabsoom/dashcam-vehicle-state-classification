import os
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\farah\Downloads\labelled_data.csv"
BASE_PATH = r"C:\Users\farah\Downloads\DS203-2025-S2-E5-Project-Data"

IMG_SIZE = (64, 64)
BLOCK_SIZE = 16
SEARCH_RANGE = 4
TEST_SIZE = 0.25
RANDOM_STATE = 42

VALID_CLASSES = ["AHEAD", "HALTED", "SHARP_LEFT", "SHARP_RIGHT"]


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(path: str, img_size=(64, 64)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    return img.astype(np.float32)


# =========================
# MANUAL BLOCK MATCHING
# =========================
def block_motion_vectors(prev, curr, block_size=16, search_range=4):
    h, w = prev.shape
    all_dx, all_dy = [], []
    left_dx, right_dx = [], []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = prev[y:y + block_size, x:x + block_size]

            best_dx, best_dy = 0, 0
            best_score = float("inf")

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ny, nx = y + dy, x + dx

                    if ny < 0 or nx < 0 or ny + block_size > h or nx + block_size > w:
                        continue

                    candidate = curr[ny:ny + block_size, nx:nx + block_size]
                    score = np.mean(np.abs(block - candidate))

                    if score < best_score:
                        best_score = score
                        best_dx, best_dy = dx, dy

            all_dx.append(best_dx)
            all_dy.append(best_dy)

            if x < w // 2:
                left_dx.append(best_dx)
            else:
                right_dx.append(best_dx)

    mean_dx = float(np.mean(all_dx)) if all_dx else 0.0
    mean_dy = float(np.mean(all_dy)) if all_dy else 0.0
    mean_left_dx = float(np.mean(left_dx)) if left_dx else 0.0
    mean_right_dx = float(np.mean(right_dx)) if right_dx else 0.0
    dx_asymmetry = mean_left_dx - mean_right_dx

    return mean_dx, mean_dy, mean_left_dx, mean_right_dx, dx_asymmetry


# =========================
# OTHER FEATURES
# =========================
def gradient_features(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    grad_x = float(np.mean(np.abs(sobelx)))
    grad_y = float(np.mean(np.abs(sobely)))
    grad_ratio = grad_x / (grad_y + 1e-6)

    return grad_x, grad_y, grad_ratio


def blur_metric(img):
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    return float(cv2.Laplacian(img_u8, cv2.CV_64F).var())


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(curr, prev):
    h, w = curr.shape

    left = curr[:, :w // 3]
    center = curr[:, w // 3:2 * w // 3]
    right = curr[:, 2 * w // 3:]
    bottom = curr[h // 2:, :]

    mean_intensity = float(np.mean(curr))
    std_intensity = float(np.std(curr))
    bottom_mean = float(np.mean(bottom))

    grad_x, grad_y, grad_ratio = gradient_features(curr)
    blur = blur_metric(curr)

    if prev is None:
        return [
            mean_intensity, std_intensity, bottom_mean,
            grad_x, grad_y, grad_ratio, blur,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        ]

    left_p = prev[:, :w // 3]
    center_p = prev[:, w // 3:2 * w // 3]
    right_p = prev[:, 2 * w // 3:]

    full_diff = float(np.mean(np.abs(curr - prev)))
    left_diff = float(np.mean(np.abs(left - left_p)))
    right_diff = float(np.mean(np.abs(right - right_p)))
    lr_balance = left_diff - right_diff

    signed_diff = float(np.mean(curr - prev))
    center_signed = float(np.mean(center - center_p))

    mean_dx, mean_dy, mean_left_dx, mean_right_dx, dx_asymmetry = block_motion_vectors(
        prev, curr, block_size=BLOCK_SIZE, search_range=SEARCH_RANGE
    )

    return [
        mean_intensity, std_intensity, bottom_mean,
        grad_x, grad_y, grad_ratio, blur,
        full_diff, left_diff, right_diff, lr_balance,
        signed_diff, center_signed,
        mean_dx, mean_dy, mean_left_dx, mean_right_dx, dx_asymmetry
    ]


FEATURES = [
    "mean_intensity", "std_intensity", "bottom_mean",
    "grad_x", "grad_y", "grad_ratio", "blur",
    "full_diff", "left_diff", "right_diff", "lr_balance",
    "signed_diff", "center_signed",
    "mean_dx", "mean_dy", "left_dx", "right_dx", "dx_asymmetry"
]


# =========================
# LOAD LABELS
# =========================
print("Loading labels...")

df = pd.read_csv(CSV_PATH, keep_default_na=False)
df["folder_name"] = df["folder_name"].astype(str).str.zfill(4)
df["frame_name"] = df["frame_name"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.upper()

df = df[df["label"].isin(VALID_CLASSES)].copy()
df = df.sort_values(["folder_name", "frame_name"]).reset_index(drop=True)

print("\nLabel distribution in CSV:")
print(df["label"].value_counts())
print(f"\nTotal labeled frames: {len(df)}")


# =========================
# BUILD FEATURE TABLE
# =========================
print("\nExtracting features...")

rows = []
prev_img = None
prev_folder = None

for _, r in df.iterrows():
    folder = r["folder_name"]
    frame = r["frame_name"]
    label = r["label"]

    path = os.path.join(BASE_PATH, folder, frame)

    if folder != prev_folder:
        prev_img = None

    img = preprocess_image(path)
    if img is None:
        print(f"Warning: Could not read {path}")
        prev_folder = folder
        continue

    feats = extract_features(img, prev_img)

    rows.append({
        "folder": folder,
        "frame": frame,
        "image_path": path,
        "label": label,
        **{FEATURES[i]: feats[i] for i in range(len(FEATURES))}
    })

    prev_img = img
    prev_folder = folder

feat_df = pd.DataFrame(rows)

print(f"\nFeature matrix shape: {feat_df.shape}")
print("\nLabel distribution after feature extraction:")
print(feat_df["label"].value_counts())


# =========================
# BALANCE TRAINING DATA ONLY
# =========================
def balance_classes(train_df):
    counts = train_df["label"].value_counts()
    target = counts.max()

    balanced_parts = []

    for cls in counts.index:
        cls_df = train_df[train_df["label"] == cls]
        if len(cls_df) < target:
            cls_df = resample(
                cls_df,
                replace=True,
                n_samples=target,
                random_state=42
            )
        balanced_parts.append(cls_df)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# FIND A SPLIT WITH ALL 4 CLASSES IN TRAIN AND TEST
# =========================
def split_with_all_classes(feat_df, test_size=0.25, base_random_state=42, max_tries=500):
    folders = feat_df["folder"].unique()

    for i in range(max_tries):
        rs = base_random_state + i

        train_folders, test_folders = train_test_split(
            folders,
            test_size=test_size,
            random_state=rs
        )

        train_df = feat_df[feat_df["folder"].isin(train_folders)]
        test_df = feat_df[feat_df["folder"].isin(test_folders)]

        train_classes = set(train_df["label"].unique())
        test_classes = set(test_df["label"].unique())

        if set(VALID_CLASSES).issubset(train_classes) and set(VALID_CLASSES).issubset(test_classes):
            return train_df, test_df, rs

    return None, None, None


train_df, test_df, used_random_state = split_with_all_classes(
    feat_df,
    test_size=TEST_SIZE,
    base_random_state=RANDOM_STATE
)

if train_df is None:
    raise ValueError(
        "Could not find a folder-level split where all 4 classes appear in both train and test. "
        "This usually means some class exists only in too few folders."
    )

print("\nUsing final split with random state:", used_random_state)
print("\nTrain label counts:")
print(train_df["label"].value_counts())
print("\nTest label counts:")
print(test_df["label"].value_counts())


# =========================
# SAVE TRAIN / TEST CSVs USED
# =========================
train_df.to_csv("train_used.csv", index=False)
test_df.to_csv("test_used.csv", index=False)
print("\nSaved train_used.csv and test_used.csv")


# =========================
# PREPARE TRAIN / TEST DATA
# =========================
train_balanced = balance_classes(train_df)

X_train = train_balanced[FEATURES].values
y_train = train_balanced["label"].values

X_test = test_df[FEATURES].values
y_test = test_df["label"].values


# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0
        ))
    ]),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            class_weight="balanced",
            kernel="rbf",
            C=10,
            gamma="scale"
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
}


# =========================
# TRAIN + COMPARE
# =========================
print("\n" + "=" * 60)
print("TRAINING AND EVALUATION")
print("=" * 60)

all_results = []
best_model = None
best_name = None
best_acc = -1
best_preds = None

for name, model in models.items():
    model.fit(X_train, y_train)

    # ===== TRAIN METRICS =====
    train_preds = model.predict(X_train)

    train_acc = accuracy_score(y_train, train_preds)
    train_prec = precision_score(y_train, train_preds, average="weighted", zero_division=0)
    train_rec = recall_score(y_train, train_preds, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, train_preds, average="weighted", zero_division=0)

    # ===== TEST METRICS =====
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    all_results.append({
        "model": name,
        "train_accuracy": train_acc,
        "train_precision": train_prec,
        "train_recall": train_rec,
        "train_f1_score": train_f1,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1_score": f1
    })

    print(f"\n{name}")
    print("TRAIN:")
    print(f"Accuracy : {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall   : {train_rec:.4f}")
    print(f"F1-score : {train_f1:.4f}")

    print("TEST:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name
        best_preds = preds

results_df = pd.DataFrame(all_results)
results_df.to_csv("model_comparison.csv", index=False)
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df)

print(f"\nBest model: {best_name}")
print("\nClassification report:")
print(classification_report(y_test, best_preds, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, best_preds))


# =========================
# SAVE TEST PREDICTIONS WITH FEATURES
# =========================
test_output = test_df.copy()
test_output["predicted_label"] = best_preds
test_output.to_csv("test_predictions_with_features.csv", index=False)

print("\nSaved test_predictions_with_features.csv")


# =========================
# RETRAIN BEST MODEL ON FULL LABELED DATA
# =========================
print("\nRetraining best model on full labeled data...")

full_balanced = balance_classes(feat_df)
X_full = full_balanced[FEATURES].values
y_full = full_balanced["label"].values

# rebuild same model family cleanly
if best_name == "Logistic Regression":
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0
        ))
    ])
elif best_name == "SVC":
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            class_weight="balanced",
            kernel="rbf",
            C=10,
            gamma="scale"
        ))
    ])
elif best_name == "Random Forest":
    final_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
else:
    final_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )

final_model.fit(X_full, y_full)
joblib.dump(final_model, "best_model.pkl")

print("Saved best_model.pkl")


# =========================
# OPTIONAL INFERENCE ON ALL FOLDERS
# =========================
def predict_video_folder(folder_path, model, img_size=IMG_SIZE):
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not files:
        return None, None

    rows = []
    prev_img = None

    for fname in files:
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, img_size).astype(np.float32)
        feats = extract_features(img, prev_img)

        pred = model.predict(np.array(feats).reshape(1, -1))[0]

        rows.append({
            "image_no": fname,
            "vehicle_state": pred
        })

        prev_img = img

    per_frame_df = pd.DataFrame(rows)

    if len(per_frame_df) == 0:
        return per_frame_df, pd.DataFrame()

    segments = []
    current_state = per_frame_df.iloc[0]["vehicle_state"]
    start_frame = per_frame_df.iloc[0]["image_no"]

    for i in range(1, len(per_frame_df)):
        state = per_frame_df.iloc[i]["vehicle_state"]
        if state != current_state:
            segments.append({
                "from_image_no": start_frame,
                "to_image_no": per_frame_df.iloc[i - 1]["image_no"],
                "vehicle_state": current_state
            })
            current_state = state
            start_frame = per_frame_df.iloc[i]["image_no"]

    segments.append({
        "from_image_no": start_frame,
        "to_image_no": per_frame_df.iloc[-1]["image_no"],
        "vehicle_state": current_state
    })

    return per_frame_df, pd.DataFrame(segments)


print("\nDone.")