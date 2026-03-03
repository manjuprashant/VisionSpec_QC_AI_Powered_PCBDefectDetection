import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------ Paths ------------------ #
DATASET_DIR = "dataset/val"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224

# ------------------ Load Models ------------------ #
print("Loading models...")
models = {
    "MobileNetV2": load_model(f"{MODEL_DIR}/mobilenetv2.h5"),
    "ResNet50": load_model(f"{MODEL_DIR}/resnet50.h5"),
    "EfficientNetB0": load_model(f"{MODEL_DIR}/efficientnetb0.h5"),
    "VisionSpec": load_model(f"{MODEL_DIR}/visionspec_qc.h5"),
}

# ------------------ Helper Functions ------------------ #
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def label_from_folder(folder_name):
    return 1 if "defect" in folder_name.lower() else 0

# ------------------ Inference ------------------ #
results = []

print("Running inference on dataset...")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    true_label = label_from_folder(class_name)

    for file in os.listdir(class_path):
        if not (file.endswith(".jpg") or file.endswith(".png")):
            continue

        path = os.path.join(class_path, file)
        img = load_image(path)

        row = {"image": file, "true_label": true_label}

        for name, model in models.items():
            prob = float(model.predict(img, verbose=0)[0][0])
            pred = 1 if prob > 0.5 else 0

            row[f"{name}_prob"] = prob
            row[f"{name}_pred"] = pred

        results.append(row)

# ------------------ Save Predictions CSV ------------------ #
df = pd.DataFrame(results)
csv_path = f"{RESULTS_DIR}/predictions.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

# ------------------ Evaluate Models ------------------ #
metrics_summary = []

plt.figure(figsize=(8, 6))
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.figure(figsize=(8, 6))
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")

for name in models.keys():
    y_true = df["true_label"]
    y_pred = df[f"{name}_pred"]
    y_prob = df[f"{name}_prob"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # store metrics
    metrics_summary.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC AUC": roc_auc
    })

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.savefig(f"{RESULTS_DIR}/{name}_cm.png")
    plt.close()

    # ROC curves (combined)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    # Precision-Recall curves (combined)
    plt.figure(2)
    plt.plot(recall, precision, label=f"{name}")

# save combined ROC and PR curves
plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="lower right")
plt.savefig(f"{RESULTS_DIR}/all_models_roc.png")
plt.close()

plt.figure(2)
plt.legend(loc="lower left")
plt.savefig(f"{RESULTS_DIR}/all_models_pr.png")
plt.close()

# ------------------ Save Metrics Summary ------------------ #
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)
print("Saved metrics_summary.csv")
print(metrics_df)

print("DONE")