import os
import numpy as np
import tensorflow as tf
import cv2
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/visionspec_qc.h5"
DATASET_PATH = "dataset"
IMG_SIZE = 224
OUTPUT_DIR = "evaluation_results"

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
GOOD_DIR = os.path.join(OUTPUT_DIR, "GOOD")
DEFECT_DIR = os.path.join(OUTPUT_DIR, "DEFECT")

os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

print("Model output shape:", model.output_shape)

# ==============================
# CLASS LABELS
# IMPORTANT: Adjust if reversed
# ==============================
class_names = ["GOOD", "DEFECT"]

# ==============================
# COLLECT DATA
# ==============================
image_paths = []
true_labels = []

for split in ["train", "val"]:
    split_path = os.path.join(DATASET_PATH, split)

    for class_name in class_names:
        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                image_paths.append(os.path.join(class_path, file))
                true_labels.append(class_name)

print(f"Total images found: {len(image_paths)}")

# ==============================
# INFERENCE
# ==============================
y_true = []
y_pred = []
results = []

for img_path, actual_label in zip(image_paths, true_labels):

    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

    except:
        print(f"Skipping invalid image: {img_path}")
        continue

    prediction = model.predict(img, verbose=0)

    # Binary sigmoid output
    if prediction.shape[-1] == 1:
        confidence = float(prediction[0][0])
        predicted_label = class_names[1] if confidence > 0.5 else class_names[0]
    else:
        confidence = float(np.max(prediction))
        predicted_label = class_names[np.argmax(prediction)]

    print(f"{os.path.basename(img_path)} -> predicted: {predicted_label} ({confidence:.4f})")

    # Save image to predicted folder
    if predicted_label == "GOOD":
        save_dir = GOOD_DIR
    else:
        save_dir = DEFECT_DIR

    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(save_dir, f"{name}_{confidence:.4f}{ext}")
    shutil.copy(img_path, save_path)

    y_true.append(actual_label)
    y_pred.append(predicted_label)

    results.append({
        "image_name": filename,
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "confidence_score": confidence
    })

# ==============================
# EVALUATION METRICS
# ==============================
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=class_names)
report = classification_report(y_true, y_pred, target_names=class_names)

print("\n==============================")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ==============================
# SAVE RESULTS
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save CSV
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "dataset_predictions.csv")
df.to_csv(csv_path, index=False)

# Save classification report
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Save confusion matrix image
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print("\nSaved files:")
print("CSV:", csv_path)
print("Confusion Matrix:", cm_path)
print("Classification Report:", report_path)
print("\nAll images saved into GOOD/ and DEFECT/ folders.")
print("\nEvaluation completed successfully.")