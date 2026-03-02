import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =========================
# CONFIG
# =========================

IMG_SIZE = 224
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_DIR = "models"
OUTPUT_DIR = "evaluation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =========================
# LOAD MODELS
# =========================

model_paths = glob.glob(os.path.join(MODEL_DIR, "*.h5"))

models = {}

for path in model_paths:
    name = os.path.basename(path).replace(".h5","").lower()
    print(f"Loading model: {name}")
    models[name] = tf.keras.models.load_model(path)

print("\nModels loaded:", list(models.keys()))

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_dataset(data, dataset_name):

    results = []

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name} on {dataset_name}...")

        probs = model.predict(data, verbose=0)

        probs = probs.flatten()

        preds = (probs > 0.5).astype(int)

        y_true = data.classes

        accuracy = accuracy_score(y_true, preds)

        precision = precision_score(y_true, preds, zero_division=0)

        recall = recall_score(y_true, preds, zero_division=0)

        f1 = f1_score(y_true, preds, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_true, probs)
        except:
            roc_auc = 0

        confidence_mean = np.mean(probs)

        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "mean_confidence": confidence_mean
        })

    return results


# =========================
# RUN EVALUATION
# =========================

train_results = evaluate_dataset(train_data, "train")

val_results = evaluate_dataset(val_data, "val")

all_results = train_results + val_results

df = pd.DataFrame(all_results)

csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")

df.to_csv(csv_path, index=False)

print("\nSaved:", csv_path)


# =========================
# BAR CHART
# =========================

val_df = df[df["dataset"]=="val"]

plt.figure()

plt.bar(val_df["model"], val_df["accuracy"])

plt.title("Model Accuracy Comparison (Validation)")

plt.ylabel("Accuracy")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR,"accuracy_bar_chart.png"))

plt.close()


# =========================
# CONFIDENCE BOXPLOT
# =========================

confidence_data = []

labels = []

for model_name, model in models.items():

    probs = model.predict(val_data, verbose=0).flatten()

    confidence_data.append(probs)

    labels.append(model_name)


plt.figure()

plt.boxplot(confidence_data, tick_labels=labels)

plt.title("Confidence Distribution")

plt.ylabel("Confidence")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR,"confidence_boxplot.png"))

plt.close()


print("\nEvaluation complete.")