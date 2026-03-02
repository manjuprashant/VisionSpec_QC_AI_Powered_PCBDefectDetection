import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

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
    print("Loading:", name)
    models[name] = tf.keras.models.load_model(path)

print("Loaded models:", list(models.keys()))

# =========================
# EVALUATE FUNCTION
# =========================

def evaluate(data, dataset_name):

    results = []

    y_true = data.classes

    roc_plot = plt.figure()
    pr_plot = plt.figure()

    confidence_all = []
    labels = []

    for model_name, model in models.items():

        print(f"Evaluating {model_name} on {dataset_name}")

        probs = model.predict(data, verbose=0).flatten()

        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        roc_auc = roc_auc_score(y_true, probs)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall_curve, precision_curve)

        # save metrics
        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mean_confidence": np.mean(probs)
        })

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, probs)

        plt.figure(roc_plot.number)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.2f})")

        # PR curve
        plt.figure(pr_plot.number)
        plt.plot(recall_curve, precision_curve,
                 label=f"{model_name} (AUC={pr_auc:.2f})")

        confidence_all.append(probs)
        labels.append(model_name)

    # finalize ROC plot
    plt.figure(roc_plot.number)
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve ({dataset_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve_{dataset_name}.png")
    plt.close()

    # finalize PR plot
    plt.figure(pr_plot.number)
    plt.title(f"Precision-Recall Curve ({dataset_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/precision_recall_{dataset_name}.png")
    plt.close()

    # confidence boxplot
    plt.figure()
    plt.boxplot(confidence_all, tick_labels=labels)
    plt.title(f"Confidence Distribution ({dataset_name})")
    plt.savefig(f"{OUTPUT_DIR}/confidence_boxplot_{dataset_name}.png")
    plt.close()

    return results


# =========================
# RUN
# =========================

train_results = evaluate(train_data, "train")
val_results = evaluate(val_data, "val")

df = pd.DataFrame(train_results + val_results)

df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

print("Saved CSV comparison")

# =========================
# BAR CHART ACCURACY
# =========================

val_df = df[df.dataset=="val"]

plt.figure()
plt.bar(val_df.model, val_df.accuracy)
plt.title("Validation Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.savefig(f"{OUTPUT_DIR}/accuracy_bar.png")
plt.close()

print("All evaluation plots saved.")