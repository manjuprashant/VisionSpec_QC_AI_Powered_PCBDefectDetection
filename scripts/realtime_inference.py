# gradcam_batch_with_confidence.py
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import csv
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
MODELS_DIR = "models"
DATASET_DIRS = ["dataset/train", "dataset/val"]
OUTPUT_DIR = "gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load latest model
# -------------------------
model_files = glob.glob(os.path.join(MODELS_DIR, "*.h5"))
if not model_files:
    raise FileNotFoundError(f"No .h5 models found in {MODELS_DIR}")
MODEL_PATH = max(model_files, key=os.path.getmtime)
print(f"Using model: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Wrap Sequential as Functional API if needed
if isinstance(model, tf.keras.Sequential):
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    outputs = model(inputs)
    model = Model(inputs, outputs)
    print("Sequential model wrapped as Functional for Grad-CAM")

# -------------------------
# Preprocess image
# -------------------------
def preprocess_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -------------------------
# Detect last convolutional layer
# -------------------------
conv_layers = [layer.name for layer in model.layers if "conv" in layer.name]
if not conv_layers:
    raise ValueError("No convolution layers found in the model")
LAST_CONV_LAYER_NAME = conv_layers[-1]
print(f"Using last conv layer for Grad-CAM: {LAST_CONV_LAYER_NAME}")

# -------------------------
# Grad-CAM function
# -------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.convert_to_tensor(predictions)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()

# -------------------------
# Superimpose heatmap
# -------------------------
def save_gradcam_overlay(img_path, heatmap, output_path, alpha=0.4):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    cv2.imwrite(output_path, superimposed_img)

# -------------------------
# Process each dataset
# -------------------------
for dataset_dir in DATASET_DIRS:
    dataset_type = os.path.basename(dataset_dir)
    report_rows = []

    class_dirs = glob.glob(os.path.join(dataset_dir, "*"))
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        image_paths = glob.glob(os.path.join(class_dir, "*"))
        total_images = len(image_paths)

        for idx, img_path in enumerate(image_paths, 1):
            img_array = preprocess_img(img_path)
            heatmap, pred_index = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

            # Prediction confidence
            pred_probs = tf.nn.softmax(model.predict(img_array))[0]
            pred_confidence = float(pred_probs[pred_index])
            all_class_probs = ','.join([f"{p:.4f}" for p in pred_probs.numpy()])

            # Save overlay
            output_path = os.path.join(OUTPUT_DIR, dataset_type, class_name, os.path.basename(img_path))
            save_gradcam_overlay(img_path, heatmap, output_path)

            # Append row to report
            report_rows.append([
                img_path,
                int(pred_index),
                pred_confidence,
                all_class_probs,
                output_path
            ])

            # PowerShell progress update
            print(f"[{dataset_type}] Processing image {idx}/{total_images}: {os.path.basename(img_path)} -> Pred: {pred_index} (Conf: {pred_confidence:.4f})")

    # Save CSV report
    csv_file = os.path.join(OUTPUT_DIR, f"gradcam_report_{dataset_type}.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "predicted_class_index",
            "predicted_class_confidence",
            "all_class_confidences",
            "gradcam_output_path"
        ])
        writer.writerows(report_rows)

    print(f"\nGrad-CAM generation completed for {dataset_type} images!")
    print(f"CSV report saved to {csv_file}\n")