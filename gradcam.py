import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D


# =========================
# CONFIG
# =========================

IMG_SIZE = (224,224)
DATASET_DIR = "dataset"
MODEL_DIR = "models"
RESULT_DIR = "results/gradcam"

os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    os.path.join(DATASET_DIR,"val"),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=True
)

CLASS_NAMES = list(val_gen.class_indices.keys())

print("Classes:", CLASS_NAMES)


# =========================
# FIND LAST CONV
# =========================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, Conv2D):
            return layer.name

        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, Conv2D):
                    return sub.name

    return None


# =========================
# FORCE BUILD MODEL
# =========================

def build_model(model):

    dummy = tf.zeros((1,224,224,3))
    model(dummy)

    return model


# =========================
# UNWRAP NESTED MODEL
# =========================

def unwrap(model):

    if len(model.layers)==1 and hasattr(model.layers[0],"layers"):
        return model.layers[0]

    return model


# =========================
# GRADCAM
# =========================

def gradcam(model, img, layer_name):

    model = unwrap(model)
    model = build_model(model)

    conv_layer = model.get_layer(layer_name)

    grad_model = Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor)

        class_idx = tf.argmax(predictions[0])

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), int(class_idx), float(tf.reduce_max(predictions))


# =========================
# GET SAMPLE IMAGE
# =========================

sample_img, _ = next(val_gen)


# =========================
# RUN FOR EACH MODEL
# =========================

for file in os.listdir(MODEL_DIR):

    if not file.endswith(".h5"):
        continue

    name = file.replace(".h5","")

    print("\nProcessing:", name)

    model = load_model(
        os.path.join(MODEL_DIR,file),
        compile=False
    )

    model = unwrap(model)
    model = build_model(model)

    layer = find_last_conv_layer(model)

    if layer is None:

        print("No conv layer found")
        continue

    print("Using layer:", layer)


    try:

        heatmap, pred, conf = gradcam(
            model,
            sample_img,
            layer
        )

        heatmap = cv2.resize(heatmap, IMG_SIZE)

        heatmap = np.uint8(255*heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        original = (sample_img[0]*255).astype(np.uint8)

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        save_path = os.path.join(
            RESULT_DIR,
            f"{name}_gradcam_{CLASS_NAMES[pred]}_{conf:.2f}.png"
        )

        cv2.imwrite(save_path, overlay)

        print("Saved:", save_path)

    except Exception as e:

        print("GradCAM failed:", e)


print("\nAll GradCAM images saved.")