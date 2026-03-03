import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/val"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# =========================
# MODEL BUILDERS
# =========================
def build_model(base_model_class, name):
    print(f"\nBuilding {name}...")
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_custom_cnn(name="CustomCNN"):
    print(f"\nBuilding {name}...")
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# TRAIN FUNCTION
# =========================
def train_and_save(model, name):
    print(f"\nTraining {name}...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )
    path = f"{MODEL_DIR}/{name}.h5"
    model.save(path)
    print(f"{name} saved:", path)
    return max(history.history["val_accuracy"])

# =========================
# TRAIN ALL MODELS
# =========================
results = {}

# Pretrained backbones
mobilenet = build_model(MobileNetV2, "MobileNetV2")
results["MobileNetV2"] = train_and_save(mobilenet, "mobilenetv2")

resnet = build_model(ResNet50, "ResNet50")
results["ResNet50"] = train_and_save(resnet, "resnet50")

efficientnet = build_model(EfficientNetB0, "EfficientNetB0")
results["EfficientNetB0"] = train_and_save(efficientnet, "efficientnetb0")

# Custom CNN
custom_cnn = build_custom_cnn()
results["CustomCNN"] = train_and_save(custom_cnn, "customcnn")

# VisionSpec QC model (optional: can initialize with same CNN structure)
visionspec_qc = build_custom_cnn("VisionSpecQC")
results["VisionSpecQC"] = train_and_save(visionspec_qc, "visionspec_qc")

# =========================
# BEST MODEL
# =========================
best = max(results, key=results.get)

print("\n=======================")
print("RESULTS")
for model_name, val_acc in results.items():
    print(f"{model_name}: {val_acc:.4f}")
print("BEST MODEL:", best)
print("=======================")