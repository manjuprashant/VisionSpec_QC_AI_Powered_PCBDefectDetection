import os
import cv2
import numpy as np
import random

# CONFIG
IMG_SIZE = 224
TOTAL_IMAGES = 1000

TRAIN_GOOD = 400
TRAIN_DEFECT = 400
VAL_GOOD = 100
VAL_DEFECT = 100

BASE_PATH = "dataset"

# Create folders
folders = [
    "dataset/train/GOOD",
    "dataset/train/DEFECT",
    "dataset/val/GOOD",
    "dataset/val/DEFECT"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# -------------------------
# PCB GENERATION FUNCTIONS
# -------------------------

def create_clean_pcb():

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # background green PCB
    img[:] = (0, 120, 0)

    # draw traces
    for i in range(10):
        x1 = random.randint(10, IMG_SIZE-10)
        y1 = random.randint(10, IMG_SIZE-10)
        x2 = random.randint(10, IMG_SIZE-10)
        y2 = random.randint(10, IMG_SIZE-10)

        cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 2)

    # draw pads
    for i in range(15):
        x = random.randint(10, IMG_SIZE-10)
        y = random.randint(10, IMG_SIZE-10)

        cv2.circle(img, (x,y), 4, (255,255,0), -1)

    return img


def create_defect_pcb():

    img = create_clean_pcb()

    defect_type = random.choice(["break","missing","noise"])

    if defect_type == "break":
        x = random.randint(20, IMG_SIZE-20)
        y = random.randint(20, IMG_SIZE-20)

        cv2.rectangle(img, (x,y), (x+20,y+5), (0,120,0), -1)

    elif defect_type == "missing":
        x = random.randint(20, IMG_SIZE-20)
        y = random.randint(20, IMG_SIZE-20)

        cv2.circle(img, (x,y), 6, (0,120,0), -1)

    elif defect_type == "noise":

        noise = np.random.randint(0,50,(IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
        img = cv2.add(img, noise)

    return img


# -------------------------
# SAVE IMAGES
# -------------------------

def save_images(count, folder, defect=False):

    for i in range(count):

        if defect:
            img = create_defect_pcb()
        else:
            img = create_clean_pcb()

        path = os.path.join(folder, f"img_{i}.png")

        cv2.imwrite(path, img)


print("Generating dataset...")

save_images(TRAIN_GOOD, "dataset/train/GOOD", defect=False)
save_images(TRAIN_DEFECT, "dataset/train/DEFECT", defect=True)

save_images(VAL_GOOD, "dataset/val/GOOD", defect=False)
save_images(VAL_DEFECT, "dataset/val/DEFECT", defect=True)

print("Dataset generated successfully!")
print("Total images:", TOTAL_IMAGES)