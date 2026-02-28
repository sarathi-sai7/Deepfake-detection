import os
import shutil
import random

source_dir = "data"
target_dir = "dataset"

train_ratio = 0.8

classes = ["Real", "DeepFake"]

for cls in classes:

    os.makedirs(f"{target_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{target_dir}/validation/{cls}", exist_ok=True)

    files = os.listdir(f"{source_dir}/{cls}")

    random.shuffle(files)

    split = int(len(files) * train_ratio)

    train_files = files[:split]
    val_files = files[split:]

    for file in train_files:
        shutil.copy(
            f"{source_dir}/{cls}/{file}",
            f"{target_dir}/train/{cls}/{file}"
        )

    for file in val_files:
        shutil.copy(
            f"{source_dir}/{cls}/{file}",
            f"{target_dir}/validation/{cls}/{file}"
        )

print("Dataset prepared successfully")