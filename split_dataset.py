import os
import shutil
import random

original_dataset_dir = 'animal_classification'
base_dir = 'animal_dataset' 
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

split_ratio = 0.8  

if not os.path.exists(base_dir):
    os.makedirs(train_dir)
    os.makedirs(val_dir)

for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print(" Dataset split complete. New structure:")
print("├── animal_dataset")
print("    ├── train/")
print("    └── val/")
