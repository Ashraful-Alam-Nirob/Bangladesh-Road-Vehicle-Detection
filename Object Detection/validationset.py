import os
import shutil
import random

def create_validation_set(train_img_dir, train_label_dir, val_size=0.2):
    # Ensure the existence of the validation directories
    val_img_dir = train_img_dir.replace('/train', '/val')
    val_label_dir = train_label_dir.replace('/train', '/val')

    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)

    # List all files in the training image directory
    images = [f for f in os.listdir(train_img_dir) if os.path.isfile(os.path.join(train_img_dir, f))]
    total_images = len(images)
    val_count = int(total_images * val_size)

    # Randomly select a subset of images for validation
    val_images = random.sample(images, val_count)

    # Move selected images and their corresponding labels to the validation directories
    for img in val_images:
        shutil.move(os.path.join(train_img_dir, img), os.path.join(val_img_dir, img))

        label = img.rsplit('.', 1)[0] + '.txt'  # Replace image file extension with .txt for label
        shutil.move(os.path.join(train_label_dir, label), os.path.join(val_label_dir, label))

    print(f"Moved {val_count} images and their labels to the validation set.")

# Paths to your training image and label directories
train_img_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/images/train'
train_label_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/labels/train'

create_validation_set(train_img_dir, train_label_dir, val_size=0.30)

