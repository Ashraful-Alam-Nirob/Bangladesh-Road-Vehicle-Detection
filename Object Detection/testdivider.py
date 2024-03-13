import os
import cv2

# Directories
image_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/images/val/'
label_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/labels/val/'

# New directories
new_image_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/images/val-rs/'
new_label_dir = '/media/nirob/Local Disk/dlenigma1/BadODD/labels/val-rs/'

# Create new directories if they don't exist
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)

# Iterate through the images
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):

        # Load image
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # Get original size
        h, w = image.shape[:2]

        resize_w= (w//32)*32
        resize_h=(h//32)*32
        print(resize_w,resize_h)

        # Resize image
        resized_image = cv2.resize(image, (resize_w, resize_h))

        # Save resized image
        cv2.imwrite(os.path.join(new_image_dir, filename), resized_image)

        # Corresponding label file
        label_filename = filename.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                # Adjust bounding box coordinates
                x_center = x_center * w
                y_center = y_center * h
                bbox_width = bbox_width * w
                bbox_height = bbox_height * h

                # Resize coordinates
                x_center = (x_center / w) * resize_w
                y_center = (y_center / h) * resize_h
                bbox_width = (bbox_width / w) * resize_w
                bbox_height = (bbox_height / h) * resize_h

                # Re-normalize coordinates
                x_center /= resize_w
                y_center /= resize_h
                bbox_width /= resize_w
                bbox_height /= resize_h

                new_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                new_lines.append(new_line)

            # Save adjusted label file
            with open(os.path.join(new_label_dir, label_filename), 'w') as file:
                file.writelines(new_lines)