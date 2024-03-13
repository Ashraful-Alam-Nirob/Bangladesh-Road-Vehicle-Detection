import os
import albumentations as A
import os
from tqdm import tqdm
from skimage import io

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.2, border_mode=0),
    A.CLAHE(clip_limit=2, p=0.2),
    A.OpticalDistortion(p=0.1),
    A.GridDistortion(p=0.1),
    A.Cutout(num_holes=8, max_h_size = 30, max_w_size = 30, fill_value=0, always_apply=False, p = 0.5)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1,min_area=1024, label_fields=['class_labels']))

# Assuming these are the directories you provided
img_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/images/cpypst"
lbl_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/labels/cpypst"
aug_img_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/images/aug"
aug_lbl_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/labels/aug"

classes = ['auto_rickshaw',"bicycle","bus", "car","cart_vehicle","construction_vehicle","motorbike","person","priority_vehicle","three_wheeler","train","truck","wheelchair"]

# Calculate total and processed files for accurate progress approximation
total_files = len(os.listdir(lbl_dir))
processed_files = total_files * 2 + int(total_files * 0.3)  # 2 full iterations + 30% of the third

# Collect all original file names (without extension)
original_files = [f.split(".")[0] for f in os.listdir(lbl_dir)]

# Generate expected augmented file names based on iterations completed
expected_augmented_files = []
for i in range(3):  # 3 iterations, with the third being partially completed
    for original_file in original_files:
        expected_augmented_files.append(f"{original_file}{i}.txt")

# Now find out which expected files are actually missing
missing_files = [f for f in expected_augmented_files if not os.path.exists(os.path.join(aug_lbl_dir, f))]


aug_img_dir = "/home/nirob/Documents/aug_img"
aug_lbl_dir = "/home/nirob/Documents/aug_lbl"
for name in tqdm(missing_files):

    nm = name[0:-5]
    img = io.imread(os.path.join(img_dir, nm + ".jpg"))
    bounding_boxes = []
    img_cls = []
    with open(os.path.join(lbl_dir, nm+'.txt'), 'r') as f:
        for line in f:
            x = int(line.split()[0])
            bounding_boxes.append(list(map(float, line.split()[1:])))
            img_cls.append(classes[x])

    transformed = transform(image=img, bboxes=bounding_boxes, class_labels=img_cls)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    lines = []
    for bbox, label in tqdm(zip(transformed_bboxes, transformed_class_labels)):
        class_index = classes.index(label)
        bbox_str = " ".join(map(str, bbox))
        line = f"{int(class_index)} {bbox_str}\n"
        lines.append(line)
    # Write to a txt file
    #exit()
    with open(os.path.join(aug_lbl_dir, nm + str(3) + '.txt'), 'w') as file:
        file.writelines(lines)
    # save image
    io.imsave(os.path.join(aug_img_dir, nm + str(3) + '.jpg'), transformed_image)