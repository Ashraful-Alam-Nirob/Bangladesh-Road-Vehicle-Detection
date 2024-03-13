import albumentations as A
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from skimage import io
import time

transform = A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
    #A.VerticalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.2),
    #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.2, border_mode=0),
    #A.CLAHE(clip_limit=2, p=0.2),
    #A.OpticalDistortion(p=0.1),
    #A.GridDistortion(p=0.1),
    #A.Cutout(num_holes=8, max_h_size = 30, max_w_size = 30, fill_value=0, always_apply=False, p = 0.5)
    A.augmentations.geometric.resize.Resize (224, 224, interpolation=1, always_apply=True, p=1),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1,min_area=1024, label_fields=['class_labels']))

img_dir = "/home/nirob/Documents/dlenigma1/BadODD/images/aug"
lbl_dir = "/home/nirob/Documents/dlenigma1/BadODD/labels/aug"
aug_img_dir = "/home/nirob/Documents/dlenigma1/BadODD/images/feat_img"
aug_lbl_dir = "/home/nirob/Documents/dlenigma1/BadODD/labels/feat_lbl"
classes = ['auto_rickshaw',"bicycle","bus", "car","cart_vehicle","construction_vehicle","motorbike","person","priority_vehicle","three_wheeler","train","truck","wheelchair"]


for i in tqdm(range(1)):
    for name in tqdm(os.listdir( lbl_dir)):
        nm = name.split(".txt")[0]
        img = io.imread(os.path.join(img_dir,nm+".jpg"))
        bounding_boxes = []
        img_cls = []
        with open(os.path.join(lbl_dir,name), 'r') as f:
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
        with open(os.path.join(aug_lbl_dir,nm+str(i)+'.txt'), 'w') as file:
            file.writelines(lines)
        # save image
        io.imsave(os.path.join(aug_img_dir,nm+str(i)+'.jpg'), transformed_image)














