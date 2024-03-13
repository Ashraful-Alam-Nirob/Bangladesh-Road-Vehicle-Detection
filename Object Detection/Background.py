import cv2
import os
import random
def yolo_to_standard(image, bbox):
    """
    Convert YOLO bbox format to standard format
    [center_x, center_y, width, height] to [x1, y1, x2, y2]
    """
    img_h, img_w, _ = image.shape
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return [x1, y1, x2, y2]


def draw_box(image,yolo_bboxes,name):
    color = (0, 255, 0)
    train_img_dir_1="/home/nirob/Documents/dlenigma1/BadODD/images/aug"
    train_label_dir_2="/home/nirob/Documents/dlenigma1/BadODD/labels/aug"
    for yolo_bbox in yolo_bboxes:
        standard_bbox = yolo_to_standard(image, yolo_bbox)
        image=cv2.rectangle(image, (standard_bbox[0], standard_bbox[1]), (standard_bbox[2], standard_bbox[3]), color, -1)
    cv2.imwrite(f'{train_img_dir_1}/{name}_bg.jpg', image)
    with open(f'{train_label_dir_2}/{name}_bg.txt','w') as f:
        pass
# Load your image

train_img_dir = '/home/nirob/Documents/dlenigma1/BadODD/images/train'
train_label_dir = '/home/nirob/Documents/dlenigma1/BadODD/labels/train'

val_size=0.5

images = [f for f in os.listdir(train_img_dir) if os.path.isfile(os.path.join(train_img_dir, f))]
total_images = len(images)
val_count = int(total_images * val_size)

    # Randomly select a subset of images for validation
val_images = random.sample(images, val_count)
x=0
print(len(val_images))
for img in val_images:
    image = cv2.imread(os.path.join(train_img_dir,img))
    lbl = os.path.join(train_label_dir,img.split(".")[0]+".txt")
    name=img.split(".")[0]
    #print(os.path.join(train_img_dir,img))
    #print(lbl)
    yolo_bboxes=[]
    with open(lbl,'r') as f :
        for line in f :
            yolo_bboxes.append(list(map(float,line.split()[1:])))
    draw_box(image,yolo_bboxes,name)

