from PIL import Image
from skimage import io
import random
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class_names = ['auto_rickshaw', 'bicycle', 'bus', 'car', 'cart_vehicle', 'construction_vehicle', 'motorbike', 'person', 'priority_vehicle', 'three_wheeler', 'train', 'truck', 'wheelchair']

# Create a class_map from the class_names list
class_map = {i: name for i, name in enumerate(class_names)}
#print(class_map)

def read_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def find_or_create_random_bbox(label_file_labels):
    if random.random()<=0.3:
        # Choose a random bbox from existing labels
        random_bbox = random.choice(label_file_labels)
        idx = label_file_labels.index(random_bbox)
        #print(idx)
    else:
        # Create a random bbox
        # Assuming a random box size, adjust these values as needed
        w = random.uniform(0.05, 0.2)  # Random width between 5% to 20% of image width
        h = random.uniform(0.05, 0.2)  # Random height between 5% to 20% of image height
        x_center = random.uniform(w / 2, 1 - w / 2)  # Ensure the bbox is within the image boundaries
        y_center = random.uniform(h / 2, 1 - h / 2)
        random_bbox = (0, x_center, y_center, w, h)  # Class ID set to 0 by default, adjust as needed
        idx=-1

    return random_bbox,idx

def yolo_to_pixels(bbox, img_width, img_height):
    """
    Convert YOLO bbox format to pixel format.
    """
    x_center, y_center, width, height = bbox
    x = int((x_center - width / 2) * img_width)
    y = int((y_center - height / 2) * img_height)
    w = int(width * img_width)
    h = int(height * img_height)
    return x, y, w, h


def draw_bboxes_on_image(img, bboxes):
    """
    Draw bounding boxes with class names on an image.
    """
    # Load the image
    img_width, img_height = img.size

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw each bounding box and class name
    for bbox in bboxes:
        class_id = bbox[0]  # Extract class identifier
        class_name = class_id  # Get class name from map

        # Convert YOLO bbox to pixel format
        x, y, w, h = yolo_to_pixels(bbox[1:], img_width, img_height)

        # Create a rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add class name text
        plt.text(x, y, class_name, color='blue', fontsize=10, ha='left', va='bottom')

    plt.show()

# bbox should contain 5 lines
def crop_and_paste(source_img, target_img, class_12_bbox, label_file_labels):
    # Load the source image

    source_width, source_height = source_img.size

    # Crop class 12 image
    x, y, w, h = yolo_to_pixels(class_12_bbox[1:], source_width, source_height)
    cropped_img = source_img.crop((x, y, x + w, y + h))

    # Find a random bbox in label file labels or create a random bbox
    replacement_bbox,idx = find_or_create_random_bbox(label_file_labels)

    # Load the target image
    #target_img = Image.open(target_img_path)
    target_width, target_height = target_img.size

    # Calculate the pixel coordinates for the random or created bbox
    rx, ry, rw, rh = yolo_to_pixels(replacement_bbox[1:], target_width, target_height)

    # Paste the cropped image onto the target image
    target_img.paste(cropped_img.resize((rw, rh)), (rx, ry))

    # Variable to hold the modified image
    modified_img = target_img

    # Save or return the modified image as needed
    # modified_img.save('/path/to/save/modified_image.jpg')

    # Update the YOLO label for the pasted image
    new_bbox_yolo_format = (class_12_bbox[0], # Class ID
                            (rx + rw / 2) / target_width, # x_center
                            (ry + rh / 2) / target_height, # y_center
                            rw / target_width, # width
                            rh / target_height) # height


    return modified_img , new_bbox_yolo_format , idx

# LOAD IMAGE HERE AND THEN PASS IT IN ICP




