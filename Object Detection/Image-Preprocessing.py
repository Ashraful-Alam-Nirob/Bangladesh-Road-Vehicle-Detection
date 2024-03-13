from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def contrast_stretching(image):
    # Normalize the image to [0, 1] if necessary
    if image.max() > 1:
        image = image / 255.0

    # Get the minimum and maximum pixel values
    min_val = 0.2
    max_val = 0.7

    # Apply contrast stretching
    stretched_image = (image - min_val) / (max_val - min_val)

    return stretched_image

# Load the image
img = io.imread("/media/nirob/Local Disk/dlenigma1/BadODD/images/train/chittagong_night1_8250.jpg")
plt.imshow(img, cmap='gray')
plt.axis('off')  # Turn off axis numbers
plt.show()
# Apply contrast stretching
img_stretched = contrast_stretching(img)

# Display the image
plt.imshow(img_stretched, cmap='gray')
plt.axis('off')  # Turn off axis numbers
plt.show()

