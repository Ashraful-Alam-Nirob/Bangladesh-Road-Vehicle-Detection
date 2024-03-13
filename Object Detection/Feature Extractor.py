import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Path to the directory containing images
image_directory = '/home/nirob/Documents/dlenigma1/BadODD/images/feat_img'
save_directory = "/home/nirob/Documents/dlenigma1/BadODD/images/train_ready"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Transform for the input images
transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = CustomDataset(image_directory, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the pre-trained ResNet101 model and modify it
model = resnet101(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move model to the appropriate device

# Function to save feature maps as images
def save_feature_as_image(feature, filename):
    # Convert the tensor to numpy
    feature = feature.squeeze().cpu().numpy()
    # Use the mean of the feature maps to get a representative image
    feature_img = feature.mean(axis=0)
    # Normalize the image to be in [0, 255]
    feature_img = 255 * (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
    feature_img = feature_img.astype(np.uint8)
    # Convert to PIL image and save
    img = Image.fromarray(feature_img)
    img.save(filename)

# Extract features and save as images
for inputs, img_name in tqdm(dataloader):
    inputs = inputs.to(device)
    with torch.no_grad():
        features = model(inputs)
    # Define the path for saving the feature image
    save_path = os.path.join(save_directory, (img_name[0]) + '.jpg')
    save_feature_as_image(features, save_path)
