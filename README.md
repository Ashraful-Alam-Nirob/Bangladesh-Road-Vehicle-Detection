
### 12. What is the process for adding data augmentation to the pipeline?
  
Data augmentation is a technique used to artificially expand the training dataset by applying transformations to the images. It helps improve model generalization by exposing it to varied versions of the same data, thereby reducing overfitting and improving robustness.

To enable data augmentation in the pipeline, follow these steps:

1. Set `augment = True` in the `config.py` file.
2. The pipeline currently includes the following augmentation techniques:
   - **VerticalFlip**: Flips the image vertically with a certain probability.
   - **HorizontalFlip**: Flips the image horizontally with a certain probability.
   - **RandomRotate90**: Rotates the image by 90 degrees randomly.
   - **Blur**: Applies a blur effect to the image.

3. To add more augmentation techniques, you can modify the `Augment` class located in the `dataset.py` file. Incorporate additional augmentation methods as needed to enhance the variability of your dataset.

**Example:**
```python
self.aug = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.5),
        A.CLAHE(p=0.5),  # Contrast Limited Adaptive Histogram Equalization
        A.ColorJitter(p=0.5),  # Random changes in brightness, contrast, saturation, and hue
    ]
)
```

By enabling and customizing augmentations, you can adapt the pipeline to specific requirements, improving the diversity and quality of training data for better model performance.

### 13. What is class balance threshold and how can I control it during training?

The class balance threshold determines the minimum percentage of the positive class that must be present in a patch image for it to be included in the training process. Any patch with a positive class proportion lower than the threshold will be discarded and not used for training.

To adjust the class balance threshold, modify the `class_balance_threshold` variable in the `config.py` file based on your requirements. This provides flexibility in controlling the inclusion criteria for patch images during training.

```python
# Example configuration in config.py
class_balance_threshold = 20  # 20% positive class required in a patch
```

- If `class_balance_threshold = 20`, only patches where the positive class (e.g., foreground pixels) covers **at least 20%** of the patch will be used for training.  
- If a patch has a positive class proportion of 15%, it will be discarded.
- If a patch has a positive class proportion of 25%, it will be included in the training set.

This mechanism ensures that patches with insufficient positive samples are excluded, improving the training process by focusing on meaningful patches.

### 14. How do I check the number of patches created during data preprocessing?

The number of patches created during preprocessing is determined using the formula:

<img src="https://latex.codecogs.com/svg.image?$$\text{number\_of\_patches}=\left\lceil\frac{\text{height}-\text{patch\_size}}{\text{stride}}&plus;1\right\rceil\times\left\lceil\frac{\text{width}-\text{patch\_size}}{\text{stride}}&plus;1\right\rceil$$" title="$$\text{number\_of\_patches}=\left\lceil\frac{\text{height}-\text{patch\_size}}{\text{stride}}+1\right\rceil\times\left\lceil\frac{\text{width}-\text{patch\_size}}{\text{stride}}+1\right\rceil$$" />

These patches are further processed and evaluated against the class balance threshold if `patch_class_balance` is set to `True` in the `config.py` file. All generated patches and their corresponding metadata are saved in the JSON file.

To count the patches, use the `number_of_patches()` function available in the `visualization.ipynb` file. This function reads the JSON file and calculates the total number of patches based on the metadata. Proper evaluation of patches ensures alignment with training requirements and dataset consistency.


## 15. What is the significance of the CSV and JSON files, and what information do they typically contain?

CSV and JSON files organize the dataset for training, testing, and evaluation. The CSV file contains the directories for the training, testing, and validation images where each CSV file contains `feature_ids` and `masks`, where `feature_ids` represent the original images and `masks` represent the ground truth of the data. It serves as a reference for locating the dataset split during the training process, while JSON files store patch metadata, including source image details and patch coordinates.

The JSON file contains details about patches such as their coordinates and corresponding source images. For example:

```json
{
  "feature_ids": ["image1.jpg", "image1.jpg"],
  "masks": ["image1_mask.jpg", "image1_mask.jpg"],
  "patch_idx": [[0, 0, 512, 512], [512, 0, 1024, 512]]
}
```
These files ensure proper data handling during training and preprocessing, avoiding errors due to inconsistencies.


## 16. How can I randomly or selectively plot data samples during training?

Plotting data samples during training helps verify preprocessing, augmentations, and dataset integrity. The behavior is controlled by the `index` variable in `config.py`:
```python
index = "random"  # For random plotting
```
- Set `index = 1` to plot the first validation sample consistently.
- Set `index = "random"` to plot samples randomly during training.

The plots can be found in the directory:
```
root/logs/prediction/model_name/validation/experiment_name.jpg
```

---

## 17. How can I control the data split for training, validation, and testing?

Controlling the data split allows you to allocate specific proportions of the dataset for training, validation, and testing, ensuring proper evaluation and model performance.

In `config.py`, adjust the following variables:
```python
train_size = 0.8  # 80% of the data for training
test_size = 0.5   # 10% for testing and 10% for validation (remaining after training)
```
This ensures data is divided as required, with the remaining portion allocated for validation and testing.

---

## 18. What steps are required to add additional performance metrics to the training process?

Adding performance metrics allows for a more comprehensive evaluation of the model. Metrics like **Mean IoU** and **Dice Coefficient** are defined in the `metrics.py` file. To add new metrics, modify the `get_metrics()` function in `metrics.py`:
```python
def get_metrics():
    return {
        "MyMeanIoU": MyMeanIoU(num_classes),
        "f1-score": tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        "dice_coef_score": dice_coef_score,
    }
```

These additional metrics will be calculated during training and validation, providing deeper insights into model performance.
