
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
