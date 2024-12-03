
## 15. What is the significance of the CSV and JSON files, and what information do they typically contain?

CSV and JSON files organize the dataset for training, testing, and evaluation. CSV files manage dataset splits, while JSON files store patch metadata, including source image details and patch coordinates.

The CSV file includes directories for training, testing, and validation images. The JSON file contains details about patches such as their coordinates and corresponding source images. For example:
```json
{
  "feature_ids": ["image1.jpg", "image1.jpg"],
  "masks": ["image1_mask.jpg", "image1_mask.jpg"],
  "patch_idx": [[0, 0, 512, 512], [512, 0, 1024, 512]]
}
```
These files ensure proper data handling during training and preprocessing, avoiding errors due to inconsistencies.

---

## 16. How can I randomly or selectively plot data samples during training?

To plot data samples during training, the `index` variable in `config.py` controls the behavior:
```python
index = "random"  # For random plotting
```
Setting `index = 1` allows selective plotting of the first validation sample, while `index = "random"` enables plotting of random samples during training. This helps inspect preprocessing and dataset integrity dynamically.

---

## 17. How can I control the data split for training, validation, and testing?

The dataset can be split into training, validation, and testing subsets by adjusting the following variables in `config.py`:
```python
train_size = 0.8  # 80% of the data for training
test_size = 0.5   # 10% for testing and 10% for validation (remaining after training)
```
These settings ensure the desired distribution of data, improving model evaluation consistency.

---

## 18. What steps are required to add additional performance metrics to the training process?

In the `metrics.py` file, performance metrics such as **Mean IoU** and **Dice Coefficient** are defined. To add new metrics, update the `get_metrics()` function. For example:
```python
def get_metrics():
    return {
        "MyMeanIoU": MyMeanIoU(num_classes),
        "f1-score": tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
        "dice_coef_score": dice_coef_score,
    }
```
Including additional metrics allows for comprehensive evaluation of model performance during training and validation.
