import torch
from ultralytics import YOLO
import os
import csv
import pandas as pd
from tqdm import tqdm

def get_prediction_string(boxes, scores, classes):
    pred_strs = []
    for i, score in enumerate(scores):
        single_pred_str = ""
        single_pred_str += str(float(classes[i])) + " " + str(float(score)) + " "

        x_center, y_center, width, height = boxes[i]
        x1 = float(x_center) - (float(width) / 2)
        y1 = float(y_center) - (float(height) / 2)
        width = float(width)
        height = float(height)
        single_pred_str += f"{x1} {y1} {width} {height}"

        pred_strs.append(single_pred_str)
    ans = ','.join(map(str, pred_strs))
    if len(ans):
        return ans

    return "0 0 0 0 0 0"

def get_prediction_entry(i, filename, boxes, scores, classes):
    return {
        "id": i,  # strating from 0 ...
        "ImageID": filename.split('.')[0],  # before the extension ...
        "PredictionString_pred": get_prediction_string(boxes, scores, classes)
    }
def predict_all_files(test_directory):
    predictions = []
    for i, filename in tqdm(enumerate(os.listdir(test_directory))):
        if filename.endswith(".jpg"):
            filepath = os.path.join(test_directory, filename)
            results = model.predict(source=filepath, conf=0.50, verbose=False)
            boxes = results[0].boxes.xywhn
            scores = results[0].boxes.conf
            classes = results[0].boxes.cls
            prediction = get_prediction_entry(i, filename, boxes, scores, classes)
            predictions.append(prediction)
    #             to csv format ...
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("submission-42.csv", index=False)

model =  YOLO("/home/nirob/PycharmProjects/PyTorch Learning/runs/detect/train45/weights/best.pt")
test_directory = "/home/nirob/Documents/dlenigma1/BadODD/images/test"
predict_all_files(test_directory)
