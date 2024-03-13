from ultralytics import YOLO
import torch 


model = YOLO("yolov8m.pt")

results = model.train(data='config.yaml',epochs=100,batch=24)




