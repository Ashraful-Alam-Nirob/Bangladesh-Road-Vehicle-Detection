from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("test1.npy")