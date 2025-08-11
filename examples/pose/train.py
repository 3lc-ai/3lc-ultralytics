from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
if __name__ == "__main__":
    results = model.train(data="C:/Project/datasets/hand-keypoints/data.yaml", epochs=10, imgsz=640)

# c:\Project\3lc-ultralytics\runs\pose\train6\weights\best.pt
