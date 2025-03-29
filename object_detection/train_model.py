from ultralytics import YOLO

model = YOLO('yolo11n.pt')

data_config = 'yolo_config.yaml'

results = model.train(
    data = data_config,
    epochs = 50,
    imgsz = 640,
    batch = 128,
    name = 'object_detection',
)
