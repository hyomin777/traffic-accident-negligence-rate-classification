from ultralytics import YOLO

model = YOLO('best.pt')

data_config = 'data_config.yaml'

results = model.train(
    data = data_config,
    epochs = 50,
    imgsz = 640,
    batch = 32,
    name = 'object_detection'
)