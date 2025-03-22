from ultralytics import YOLO
from PIL import Image


model = YOLO("best.pt")

image_path = "test_image.png"
image = Image.open(image_path)

results = model(image)
print(results)