import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from PIL import Image
from utils import load_yolo_model

yolo_model = load_yolo_model()

image_path = "test_image.png"
image = Image.open(image_path).convert("RGB")

results = yolo_model(image)


