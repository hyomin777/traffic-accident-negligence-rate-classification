import torch
import numpy as np
from config import MAX_DETACTIONS, NEGLIGENCE_CATEGORIES
from ultralytics import YOLO


def load_yolo_model(weights_path="best.pt"):
    return YOLO(weights_path)

def detect_objects_in_frame(yolo_model, frame):
    results = yolo_model(frame, verbose=False)
    return results[0]

def yolo_results_to_tensor(results, max_detections=MAX_DETACTIONS):
    if results.boxes is None or len(results.boxes) == 0:
        return torch.zeros((max_detections, 6))
    
    boxes = results.boxes.cpu()
    xyxy = boxes.xyxy.numpy()
    conf = boxes.conf.numpy()[:, None]
    cls = boxes.cls.numpy()[:, None]

    xywh = np.zeros_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]      
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1] 

    h, w = results.orig_shape
    xywh[:, 0] /= w
    xywh[:, 1] /= h
    xywh[:, 2] /= w
    xywh[:, 3] /= h

    detections = np.hstack([xywh, conf, cls])
    
    if len(detections) > max_detections:
        detections = detections[np.argsort(-detections[:, 4])][:max_detections]
    else:
        padding = np.zeros((max_detections - len(detections), 6))
        detections = np.vstack([detections, padding])
    
    return torch.from_numpy(detections).float()
    

def get_negligence_category(rateA:int, rateB=None):
    if rateB is None:
        rateB = 100 - rateA
    category = f"{rateA}:{rateB}"
    return NEGLIGENCE_CATEGORIES[category]

