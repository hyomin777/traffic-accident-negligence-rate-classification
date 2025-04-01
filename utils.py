import torch
from collections import Counter
import json
from pathlib import Path
import numpy as np
from config import MAX_DETACTIONS, NEGLIGENCE_CATEGORIES, NUM_NEGLIGENCE_CLASSES
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

def get_negligence_category(rateB:int):
    rateB = max(0, min(100, rateB))
    if rateB % 10 != 0:
        rateB = rateB // 10 * 10 
    rateA = 100 - rateB
    category = f"{rateB}:{rateA}"
    return NEGLIGENCE_CATEGORIES[category]

def compute_class_weights(annotation_dir, smoothing=0.5, num_classes=NUM_NEGLIGENCE_CLASSES):
    counts = Counter()
    annotation_dir = Path(annotation_dir)
    
    for json_file in annotation_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'accident_negligence_rate' in data['video']:
            rateA = data['video']['accident_negligence_rate']
        else:
            rateA = data['video'].get('accident_negligence_rateA', 50)

        label = get_negligence_category(rateA)
        counts[label] += 1
    
    total_samples = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0: 
            weights.append(10.0)
        else:
            weight = (total_samples / (count + smoothing)) ** 1.5
            weights.append(weight)

    weight_tensor = torch.tensor(weights)
    weight_tensor = weight_tensor * num_classes / weight_tensor.sum()
    print("Class weights:", weight_tensor)
    return weight_tensor
