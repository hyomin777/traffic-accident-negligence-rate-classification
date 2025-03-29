import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import cv2
from pathlib import Path
from utils import load_yolo_model, detect_objects_in_frame, yolo_results_to_tensor, get_negligence_category
from config import IMAGE_SIZE


class TrafficAccidentDataset(Dataset):
    def __init__(self, video_dir, annotation_dir, transform=None, max_frames=32, frame_interval=3, yolo_model=None):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.yolo_model = yolo_model if yolo_model else load_yolo_model()
        
        self.samples = []
        self._load_annotations()
    
    def _load_annotations(self):
        for json_file in self.annotation_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            video_name = data['video']['video_name']
            video_path = self.video_dir / f"{video_name}.mp4"
            
            if not video_path.exists():
                continue
            
            if 'accident_negligence_rate' in data['video']:
                rateA = data['video']['accident_negligence_rate']
                rateB = 100 - rateA
            else:
                rateA = data['video'].get('accident_negligence_rateA', 50)
                rateB = data['video'].get('accident_negligence_rateB', 50)
            
            accident_type = data['video'].get('traffic_accident_type', 0)
            accident_place = data['video'].get('accident_place', 0)
            accident_place_feature = data['video'].get('accident_place_feature', 0)
            vehicle_a_progress = data['video'].get('vehicle_a_progress_info', 0)
            vehicle_b_progress = data['video'].get('vehicle_b_progress_info', 0)
            
            self.samples.append({
                'video_path': str(video_path),
                'rateA': rateA,
                'rateB': rateB,
                'accident_type': accident_type,
                'accident_place': accident_place,
                'accident_place_feature': accident_place_feature,
                'vehicle_a_progress': vehicle_a_progress,
                'vehicle_b_progress': vehicle_b_progress
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= self.max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)

        frames = []
        yolo_tensors = []
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = detect_objects_in_frame(self.yolo_model, frame)
            yolo_tensor = yolo_results_to_tensor(results)
            
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            frames.append(frame)
            yolo_tensors.append(yolo_tensor)
        
        cap.release()
        
        if len(frames) < self.max_frames:
            if len(frames) > 0:
                last_frame = frames[-1]
                last_yolo = yolo_tensors[-1]
            else:
                last_frame = torch.zeros((3, *IMAGE_SIZE[::-1]))
                last_yolo = torch.zeros((20, 6))
            
            frames.extend([last_frame] * (self.max_frames - len(frames)))
            yolo_tensors.extend([last_yolo] * (self.max_frames - len(yolo_tensors)))
        
        frames_tensor = torch.stack(frames)  # (T, C, H, W)
        yolo_tensor = torch.stack(yolo_tensors)  # (T, max_detections, 6)
        
        negligence_category = get_negligence_category(sample['rateA'])
        
        metadata = torch.tensor([
            sample['accident_type'],
            sample['accident_place'],
            sample['accident_place_feature'],
            sample['vehicle_a_progress'],
            sample['vehicle_b_progress']
        ], dtype=torch.float32)
        
        return {
            'frames': frames_tensor,
            'yolo_detections': yolo_tensor,
            'negligence_category': negligence_category,
            'metadata': metadata
        }