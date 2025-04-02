import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import json
import cv2
from pathlib import Path
from utils import load_yolo_model, detect_objects_in_frame, yolo_results_to_tensor, get_negligence_category
from config import IMAGE_SIZE, NUM_ACCIDENT_TYPES, NUM_ACCIDENT_PLACES, NUM_ACCIDENT_PLACE_FEATURES, NUM_VEHICLE_A_PROGRESS_INFO, NUM_VEHICLE_B_PROGRESS_INFO


class BaseTrafficAccidentDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None, max_frames=32, frame_interval=3, yolo_model=None):
        self.video_dir = data_dir / 'data'
        self.annotation_dir = data_dir / 'annotation'
        self.transform = transform
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.yolo_model = yolo_model if yolo_model else load_yolo_model()
        
        self.samples = []
        self._load_annotations()
    
    def _load_annotations(self):
        unavailable_data_cnt = 0
        for json_file in self.annotation_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            video_name = data['video']['video_name']
            video_path = self.video_dir / f"{video_name}.mp4"
            
            if not video_path.exists():
                continue
            
            if 'accident_negligence_rate' in data['video']:
                rateB = data['video']['accident_negligence_rate']
            else:
                rateB = data['video'].get('accident_negligence_rateB', 50)
            
            video_point_of_view = int(data['video'].get('video_point_of_view', 3))
            accident_type = int(data['video'].get('traffic_accident_type', 0))
            accident_place = int(data['video'].get('accident_place', 0))
            accident_place_feature = int(data['video'].get('accident_place_feature', 0))

            vehicle_a_progress = int(data['video'].get('vehicle_a_progress_info', 0))
            vehicle_b_progress = int(data['video'].get('vehicle_b_progress_info', 0))

            if video_point_of_view != 1:
                unavailable_data_cnt += 1
                continue
            if accident_type < 0 or accident_type >= NUM_ACCIDENT_TYPES:
                unavailable_data_cnt += 1
                continue
            if accident_place < 0 or accident_place >= NUM_ACCIDENT_PLACES:
                unavailable_data_cnt += 1
                continue
            if accident_place_feature < 0 or accident_place_feature >= NUM_ACCIDENT_PLACE_FEATURES:
                unavailable_data_cnt += 1
                continue
            if vehicle_a_progress < 0 or vehicle_a_progress >= NUM_VEHICLE_A_PROGRESS_INFO:
                unavailable_data_cnt += 1
                continue
            if vehicle_b_progress < 0 or vehicle_b_progress >= NUM_VEHICLE_B_PROGRESS_INFO:
                unavailable_data_cnt += 1
                continue

            negligence_category = get_negligence_category(rateB)
            
            self.samples.append({
                'video_path': str(video_path),
                'negligence_category': negligence_category,
                'accident_type': accident_type,
                'accident_place': accident_place,
                'accident_place_feature': accident_place_feature,
                'vehicle_a_progress': vehicle_a_progress,
                'vehicle_b_progress': vehicle_b_progress
            })
        print(f"Num of Unavailable Data : {unavailable_data_cnt}")

    def __len__(self):
        return len(self.samples)
    
    def _load_and_preprocess_frame(self, frame, apply_transform=True):
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detect_objects_in_frame(self.yolo_model, frame)
        yolo_tensor = yolo_results_to_tensor(results)
        
        if apply_transform and self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return frame, yolo_tensor
        
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
                
            processed_frame, yolo_tensor = self._process_frame(frame)
            
            frames.append(processed_frame)
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
            'negligence_category': sample['negligence_category'],
            'metadata': metadata
        }
    
    def _process_frame(self, frame):
        raise NotImplementedError("Subclasses must implement _process_frame")


class TrainDataset(BaseTrafficAccidentDataset):
    def __init__(self, data_dir, transform=None, augment=True, max_frames=32, frame_interval=3, yolo_model=None):
        super().__init__(data_dir, transform, max_frames, frame_interval, yolo_model)
        self.augment = augment
        self.aug_transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
    
    def _process_frame(self, frame):
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detect_objects_in_frame(self.yolo_model, frame)
        yolo_tensor = yolo_results_to_tensor(results)

        pil_frame = T.ToPILImage()(torch.from_numpy(frame).permute(2, 0, 1))

        if self.augment and np.random.random() < 0.5:
            pil_frame = self.aug_transform(pil_frame)
        
        if self.transform:
            frame_tensor = self.transform(pil_frame)
        else:
            frame_tensor = T.ToTensor()(pil_frame)
        
        return frame_tensor, yolo_tensor


class ValDataset(BaseTrafficAccidentDataset):
    def _process_frame(self, frame):
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detect_objects_in_frame(self.yolo_model, frame)
        yolo_tensor = yolo_results_to_tensor(results)
        
        pil_frame = T.ToPILImage()(torch.from_numpy(frame).permute(2, 0, 1))
        
        if self.transform:
            frame_tensor = self.transform(pil_frame)
        else:
            frame_tensor = T.ToTensor()(pil_frame)
        
        return frame_tensor, yolo_tensor
