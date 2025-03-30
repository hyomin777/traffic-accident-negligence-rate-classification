import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t
from config import (
    NUM_NEGLIGENCE_CLASSES, 
    MAX_DETACTIONS, 
    NUM_ACCIDENT_TYPES, 
    NUM_ACCIDENT_PLACES, 
    NUM_ACCIDENT_PLACE_FEATURES,
    NUM_VEHICLE_A_PROGRESS_INFO,
    NUM_VEHICLE_B_PROGRESS_INFO
)

            
class AccidentAnalysisModel(nn.Module):
    def __init__(
            self, 
            num_classes=NUM_NEGLIGENCE_CLASSES, 
            max_objects=MAX_DETACTIONS,
            num_accident_types=NUM_ACCIDENT_TYPES,
            num_accident_places=NUM_ACCIDENT_PLACES,
            num_accident_place_features=NUM_ACCIDENT_PLACE_FEATURES,
            num_vehicle_a_progress_info=NUM_VEHICLE_A_PROGRESS_INFO,
            num_vehicle_b_progress_info=NUM_VEHICLE_B_PROGRESS_INFO,
            weights_exist=False):
        super().__init__()
        self.backbone = swin3d_t(weights=None)
        if not weights_exist:
            pretrained_state_dict = torch.load('swin3d_t-7615ae03.pth')
            self.backbone.load_state_dict(pretrained_state_dict)
            
        backbone_out_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        self.yolo_encoder = nn.Sequential(
            nn.Linear(max_objects * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        self.metadata_predictor = MetadataPredictor(
            num_accident_types, 
            num_accident_places, 
            num_accident_place_features,
            num_vehicle_a_progress_info,
            num_vehicle_b_progress_info,
            backbone_out_dim
            )
        
        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_dim + 256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, frames, yolo_detections):
        batch_size = frames.shape[0]
        seq_len = frames.shape[1]
        
        # (B, T, C, H, W) -> (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)
        video_features = self.backbone(frames)

        type_pred, place_pred, place_feature_pred, a_progress_info_pred, b_progress_info_pred = self.metadata_predictor(video_features)
        
        meta_input = torch.cat([
                F.softmax(type_pred),
                F.softmax(place_pred),
                F.softmax(place_feature_pred),
                F.softmax(a_progress_info_pred),
                F.softmax(b_progress_info_pred)
        ], dim=1)  # (B, 5)
        metadata_features = self.metadata_encoder(meta_input)

        yolo_flat = yolo_detections.reshape(batch_size, seq_len, -1)
        yolo_features = []
        for t in range(seq_len):
            frame_yolo = yolo_flat[:, t, :]
            frame_features = self.yolo_encoder(frame_yolo)
            yolo_features.append(frame_features)
        
        yolo_features = torch.stack(yolo_features, dim=1)
        yolo_temporal = self.temporal_encoder(yolo_features)
        yolo_pooled = torch.mean(yolo_temporal, dim=1)

        combined_features = torch.cat([video_features, yolo_pooled, metadata_features], dim=1)
        logits = self.fusion(combined_features)
        return logits, (type_pred, place_pred, place_feature_pred, a_progress_info_pred, b_progress_info_pred)


class MetadataPredictor(nn.Module):
    def __init__(
            self,
            num_accident_types, 
            num_accident_places, 
            num_accident_place_features,
            num_vehicle_a_progress_info,
            num_vehicle_b_progress_info,
            dim
            ):
        super().__init__()

        self.accident_type_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_accident_types)
        )
        self.accident_place_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_accident_places)
        )
        self.accident_place_feature_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_accident_place_features)
        )
        self.a_progress_info_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_vehicle_a_progress_info)
        )
        self.b_progress_info_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_vehicle_b_progress_info)
        )
    
    def forward(self, video_features):
        type_pred = self.accident_type_head(video_features)
        place_pred = self.accident_place_head(video_features)
        place_feature_pred = self.accident_place_feature_head(video_features)
        a_progress_pred = self.a_progress_info_head(video_features)
        b_progress_pred = self.b_progress_info_head(video_features)
        return type_pred, place_pred, place_feature_pred, a_progress_pred, b_progress_pred