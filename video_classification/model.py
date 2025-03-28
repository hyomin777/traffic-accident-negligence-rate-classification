import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t
from config import NUM_NEGLIGENCE_CLASSES, MAX_DETACTIONS

            
class AccidentAnalysisModel(nn.Module):
    def __init__(
            self, 
            num_classes=NUM_NEGLIGENCE_CLASSES, 
            max_objects=MAX_DETACTIONS,
            pretrained=False):
        super().__init__()
        
        self.backbone = swin3d_t(weights=None)
        if pretrained:
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

        self.metadata_predictor = MetadataPredictor()
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
    
    def forward(self, frames, yolo_detections, metadata=None):
        batch_size = frames.shape[0]
        seq_len = frames.shape[1]
        
        frames = frames.permute(0, 2, 1, 3, 4)
        video_features = self.backbone(frames)

        if metadata is None:
            metadata = self.metadata_predictor(video_features)
        
        yolo_flat = yolo_detections.reshape(batch_size, seq_len, -1)
        yolo_features = []
        for t in range(seq_len):
            frame_yolo = yolo_flat[:, t, :]
            frame_features = self.yolo_encoder(frame_yolo)
            yolo_features.append(frame_features)
        
        yolo_features = torch.stack(yolo_features, dim=1)
        yolo_temporal = self.temporal_encoder(yolo_features)
        yolo_pooled = torch.mean(yolo_temporal, dim=1)
        
        metadata_features = self.metadata_encoder(metadata)

        combined_features = torch.cat([video_features, yolo_pooled, metadata_features], dim=1)
        logits = self.fusion(combined_features)
        return logits


class MetadataPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    
    def forward(self, video_features):
        return self.predictor(video_features)