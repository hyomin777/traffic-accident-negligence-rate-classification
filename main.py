import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from video_classification.model import AccidentAnalysisModel
from video_classification.dataset import TrafficAccidentDataset
from utils import load_yolo_model
from config import DEVICE, MAX_FRAMES, BATCH_SIZE, NUM_NEGLIGENCE_CLASSES, VIDEO_DIR_PATH, ANNOTATION_DIR_PATH, PRETRAINED
from train import train_model, test_model


def main():
    video_dir = VIDEO_DIR_PATH
    annotation_dir = ANNOTATION_DIR_PATH
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    yolo_model = load_yolo_model("best.pt").to(DEVICE)
    
    dataset = TrafficAccidentDataset(
        video_dir=video_dir,
        annotation_dir=annotation_dir,
        transform=transform,
        max_frames=MAX_FRAMES,
        yolo_model=yolo_model
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = AccidentAnalysisModel(num_classes=NUM_NEGLIGENCE_CLASSES, pretrained=PRETRAINED)
    if PRETRAINED:
        pretrained_state_dict = torch.load('best_accident_model.pth.pth')
        model.load_state_dict(pretrained_state_dict)
        
    trained_model = train_model(
        model,
        train_loader,
        val_loader
    )

    accuracy, confusion_matrix, report = test_model(trained_model, test_loader)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
