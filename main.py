import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from video_classification.model import AccidentAnalysisModel
from video_classification.dataset import TrafficAccidentDataset
from utils import load_yolo_model, compute_class_weights
from config import DEVICE, MAX_FRAMES, BATCH_SIZE, EPOCHS, LR, NUM_NEGLIGENCE_CLASSES
from train import train_model, test_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Accident Negligence Rate Classification"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to the directory containing videos")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Path to the directory containing annotations")
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="Path to YOLO model weights")
    parser.add_argument("--model_weights", type=str, default="",
                        help="Path to pre-trained accident model weights."
                        "If not provided, the backbone will be initialized with the default pretrained weights (swin3d_t-7615ae03.pth).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for training (default: {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Initial learning rate (default: {LR})")
    return parser.parse_args()

def main():
    args = parse_args()
    video_dir = args.video_dir
    annotation_dir = args.annotation_dir
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    yolo_model = load_yolo_model(args.yolo_weights).to(DEVICE)
    
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    weights_exist = True if args.model_weights != "" else False
    model = AccidentAnalysisModel(num_classes=NUM_NEGLIGENCE_CLASSES, weights_exist=weights_exist)
    if weights_exist:
        pretrained_state_dict = torch.load(args.model_weights)
        model.load_state_dict(pretrained_state_dict)
    
    class_weights = compute_class_weights(annotation_dir)
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        weights=class_weights,
        num_epochs=args.epochs,
        lr=args.lr,
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
