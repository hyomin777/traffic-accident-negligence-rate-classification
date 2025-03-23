import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from config import EPOCHS, LR, NUM_NEGLIGENCE_CLASSES


def train_model(model:nn.Module, train_loader, val_loader, num_epochs=EPOCHS, lr=LR):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            frames = batch['frames'].to(device)
            yolo_detections = batch['yolo_detections'].to(device)
            targets = batch['negligence_category'].to(device)
            metadata = batch['metadata'].to(device)
            
            optimizer.zero_grad()
            outputs = model(frames, yolo_detections, metadata)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100.0 * train_correct / train_total
        

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        confusion = torch.zeros(NUM_NEGLIGENCE_CLASSES, NUM_NEGLIGENCE_CLASSES, dtype=torch.long).to(device)
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                yolo_detections = batch['yolo_detections'].to(device)
                targets = batch['negligence_category'].to(device)
                metadata = batch['metadata'].to(device)
                
                outputs = model(frames, yolo_detections, metadata)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_accident_model.pth')
            print(f'Model saved with accuracy: {val_acc:.2f}%')
            print("\nConfusion Matrix:")
            print(confusion)
            
            class_acc = confusion.diag().float() / confusion.sum(1).float() * 100
            for i, acc in enumerate(class_acc):
                print(f'Class {i} (Negligence {i*10}:{100-i*10}): {acc:.2f}%')
    
    return model

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch['frames'].to(device)
            yolo_detections = batch['yolo_detections'].to(device)
            targets = batch['negligence_category'].to(device)
            metadata = batch['metadata'].to(device)
            
            outputs = model(frames, yolo_detections, metadata)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = (all_preds == all_targets).mean() * 100
    cm = confusion_matrix(all_targets, all_preds)
    
    class_names = [f"{i*10}:{100-i*10}" for i in range(11)]
    report = classification_report(all_targets, all_preds, target_names=class_names)
    
    return accuracy, cm, report