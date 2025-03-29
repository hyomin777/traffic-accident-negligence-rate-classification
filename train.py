import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from config import DEVICE, EPOCHS, LR, NUM_NEGLIGENCE_CLASSES


def train_model(model: nn.Module, train_loader, val_loader, num_epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # AMP를 위한 GradScaler 초기화
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # GPU로 데이터 비동기 이동 (non_blocking)
            frames = batch['frames'].to(DEVICE, non_blocking=True)
            yolo_detections = batch['yolo_detections'].to(DEVICE, non_blocking=True)
            targets = batch['negligence_category'].to(DEVICE, non_blocking=True)
            metadata = batch['metadata'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # AMP 자동 캐스팅 사용
            with torch.cuda.amp.autocast():
                outputs = model(frames, yolo_detections, metadata)
                loss = criterion(outputs, targets)
            
            # AMP 스케일러로 역전파 및 옵티마이저 스텝
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = 100.0 * train_correct / train_total
        
        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        confusion = torch.zeros(NUM_NEGLIGENCE_CLASSES, NUM_NEGLIGENCE_CLASSES, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(DEVICE, non_blocking=True)
                yolo_detections = batch['yolo_detections'].to(DEVICE, non_blocking=True)
                targets = batch['negligence_category'].to(DEVICE, non_blocking=True)
                metadata = batch['metadata'].to(DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(frames, yolo_detections, metadata)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 배치 단위로 confusion matrix 업데이트
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # 모델 저장 조건
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
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch['frames'].to(DEVICE)
            yolo_detections = batch['yolo_detections'].to(DEVICE)
            targets = batch['negligence_category'].to(DEVICE)
            metadata = batch['metadata'].to(DEVICE)
            
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