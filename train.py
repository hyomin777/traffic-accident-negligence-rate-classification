import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from video_classification.loss import FocalLoss
from config import DEVICE, EPOCHS, LR, AUX_LAMBDA, GAMMA, AUX_GAMMA, NUM_NEGLIGENCE_CLASSES
import os
from pathlib import Path


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_acc, experiment_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_acc': best_val_acc
    }

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{experiment_name}_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(model, optimizer, scheduler, scaler, experiment_name):
    checkpoint_path = Path('checkpoints') / f'{experiment_name}_checkpoint.pth'

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f'Checkpoint loaded from epoch {checkpoint["epoch"]}')
        return start_epoch, best_val_acc
    return 1, 0.0


def save_weights(model, val_acc, experiment_name):
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    weigths_path = weights_dir / f'{experiment_name}_{val_acc}.pth'
    torch.save(model.state_dict(), weigths_path)
    print(f'Model saved with accuracy: {val_acc:.2f}%')



def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    weights,
    num_epochs=EPOCHS,
    lr=LR,
    aux_lambda=AUX_LAMBDA,
    experiment_name="experiment",
    resume=False
):
    writer = SummaryWriter(log_dir=f'runs/{experiment_name}')
    model = model.to(DEVICE)

    criterion = FocalLoss(weight=weights, gamma=GAMMA).to(DEVICE)
    aux_criterion = FocalLoss(gamma=AUX_GAMMA).to(DEVICE)
    aux_lambda = aux_lambda

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(DEVICE)

    start_epoch = 1
    best_val_acc = 0.0

    if resume:
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, scaler, experiment_name)

    for epoch in tqdm(range(start_epoch, num_epochs+1), desc="Epochs"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(DEVICE, non_blocking=True)
            yolo_detections = batch['yolo_detections'].to(
                DEVICE, non_blocking=True)
            targets = batch['negligence_category'].to(
                DEVICE, non_blocking=True)
            metadata = batch['metadata'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(DEVICE):
                outputs, meta_preds = model(frames, yolo_detections)
                loss_cls = criterion(outputs, targets)

                type_pred, place_pred, place_feature_pred, a_progress_info, b_progress_info = meta_preds

                gt_type = metadata[:, 0].long()
                gt_place = metadata[:, 1].long()
                gt_place_feature = metadata[:, 2].long()
                gt_a_progress_info = metadata[:, 3].long()
                gt_b_progress_info = metadata[:, 4].long()

                loss_type = aux_criterion(type_pred, gt_type)
                loss_place = aux_criterion(place_pred, gt_place)
                loss_place_feature = aux_criterion(
                    place_feature_pred, gt_place_feature)
                loss_a_progess_info = aux_criterion(
                    a_progress_info, gt_a_progress_info)
                loss_b_progress_info = aux_criterion(
                    b_progress_info, gt_b_progress_info)

                loss_aux = loss_type + loss_place + loss_place_feature + \
                    loss_a_progess_info + loss_b_progress_info

                loss = loss_cls + aux_lambda * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        train_acc = 100.0 * train_correct / train_total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        confusion = torch.zeros(
            NUM_NEGLIGENCE_CLASSES, NUM_NEGLIGENCE_CLASSES, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(DEVICE, non_blocking=True)
                yolo_detections = batch['yolo_detections'].to(
                    DEVICE, non_blocking=True)
                targets = batch['negligence_category'].to(
                    DEVICE, non_blocking=True)
                metadata = batch['metadata'].to(DEVICE, non_blocking=True)

                with torch.amp.autocast(DEVICE):
                    outputs, meta_preds = model(frames, yolo_detections)
                    loss_cls = criterion(outputs, targets)

                    type_pred, place_pred, place_feature_pred, a_progress_info, b_progress_info = meta_preds

                    gt_type = metadata[:, 0].long()
                    gt_place = metadata[:, 1].long()
                    gt_place_feature = metadata[:, 2].long()
                    gt_a_progress_info = metadata[:, 3].long()
                    gt_b_progress_info = metadata[:, 4].long()

                    loss_type = aux_criterion(type_pred, gt_type)
                    loss_place = aux_criterion(place_pred, gt_place)
                    loss_place_feature = aux_criterion(
                        place_feature_pred, gt_place_feature)
                    loss_a_progess_info = aux_criterion(
                        a_progress_info, gt_a_progress_info)
                    loss_b_progress_info = aux_criterion(
                        b_progress_info, gt_b_progress_info)

                    loss_aux = loss_type + loss_place + loss_place_feature + \
                        loss_a_progess_info + loss_b_progress_info

                    loss = loss_cls + aux_lambda * loss_aux

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion[t.long(), p.long()] += 1

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        print("\nConfusion Matrix:")
        print(confusion)

        class_acc = confusion.diag().float() / confusion.sum(1).float() * 100
        for i, acc in enumerate(class_acc):
            print(f'Class {i} (Negligence {i*10}:{100-i*10}): {acc:.2f}%')

        # 체크포인트 저장
        save_checkpoint(epoch, model, optimizer, scheduler,
                        scaler, best_val_acc, experiment_name)

        # 모델 저장 조건
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_weights(model, val_acc, experiment_name)
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
            # metadata = batch['metadata'].to(DEVICE)

            outputs, _ = model(frames, yolo_detections)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean() * 100
    cm = confusion_matrix(all_targets, all_preds)

    class_names = [f"{i*10}:{100-i*10}" for i in range(11)]
    report = classification_report(
        all_targets, all_preds, target_names=class_names)

    return accuracy, cm, report
