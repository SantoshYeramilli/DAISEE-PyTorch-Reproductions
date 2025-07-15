import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from frame_level_inception import InceptionV3MultiTask
from LRCN.dataloader import get_dataloader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
log_file = open("training_log_lrcn.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

from tqdm import tqdm
from collections import defaultdict

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """

    :param model: The neural network model.
    :param train_loader: DataLoader for training data.
    :param criterion: Loss function (CrossEntropyLoss).
    :param optimizer: Optimizer (Adam, SGD, etc.).
    :param device: Device to run the training (cuda/cpu).
    :param epochs: Number of training epochs.
    """
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_correct = [0, 0, 0, 0]  # Per-task frame-level accuracy
        total_samples = 0

        # Track per-video accuracy
        video_predictions = defaultdict(lambda: [[], [], [], []])  # Stores accuracy per affective state per video
        video_counts = defaultdict(int)  # Tracks frame count per video

        for batch_frames, batch_labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_frames)  # List of 4 outputs (one per affective state)

            # Average across time (dim=1 = sequence)
            avg_outputs = [output.mean(dim=1) for output in outputs]  # shape: (batch_size, num_classes)

            # Compute multi-task loss for LRCN
            loss = sum(criterion(avg_outputs[i], batch_labels[:, i]) for i in range(4))


            # Compute separate losses for each affective state
            #loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            for vid, pred in zip(video_ids, outputs):
                if vid not in video_predictions:
                    video_predictions[vid] = []
                video_predictions[vid].append(pred)  # Append logits

            
            # Compute Accuracy Per Clip (as before)
            for i in range(4):
                preds = torch.argmax(avg_outputs[i], dim=1)  # Get predictions per task
                total_correct[i] += (preds == batch_labels[:, i]).sum().item()
            
            total_samples += batch_labels.size(0)
            '''
            # ✅ Compute Final Video-Level Predictions (Fixed)
            video_final_predictions = {
                vid: torch.stack(video_predictions[vid]).mean(dim=0) for vid in video_predictions
            }

            

        # Compute Final Video-Level Accuracy
        video_correct = [0] * 4
        video_total = len(video_final_predictions)

        for vid, avg_pred in video_final_predictions.items():
            final_preds = torch.argmax(avg_pred, dim=1)  # Convert averaged logits to class predictions
            video_correct = [video_correct[i] + (final_preds[i] == batch_labels[:, i]).item() for i in range(4)]  # Compare with first label

        video_acc = [(correct / video_total) * 100 for correct in video_correct]
        '''

        # Compute Average Clip-Level Accuracy
        avg_loss = total_loss / len(train_loader)
        avg_clip_acc = [(correct / total_samples) * 100 for correct in total_correct]

        # Print Training Stats
        log(f"\nEpoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        log(f"clip Accuracy per Affective State: {avg_clip_acc}")
        #log(f"Video-Level Accuracy per Affective State: {video_acc}")

    log_file.close()
    print("Training Complete!")
