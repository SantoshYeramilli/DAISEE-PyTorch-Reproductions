import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score

log_file = open("test_log_lrcn.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

import torch
from tqdm import tqdm

import torch
from collections import defaultdict
from tqdm import tqdm

def test_model(model, test_loader, criterion, device):
    """
    Evaluates the model, aggregating accuracy per video by averaging clip-wise accuracy.

    :param model: The trained neural network model.
    :param test_loader: DataLoader for test data.
    :param criterion: Loss function (CrossEntropyLoss).
    :param device: Device to run evaluation (cuda/cpu).
    """
    model.eval()
    total_loss = 0
    total_correct = [0, 0, 0, 0]  # Stores clip-wise accuracy
    total_samples = 0

    # Store per-video accuracy
    video_accuracy = defaultdict(lambda: [[] for _ in range(4)])  # Stores accuracy per affective state per video

    with torch.no_grad():
        for batch_frames, batch_labels, video_ids in tqdm(test_loader, desc="Testing"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            outputs = model(batch_frames)  # Outputs shape: (batch_size, 4, num_classes)

            # Compute loss for LRCN
            avg_outputs = [output.mean(dim=1) for output in outputs]  # shape: (batch_size, num_classes)

            # Compute multi-task loss for LRCN
            loss = sum(criterion(avg_outputs[i], batch_labels[:, i]) for i in range(4))
            #loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            total_loss += loss.item()

            # Compute Clip-Level Accuracy & Store in Video Group

            # change avg_outputs to outputs for c3d
            for i in range(4):
                preds = torch.argmax(avg_outputs[i], dim=1)  # Get predictions per affective state
                correct_per_clip = (preds == batch_labels[:, i]).float()  # Convert to float for averaging

                total_correct[i] += correct_per_clip.sum().item()  # Store clip accuracy

                for vid, correct_pred in zip(video_ids, correct_per_clip):
                    video_accuracy[vid][i].append(correct_pred)  # Store per-affective-state accuracy

            total_samples += batch_labels.size(0)

    # Compute Per-Video Accuracy
    video_final_accuracy = {}
    for vid, acc_list in video_accuracy.items():
        per_state_accuracy = [sum(state_acc) / len(state_acc) for state_acc in acc_list]  # Compute per-affective state accuracy
        video_final_accuracy[vid] = per_state_accuracy

    # Compute Average Video-Level Accuracy
    video_accuracy_final = [0] * 4
    video_total = len(video_final_accuracy)

    for acc_list in video_final_accuracy.values():
        for i in range(4):
            video_accuracy_final[i] += acc_list[i]

    video_accuracy_final = [(acc / video_total) * 100 for acc in video_accuracy_final]  # Normalize

    # Compute Final Clip-Level Accuracy
    avg_loss = total_loss / len(test_loader)
    avg_clip_acc = [(correct / total_samples) * 100 for correct in total_correct]

    # Print Final Accuracy Results
    print(f"\nClip-Level Accuracy per Affective State: {avg_clip_acc}")
    print(f"Video-Level Accuracy per Affective State: {video_accuracy_final}")
    print(f"Loss: {avg_loss:.4f}")
    return avg_loss, avg_clip_acc, video_accuracy