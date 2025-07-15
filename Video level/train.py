import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

log_file = open("results/video_level/training_log_taskhead_video.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Trains the Multi-Task InceptionV3 model on DAiSEE dataset,
    and tracks both frame-level and video-level accuracy.

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

        # ✅ Track per-video accuracy
        video_accuracy = defaultdict(lambda: [[], [], [], []])  # Stores accuracy per affective state per video
        video_counts = defaultdict(int)  # Tracks frame count per video

        for batch_frames, batch_labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_frames)  # List of 4 outputs (one per affective state)

            # Compute separate losses for each affective state
            loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Compute Accuracy Per Frame and Store by `video_id`
            for i in range(4):
                preds = torch.argmax(outputs[i], dim=1)  # Convert logits to predicted class index
                correct = (preds == batch_labels[:, i]).cpu().numpy()

                for vid, correct_pred in zip(video_ids, correct):
                    video_accuracy[vid][i].append(correct_pred)  # Store frame accuracy per video ID

                total_correct[i] += correct.sum()

            total_samples += batch_labels.size(0)

        # ✅ Compute Per-Video Accuracy
        video_final_accuracy = {}
        for vid, acc_list in video_accuracy.items():
            per_state_accuracy = [sum(state_acc) / len(state_acc) for state_acc in acc_list]
            video_final_accuracy[vid] = per_state_accuracy

        # ✅ Compute Average Frame-Level Accuracy
        avg_loss = total_loss / len(train_loader)
        avg_frame_acc = [(correct / total_samples) * 100 for correct in total_correct]

        # ✅ Compute Average Video-Level Accuracy
        avg_video_acc = [sum(state_acc) / len(video_final_accuracy) for state_acc in zip(*video_final_accuracy.values())]

        # ✅ Print Training Stats
        log(f"\nEpoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        log(f"Frame-Level Accuracy per Affective State: {avg_frame_acc}")
        log(f" Video-Level Accuracy per Affective State: {avg_video_acc}")
    log_file.close()
    print(" Training Complete!")
