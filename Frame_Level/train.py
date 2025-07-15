import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

log_file = open("results/Frame_level/training_log.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Trains the Multi-Task InceptionV3 model on DAiSEE dataset.

    :param model: The neural network model.
    :param train_loader: DataLoader for training data.
    :param criterion: Loss function (CrossEntropyLoss).
    :param optimizer: Optimizer (Adam, SGD, etc.).
    :param device: Device to run the training (cuda/cpu).
    :param epochs: Number of training epochs.
    """
    model = model.to(device)  # Move model to GPU/CPU

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_correct = [0, 0, 0, 0]  # Track accuracy for each affective state
        total_samples = 0

        for batch_frames, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_frames)  # List of 4 outputs (one per affective state)

            # Compute separate losses for each affective state
            loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Compute Accuracy Per Affective State
            for i in range(4):
                preds = torch.argmax(outputs[i], dim=1)  # Convert logits to predicted class index
                total_correct[i] += (preds == batch_labels[:, i]).sum().item()

            total_samples += batch_labels.size(0)

        # Compute Average Loss and Accuracy
        avg_loss = total_loss / len(train_loader)
        avg_acc = [(correct / total_samples) * 100 for correct in total_correct]

        # Print Training Stats
        log(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc}")
    log_file.close()
    print("Training Complete!")