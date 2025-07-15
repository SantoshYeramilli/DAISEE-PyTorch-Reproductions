import torch
from tqdm import tqdm

log_file = open("test_log_emotionnet.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

def test_model(model, test_loader, criterion, device):
    """
    Evaluates the trained Multi-Task InceptionV3 model on a test dataset.

    :param model: The trained neural network model.
    :param test_loader: DataLoader for test data.
    :param criterion: Loss function (CrossEntropyLoss).
    :param device: Device to run evaluation (cuda/cpu).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = [0, 0, 0, 0]  # Track accuracy for each affective state
    total_samples = 0

    with torch.no_grad():  # No gradient computation during testing
        for batch_frames, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            outputs = model(batch_frames)  # List of 4 outputs (one per affective state)

            # Compute separate losses for each affective state
            loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            total_loss += loss.item()

            # Compute Accuracy Per Affective State
            for i in range(4):
                preds = torch.argmax(outputs[i], dim=1)  # Convert logits to predicted class index
                total_correct[i] += (preds == batch_labels[:, i]).sum().item()

            total_samples += batch_labels.size(0)

    # Compute Average Loss and Accuracy
    avg_loss = total_loss / len(test_loader)
    avg_acc = [(correct / total_samples) * 100 for correct in total_correct]

    # Print Test Results
    log(f"\n Test Loss: {avg_loss:.4f}")
    log(f"Test Accuracy per Affective State: {avg_acc}")
    log_file.close()
    return avg_loss, avg_acc  # Return metrics for further analysis