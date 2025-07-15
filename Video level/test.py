import torch
from collections import defaultdict
from tqdm import tqdm

log_file = open("results/video_level/test_log_taskhead_video.txt", "w")

def log(msg):
    print(msg)
    log_file.write(msg +"\n")
    log_file.flush()

def test_model(model, test_loader, criterion, device):
    """
    Evaluates the Multi-Task InceptionV3 model on a test dataset,
    aggregating accuracy per video.

    :param model: The trained neural network model.
    :param test_loader: DataLoader for test data.
    :param criterion: Loss function (CrossEntropyLoss).
    :param device: Device to run evaluation (cuda/cpu).
    """
    model.eval()
    total_loss = 0
    video_accuracy = defaultdict(lambda: [[], [], [], []])  # Stores accuracy per affective state per video
    video_counts = defaultdict(int)  # Tracks number of frames per video

    with torch.no_grad():
        for batch_frames, batch_labels, video_ids in tqdm(test_loader, desc="Testing"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            outputs = model(batch_frames)  # List of 4 tensors (one per affective state)

            # Compute loss
            loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
            total_loss += loss.item()

            # Compute Accuracy Per Frame and Store by `video_id`
            for i in range(4):
                preds = torch.argmax(outputs[i], dim=1)  # Convert logits to predicted class index
                correct = (preds == batch_labels[:, i]).cpu().numpy()

                for vid, correct_pred in zip(video_ids, correct):
                    video_accuracy[vid][i].append(correct_pred)  # Store frame accuracy per video ID

                video_counts[vid] += 1  # Track frames per video

    # Compute Per-Video Accuracy
    video_final_accuracy = {}
    for vid, acc_list in video_accuracy.items():
        per_state_accuracy = [sum(state_acc) / len(state_acc) for state_acc in acc_list]
        video_final_accuracy[vid] = per_state_accuracy

    # Compute Overall Accuracy Across All Videos
    avg_loss = total_loss / len(test_loader)
    avg_acc = [sum(state_acc) / len(video_final_accuracy) for state_acc in zip(*video_final_accuracy.values())]

    # Print Final Metrics
    log(f"\nTest Loss: {avg_loss:.4f}")
    log(f"Average Test Accuracy per Affective State: {avg_acc}")
    #print(f"Per-Video Accuracy: {video_final_accuracy}")
    log_file.close()
    return avg_loss, avg_acc, video_final_accuracy
