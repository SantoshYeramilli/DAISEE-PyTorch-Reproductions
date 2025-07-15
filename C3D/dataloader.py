import torch
import torch.utils.data as data
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from natsort import natsorted 
from torch.utils.data import Dataset, DataLoader # Ensures frames are loaded in correct order

class loaddata(data.Dataset):
    def __init__(self, csv_file, video_dir, transform=None, num_frames=16):
        """
        Args:
            csv_file (str): Path to the CSV file containing labels.
            video_dir (str): Path to the directory containing video frame folders.
            transform (callable, optional): Transformations for images.
            num_frames (int): Number of frames per video.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.video_label_dict = {row.iloc[0]: row.iloc[1:].tolist() for _, row in self.csv_file.iterrows()}
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.data = []

        for video_folder in os.listdir(video_dir):
            if video_folder.startswith('.'):  # Ignore hidden files
                continue

            video_folder_path = os.path.join(video_dir, video_folder)
            if not os.path.isdir(video_folder_path):  # Ensure it's a directory
                continue

            labels = self.video_label_dict.get(video_folder + ".avi") or self.video_label_dict.get(video_folder + ".mp4")
            if labels is None:
                continue

            # Get all frame file paths, sorted in order
            frames_list = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".jpg")]
            frames_list = natsorted(frames_list)  # Sort frames in order

            if len(frames_list) < 60:
                continue  

            # Store video as sequences of `num_frames`
            # If more than 16 frames, extract multiple overlapping sequences
            for i in range(0, len(frames_list) - self.num_frames + 1, 8):
                frame_sequence = frames_list[i:i+self.num_frames]
                self.data.append((frame_sequence, labels, video_folder))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_sequence, label, video_id = self.data[idx]  # Retrieve video_id

        frames = []
        for frame_path in frame_sequence:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # Convert to tensor: (16, 3, 112, 112)
        video_tensor = torch.stack(frames)  # Stack into shape (16, 3, 112, 112)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Convert to (3, 16, 112, 112) for C3D

        label = torch.tensor(label, dtype=torch.long)

        return video_tensor, label, video_id  # Now returns a sequence of frames

def get_dataloader(frame_paths, labels, batch_size = 128, shuffle = False):
    transform = transforms.Compose([
        #transforms.Resize((112, 112)),
        transforms.Resize((224,224)), #for LRCN/Alexnet
        transforms.ToTensor(),
        ])
   
    dataset = loaddata(csv_file=labels, video_dir=frame_paths,transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize for C3D
    #transforms.Resize((224,224)),
    transforms.ToTensor(),  # Convert to tensor
    ])
    test_label_path = "/mnt/pub/CognitiveDataset/DAiSEE/Labels/TestLabels.csv"
    train_label_path = "/mnt/pub/CognitiveDataset/DAiSEE/Labels/TrainLabels.csv"
    val_label_path = "/mnt/pub/CognitiveDataset/DAiSEE/Labels/ValidationLabels.csv"

    test_dataset_path = "/mnt/pub/CognitiveDataset/DAiSEE_Process/DataSet/Test"
    train_dataset_path = "/mnt/pub/CognitiveDataset/DAiSEE_Process/DataSet/Train"
    val_dataset_path = "/mnt/pub/CognitiveDataset/DAiSEE_Process/DataSet/Validation"
    # Create dataset and dataloader
    train_dataset = loaddata(train_label_path, train_dataset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    # Fetch a batch
    data_iter = iter(train_loader)
    videos, labels, video_ids = next(data_iter)

    print(f"Batch shape: {videos.shape}")  # Expected: (8, 3, 16, 112, 112)
    print(f"Labels shape: {labels.shape}")  # Expected: (8, num_classes)
    print(f"Video IDs: {video_ids}")  # Prints video folder names