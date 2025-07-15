import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class loaddata(torch.utils.data.Dataset):
  def __init__(self, csv_file, video_dir, transform):
    self.csv_file = pd.read_csv(csv_file)
    self.video_label_dict = {row.iloc[0]: row.iloc[1:].tolist() for _, row in self.csv_file.iterrows()}
    self.video_dir = video_dir
    self.transform = transform

    self.data = []

    for video_folder in os.listdir(video_dir):  
      if video_folder.startswith('.'):  
        continue

      video_folder_path = os.path.join(video_dir, video_folder)
      if not os.path.isdir(video_folder_path): 
        continue
      if os.path.isdir(video_folder_path):
        labels = self.video_label_dict.get(video_folder + ".avi") or self.video_label_dict.get(video_folder + ".mp4")
        if labels is None:
          continue
        video_id = video_folder
        for frames in os.listdir(video_folder_path):
          if frames.endswith(".jpg"):
            frames_path = os.path.join(video_folder_path, frames)
            self.data.append((frames_path, labels, video_id))
                            
           
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    frame_path, label, video_id = self.data[idx]
    image = Image.open(frame_path)
    image = self.transform(image)
    label = torch.tensor(label, dtype=torch.long)


    return image, label, video_id

def get_dataloader(frame_paths, labels, batch_size = 128, shuffle = False):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
   
    dataset = loaddata(csv_file=labels, video_dir=frame_paths,transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader
    