#Frame level imports
#from Frame_level.dataloader import get_dataloader
#from Frame_level.inceptionnet_v3_model import InceptionV3MultiTask
#from Frame_level.test import test_model
#from Frame_level.train import train_model

#Video level imports 
##from Video_level.dataloader import get_dataloader
#from Video_level.inceptionnet_v3_model import InceptionV3MultiTask
#from Video_level.test import test_model
#from Video_level.train import train_model

#C3D import
#from C3D.dataloader import get_dataloader
#from c3d_pytorch.C3D_model import C3D
#from C3D.test import test_model
#from C3D.train import train_model

#LRCN imports
from LRCN.dataloader import get_dataloader
from LRCN.LRCN_model import LRCN
from LRCN.train import train_model
from LRCN.test import test_model


from save_pkl import save_model_pickle, load_model_pickle

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def main():


    test_label_path = "/mnt/pub/Cognitive/DAiSEE/Labels/TestLabels.csv"
    train_label_path = "/mnt/pub/Cognitive/DAiSEE/Labels/TrainLabels.csv"
    val_label_path = "/mnt/pub/Cognitive/DAiSEE/Labels/ValidationLabels.csv"

    test_dataset_path = "/mnt/pub/Cognitive/DAiSEE_Process/DataSet/Test"
    train_dataset_path = "/mnt/pub/Cognitive/DAiSEE_Process/DataSet/Train"
    val_dataset_path = "/mnt/pub/Cognitive/DAiSEE_Process/DataSet/Validation"

    train_loader = get_dataloader(train_dataset_path, train_label_path, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset_path, test_label_path, batch_size=32, shuffle=False)
    val_loader = get_dataloader(val_dataset_path, val_label_path, batch_size=32, shuffle=False)

    for images, labels,video_id in test_loader:
        print("Batch of Images Shape:", images.shape)  # Expected: (batch_size, Channel, Width, Height)
        print("Batch of Labels Shape:", labels.shape)  # Expected: (batch_size, num_classes)
        print("First Label in Batch:", labels)
        print("Video_Ids:",video_id)
        break 


    #model = InceptionV3MultiTask(4)
    #model = C3D(4)
    model = LRCN(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    train_model(model, train_loader, criterion, optimizer, device, epochs=5)


    #model.load_state_dict(torch.load("/home/hbml-syeramil/Documents/spring_proj/results/c3d_full_train/c3d_model.pth"))
    #model = load_model_pickle("best_model.pkl", device)
    model = model.to(device=device)
    test_model(model, test_loader, criterion, device)

    torch.save(model.state_dict(),"lrcn_model.pth")
    save_model_pickle(model, path="lrcn_model.pkl")


if __name__ == "__main__":
    main()