import torch
import torch.nn as nn
import torchvision.models as models

class LRCN(nn.Module):
    def __init__(self, num_classes=4, hidden_size=512, num_layers=1, feature_dim=4096, sequence_length=16):
        super(LRCN, self).__init__()

        
        self.caffenet = models.alexnet(pretrained=True)  # Load CaffeNet (similar to AlexNet)
        self.caffenet.classifier = nn.Sequential(*list(self.caffenet.classifier.children())[:-1])  # Remove last FC layer

        #  LSTM for Sequence Learning
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Separate Task Heads for Affective States
        self.task_heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(4)])

    def forward(self, x):
        batch_size, C, timesteps, H, W = x.shape  # Expecting (batch, 3, 16, 224, 224)

        #  Permute dimensions so that the sequence dimension comes first
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, timesteps, 3, 224, 224)
        
        #  Flatten time dimension before passing to CaffeNet
        x = x.reshape(batch_size * timesteps, C, H, W)  # Shape: (batch * timesteps, 3, 224, 224)
        
        # Extract Features Per Frame Using CaffeNet
        features = self.caffenet(x)  # Feature shape: (batch * timesteps, 4096)

        # Reshape back into (batch, timesteps, feature_dim)
        features = features.view(batch_size, timesteps, -1)  # Shape: (batch, 16, 4096)

        # Pass Through LSTM
        lstm_out, _ = self.lstm(features)  # Output shape: (batch, timesteps, hidden_size)
        

        # Compute Separate Predictions for Each Affective State
        outputs = [head(lstm_out) for head in self.task_heads]  # Output shape: (batch, 4, num_classes)

        return outputs # Shape: (batch, 4, num_classes)


