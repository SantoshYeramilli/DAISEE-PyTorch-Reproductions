import torch
import torch.nn as nn
import torchvision.models as models

class InceptionV3MultiTask(nn.Module):
    def __init__(self, num_classes=4):
        """
        InceptionV3 for Multi-Label, Multi-Class Classification.
        Each sample gets 4 independent predictions (one per affective state), with 4 possible class scores per state.
        """
        super(InceptionV3MultiTask, self).__init__()

        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove Inception's final classifier

        # Define 4 separate classification heads (one for each affective state)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)  # Predict 4 classes per task
            ) for _ in range(4)  # 4 affective states
        ])

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "fc" in name or "Mixed_7b" in name or "Mixed_7c" in name:
                param.requires_grad = True

    def forward(self, x):
        """
        :param x: Input shape (batch_size, 3, 299, 299)
        :return: List of 4 logits tensors, each of shape (batch_size, num_classes)
        """
        output = self.model(x)  

        if isinstance(output, tuple):  # InceptionV3 has auxiliary outputs
            final_output, _ = output  # We only use the final output
        else:
            final_output = output

        # Pass extracted features through 4 classification heads
        return [task_head(final_output) for task_head in self.task_heads]
