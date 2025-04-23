from torchvision import models
import torch.nn as nn
import torch

class FaceRecognizer(nn.Module):
    def __init__(self):
        super(FaceRecognizer, self).__init__()
        self.convolution = models.efficientnet_b0(pretrained=True)
        self.convolution.fc = nn.Identity() # removes fully connected dense layer
        self.classifier = nn.Sequential(

            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.classifier(x)
        return x

def load_classifier():
    model_path = "classifier_model.pth"  # Path to your model file
    model = FaceRecognizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path if model_path else "classifier.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model
