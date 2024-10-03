import torch
import torch.nn as nn

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: [32, 112, 112]
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: [64, 56, 56]
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output size: [128, 28, 28]
        )
        self.lstm = nn.LSTM(128 * 28 * 28, 256, batch_first=True)  # LSTM input size
        self.fc = nn.Linear(256, num_classes)  # Final output layer

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        # Reshape for CNN
        x = x.view(-1, c, h, w)  # Flatten the batch of frames for CNN
        x = self.cnn(x)  # Apply CNN
        x = x.reshape(batch_size, num_frames, -1)  # Reshape to (batch_size, num_frames, features)
        x, _ = self.lstm(x)  # Apply LSTM
        x = x[:, -1, :]  # Get the output of the last frame
        x = self.fc(x)  # Final output
        return x

