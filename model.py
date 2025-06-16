import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # (B, 32, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 32, 32, 32)
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 64, 16, 16)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # (B, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2),   # (B, 1, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, input):
        latent = self.encoder(input)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

