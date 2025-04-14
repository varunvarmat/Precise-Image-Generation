import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, z_dim=64, latent_dim=128):
        super().__init__()
        # Encoder layers to compress the image down to the z_dim size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, z_dim, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(z_dim),
            nn.ReLU(True)
        )
        
        # Bottleneck MLP to compress features
        self.flatten = nn.Flatten()
        self.compress = nn.Sequential(
            nn.Linear(z_dim * 4 * 4, 512),  # Flatten to latent space
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.compress(x)  # Compressed latent vector
        return z  # [B, latent_dim]

class Postnet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.refine(x)  # Residual enhancement

class ImageDecoder(nn.Module):
    def __init__(self, z_dim=64, latent_dim=128):
        super().__init__()
        
        # Decoder layers for reconstructing the image from the latent vector
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, z_dim * 4 * 4),  # Expand to z_dim * 4 * 4
        )

        self.unflatten = nn.Unflatten(1, (z_dim, 4, 4))  # Unflatten to match the image dimensions
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.Sigmoid() 
        )

        self.postnet = Postnet(in_channels=3)

    def forward(self, z):
        z = self.expand(z)  # Expand to z_dim * 4 * 4
        z = self.unflatten(z)  # Unflatten to match the image dimensions
        x_reconstructed = self.decoder(z)  # Reconstruct image
        x_reconstructed = self.postnet(x_reconstructed)
        return x_reconstructed


class ImageAutoencoder(nn.Module):
    def __init__(self, z_dim=64, latent_dim=128):
        super().__init__()
        self.encoder = ImageEncoder(z_dim=z_dim, latent_dim=latent_dim)
        self.decoder = ImageDecoder(z_dim=z_dim, latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)  # Get compressed latent vector
        x_reconstructed = self.decoder(z)  # Reconstruct image from latent vector
        return x_reconstructed, z  # Return both the reconstruction and latent vector

