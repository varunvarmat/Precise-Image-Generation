import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 128x128x16
            nn.BatchNorm2d(16),  # Normalize activations
            nn.ReLU(),
            nn.Dropout(0.1),  # Small dropout to prevent overfitting

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 64x64x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16 * 128, 1024),  
            nn.ReLU(),
            nn.Dropout(0.3),  # More dropout in fully connected layers
            nn.Linear(1024, 128)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Apply Conv Layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)  # Apply Fully Connected Layers
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),  
            nn.ReLU(),
            nn.Dropout(0.3),  # Fully connected dropout
            nn.Linear(1024, 16*16*128),  
            nn.ReLU()
        )

        # Transposed Convolution Layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()  # Normalize output to [0,1] range (optional, useful if working with images)
        )

    def forward(self, x):
        x = self.fc_layers(x)  # Expand with FC layers
        x = x.view(x.size(0),128, 16, 16)  # Reshape to required
        x = self.deconv_layers(x)  # Apply Transposed Convolutions
        return x



class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()

        # Encoder
        self.encoder= Encoder()
        self.decoder = Decoder()
        
        
    def forward(self, x):
        latent=self.encoder(x)
        reconstructed=self.decoder(latent)
        return reconstructed, latent