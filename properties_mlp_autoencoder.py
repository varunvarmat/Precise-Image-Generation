import torch
import torch.nn as nn
class MLP_Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.block(x)
class MLP_Autoencoder(nn.Module):
    def __init__(self, N , schedule):
        super(MLP_Autoencoder, self).__init__()

        # Encoder
        assert N==len(schedule)-1
        self.encoder = nn.ModuleList()
        self.input_dim=None
        self.output_dim=None
        self.N=N
        for i in range(N):
            self.input_dim=schedule[i]
            self.output_dim=schedule[i+1]
            self.encoder.append(
                MLP_Block(self.input_dim,self.output_dim)
            )
        schedule=schedule[::-1]

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(N):
            self.input_dim=schedule[i]
            self.output_dim=schedule[i+1]
            self.decoder.append(
                MLP_Block(self.input_dim,self.output_dim)
            )
        
    def forward(self, x):
        out=x
        for layer in self.encoder:
            out = layer(out)
        latent=out
        for layer in self.decoder:
            out = layer(out)
        return out, latent