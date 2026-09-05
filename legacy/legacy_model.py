import torch.nn as nn

class SpectrogramAutoencoder(nn.Module):
    
    def __init__(self, input_dim=513, bottleneck_dim=32):
        super(SpectrogramAutoencoder, self).__init__()
        
        # Encoder [ 513 freq. bins -> 32 ]
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, bottleneck_dim),
                nn.ReLU()
        )
        
        # Decoder [ 32 freq. bins -> 513 ]
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # output bounded between 0 and 1
        )
        
    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed