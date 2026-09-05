import torch.nn as nn

class SpectrogramAutoencoder(nn.Module):
    
    def __init__(self, input_dim=84, bottleneck_dim=32):
        super(SpectrogramAutoencoder, self).__init__()
        
        # Encoder [ 84 (piano keys) -> 32 ]
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.Linear(48, bottleneck_dim),
                nn.ReLU()
        )
        
        # Decoder [ 32 freq. bins -> 84 ]
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # output bounded between 0 and 1
        )
        
    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed