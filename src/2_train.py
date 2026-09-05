import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LocalAudioSandboxDataset

from model import SpectrogramAutoencoder

def train_model():
    
    # hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DATA_DIR = os.path.join("..", "data", "matrices")
    
    # loading dataset
    try:
        dataset = LocalAudioSandboxDataset(data_dir=DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"ERROR ! failed to load dataset : {e}")
        return
    
    model = SpectrogramAutoencoder(input_dim=84, bottleneck_dim=32)
    criterion = nn.MSELoss() # MSE calculates reconstruction accuracy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training model over {len(dataset)} hop slices for {EPOCHS} epochs...\n")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_X, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X) # forward pass (compress -> decompress)
            loss = criterion(outputs, batch_X) # (output vs original input)
            # backward pass (optimizing weights)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Reconstruction MSE Loss : {epoch_loss:.6f}")
        
    print("Operation Successful. Saving model weights...")
    
    model_save_dir = os.path.join("..", "data", "saved_models")
    os.makedirs(model_save_dir, exist_ok = True)
    save_path = os.path.join(model_save_dir, "autoencoder.pt")
    
    torch.save(model.state_dict(), save_path)
    print(f"Trained weights saved as : {save_path}")
    
if __name__ == '__main__':
    train_model()