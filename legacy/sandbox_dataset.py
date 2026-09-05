import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LocalAudioSandboxDataset(Dataset):
    def __init__(self, data_dir=os.path.join("..", "data", "matrices", "legacy")):
        self.data_dir = data_dir
        self.matrix_files = glob.glob(os.path.join(data_dir, "*.npy"))
        if not self.matrix_files:
            raise FileNotFoundError(
                f"No processed data found in '{data_dir}'. "
                f"Please run 'python generate_spectrogram.py' first."
            )
            
        # load and stack all the individual file matrices into one massive timeline
        all_hops = []
        for file_path in self.matrix_files:
            matrix = np.load(file_path) # shape : (513, total_hops)
            all_hops.append(matrix.T) # transpose to shape (total_hops, 513) so each row is one slice of time
            
        # combine everything into a single continuous numpy array
        self.data = np.vstack(all_hops).astype(np.float32)
        print(f"Successfully initialized {self.data.shape[0]} musical hop slices.")

    def __len__(self):
        return self.data.shape[0] # total slices

    def __getitem__(self, idx):
        hop_slice = self.data[idx] # fetch a single slice at a specific moment in time
        tensor_x = torch.from_numpy(hop_slice) # convert into tensor
        return tensor_x, tensor_x # self-supervised learning (target y is same as input X)

if __name__ == "__main__":
    try:
        ds = LocalAudioSandboxDataset()
        print(f"Success. Sample slice shape: {ds[0][0].shape}")
    except Exception as e:
        print(e)