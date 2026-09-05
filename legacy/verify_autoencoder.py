import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from legacy.legacy_model import SpectrogramAutoencoder

DATA_DIR = os.path.join("..", "data")
MODEL_WEIGHTS = os.path.join("..", "data", "saved_models", "legacy", "autoencoder_legacy.pt")

def verify_and_plot():
    
    # necessary file checks
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR ! model weights file '{MODEL_WEIGHTS}' not found.")
        return
    
    npy_files = glob.glob(os.path.join(DATA_DIR, "matrices", "legacy", "*.npy"))
    if not npy_files:
        print(f"ERROR ! No processed data matrices (.npy) file found")
        return
    
    target_file = None
    for file in npy_files:
        matrix = np.load(file)
        if np.max(matrix) > 0: # skip silence
            target_file = file
            break
        
    if not target_file:
        target_file = npy_files[0]
        matrix = np.load(target_file)
        
    print(f"Target Track selected : {os.path.basename(target_file)}")
    
    # Reshape and condition input data exactly like the dataset loader does
    # Original Shape : (513, total_hops) -> Transpose : (total_hops, 513)
    original_matrix = matrix.copy()
    input_data = matrix.T.astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    
    # Load model and weights
    model = SpectrogramAutoencoder(input_dim=513, bottleneck_dim=32)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print(f"Model layers successfully populated with weights from '{MODEL_WEIGHTS}'")
    except Exception as e:
        print(f"ERROR ! Failed to load model weights : {e}")
        return
    
    model.eval()
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
        reconstructed_matrix = reconstructed_tensor.numpy().T # Transpose back to original shape (513, total_hops)
        
    print(f"Inference Completed. Output Shape : {reconstructed_matrix.shape}")
    
    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)
    
    # Left Plot : Original
    im0 = axes[0].imshow(original_matrix, aspect='auto', origin='lower', cmap='inferno', vmin=0.0, vmax=1.0)
    axes[0].set_title("Original Input Spectrogram (X)")
    axes[0].set_ylabel("Frequency Bins (0 - 513)")
    axes[0].set_xlabel("Timeline (Hops)")
    
    # Right Plot : Reconstructed
    im1 = axes[1].imshow(reconstructed_matrix, aspect='auto', origin='lower', cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title("Autoencoder Reconstruction Output (y_pred)")
    axes[1].set_xlabel("Timeline (Hops)")
    
    # shared colorbar overlay
    fig.colorbar(im1, ax=axes.ravel().tolist(), label='Normalized Energy Intensity')
    
    output_png = os.path.join(DATA_DIR, "plots", "legacy", "verification_comparison.png")
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    print(f"Comparative plot generated successfully and saved to : {output_png}")
    
if __name__ == '__main__':
    verify_and_plot()