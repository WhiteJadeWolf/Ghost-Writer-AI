import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Global signal parameters
SAMPLE_RATE = 22050 # optimal resolution for musical Constant-Q Transform
HOP_SIZE = 512

def process_wav_to_cqt(file_path):
    print(f"Ingesting file : {os.path.basename(file_path)}")
    audio_signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, mono=True) # loading audio natively at 22050Hz
    cqt_matrix = np.abs(librosa.cqt(audio_signal, sr=sample_rate, hop_length=HOP_SIZE, fmin=librosa.note_to_hz('C1'), n_bins=84)) # computing CQT, 84 bins = 7 full octaves (12 semitones per octave), starting at C1 (note 24)
    cqt_db = librosa.amplitude_to_db(cqt_matrix, ref=np.max) # converting into human decibel (log) scale
    cqt_norm = (cqt_db - np.min(cqt_db)) / (np.max(cqt_db) - np.min(cqt_db) + 1e-6) # normalize between 0.0 and 1.0
    cqt_norm[cqt_norm < 0.4] = 0.0 # noise gating filter
    print(f"Processing Complete. Extracted Shape : {cqt_norm.shape} (84 musical notes, total_hops)")
    return cqt_norm

def export_plot(matrix, output_path):
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect='auto', origin='lower', cmap='inferno', vmin=0.0, vmax=1.0) # no symlog since CQT bins already perfectly spaced by musical pitch
    plt.title(f"CQT FEATURE MAP : {os.path.basename(output_path)}")
    plt.ylabel("CQT Bins (0 = C1, 83 = B7)")
    plt.xlabel("Timeline (total hops)")
    plt.colorbar(label="Normalized Energy Intensity")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plot generated and saved to {output_path}\n")

def main():
    TARGET_DIR = os.path.join("..", "data", "wavs")
    wav_files = glob.glob(os.path.join(TARGET_DIR, "*.wav"))
    print(f"Found {len(wav_files)} tracks for musical feature extraction.\n")
    
    for file_path in wav_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        npy_path = os.path.join("..", "data", "matrices", f"{base_name}.npy")
        img_path = os.path.join("..", "data", "spectrograms", f"{base_name}_spectrogram.png")
        
        matrix = process_wav_to_cqt(file_path)
        if matrix is not None:
            np.save(npy_path, matrix)
            print(f"Saved musical CQT matrix to {npy_path}")
            export_plot(matrix, img_path)
            
    print("Musical Feature Extraction completed successfully.")

if __name__ == "__main__":
    main()