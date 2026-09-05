import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Global signal parameters
WINDOW_SIZE = 1024
HOP_SIZE = 512

def process_wav_to_spectogram(file_path):
    """ Ingests a .wav file, enforces structural uniformity (mono and 16kHz), applies FFT signal conditioning, and transforms into clean 2D feature matrix """
    print(f"Ingesting file : {os.path.basename(file_path)}")
    
    sample_rate, raw_audio = wav.read(file_path) # Read raw signal data
    
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio.mean(axis=1) # Convert to mono if stereo (mean of channels, less info loss than just taking one channel)
        
    audio_signal = raw_audio.astype(np.float32) / 32768.0 # Covert audio samples from 16-bit integer to 32 bit float and normalizing in the range [-1, 1]
    
    total_samples = len(audio_signal)
    num_hops = (total_samples - WINDOW_SIZE) // HOP_SIZE # frame hopping capacity
    
    if num_hops <= 0:
        print(f"Skipping {os.path.basename(file_path)}, Track length too short.")
        return None, sample_rate
    
    # Initialize empty 2D matrix canvas layout : rows = freqs, cols = hops timeline
    freq_bins = WINDOW_SIZE // 2 + 1
    spectrogram_matrix = np.zeros((freq_bins, num_hops))
    
    # Step-by-Step Sliding Frame Window Transformation Loop
    for hop_idx in range(num_hops):
        start = hop_idx * HOP_SIZE
        end = start + WINDOW_SIZE
        audio_slice = audio_signal[start : end]
        windowed_slice = audio_slice * np.hanning(WINDOW_SIZE) # Apply a Hanning window to prevent transient spectral leakage artifacts (to smooth the edges of the audio frame before performing the FFT, reducing false frequencies (spectral leakage))
        fft_complex = np.fft.rfft(windowed_slice) # compute real_valued Discrete Fourier Transform (Time -> Freq Domain)
        fft_mag = np.abs(fft_complex)
        
        # adaptive gating (ref. upgraded_mictest.py)
        buffer_spike = np.max(audio_slice) - np.min(audio_slice) 
        frame_variance = np.var(fft_mag) 
        if frame_variance > 1e-6 or buffer_spike > 0.1: 
            scaled_frame = np.log1p(fft_mag * 500) 
            scaled_frame[scaled_frame < 1.5] = 0.0 
            max_scaled = np.max(scaled_frame)
            if max_scaled > 0:
                normalized_frame = scaled_frame / (max_scaled + 1e-6)
            else:
                normalized_frame = np.zeros_like(fft_mag)
        else:
            normalized_frame = np.zeros_like(fft_mag) # If background hiss or silence, keep the timeline dark
            
        spectrogram_matrix[:, hop_idx] = normalized_frame # Append completed vertical column array into the timeline matrix
        
    print(f"Processing Complete. Extracted Shape : {spectrogram_matrix.shape} (freq_bins, total_hops)")
    return spectrogram_matrix, sample_rate

def export_plot(matrix, sample_rate, output_path):
    """ Saves a high contrast visual rep. of the matrix to verify extracted features """
    plt.figure(figsize=(12, 6))
    extent = [0, matrix.shape[1], 0, sample_rate // 2]
    plt.imshow(matrix, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=0.0, vmax=1.0) # locked visual thresholds to prevent zero state color mapping bugs like before
    plt.yscale('symlog', linthresh=100)
    plt.title(f"FEATURE MAP : {os.path.basename(output_path)}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Timeline (total hops)")
    plt.colorbar(label="Normalized Energy Intensity")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plot generated and saved to {output_path}\n")
    
def main():
    TARGET_DIR = os.path.join("..", "data", "wavs")
    search_path = os.path.join(TARGET_DIR, "*.wav")
    wav_files = glob.glob(search_path)
    print(f"Found {len(wav_files)} tracks for feature extraction.\n")
    if not wav_files:
        print(f"ERROR !! No .wav files in target directory : {TARGET_DIR}\n")
        return
    for file_path in wav_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        img_path = os.path.join("..", "data", "spectrograms", "legacy", f"{base_name}_spectrogram.png")
        npy_path = os.path.join("..", "data", "matrices", "legacy", f"{base_name}.npy")
        matrix, sample_rate = process_wav_to_spectogram(file_path)
        if matrix is not None:
            np.save(npy_path, matrix)
            export_plot(matrix, sample_rate, img_path)
    print("Batch Feature Extraction completed successfully.")
    
if __name__ == "__main__":
    main()
        