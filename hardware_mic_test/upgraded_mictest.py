import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# audio stream parameters for frequency extraction
SAMPLE_RATE = 44100 # samples per second
WINDOW_SIZE = 1024 # number of audio points to show on screen at once (power of 2 for FFT)
HOP_SIZE = 512 # number of audio points to shift for each new frame (how much we slide the window)
NUM_FRAMES = 100 # number of time frames to show on screen


# matrix to hold the rolling spectogram data
spectogram_data = np.zeros((WINDOW_SIZE // 2 + 1, NUM_FRAMES))
# FFT outputs symmetrical/mirrored data, // 2 discards the redundant top half of mirrored freqs, + 1 retains index 0, representing the 0Hz DC offset component
# 2nd parameter NUM_FRAMES determines how many time frames we want to show on the spectogram at once, creating a rolling window effect dropping the oldest frame on the left, appends new frame to the right\


# setting up plot window
fig, ax = plt.subplots(figsize=(10, 6))
extent = [0, NUM_FRAMES, 0, SAMPLE_RATE // 2] # x-axis from 0 to NUM_FRAMES, y-axis from 0 to Nyquist frequency (half the sample rate)
im = ax.imshow(spectogram_data, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=0.0, vmax=1.0) # display the spectogram as an image
ax.set_yscale('symlog', linthresh=100) # set y-axis to symmetric log scale (normal log doesnt include 0Hz, symlog allows for small linear region near zero)
# linthresh=100 means freq <= 100Hz -> linear scale, freq > 100Hz -> log scale, prevents graph from getting distorted near 0Hz
ax.set_title("Live Audio Spectogram")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (Rolling Frames)")


# audio processing buffer
audio_buffer = np.zeros(WINDOW_SIZE)


# this function gets called automatically every time the mic hears a sound chunk
def audio_callback(indata, frames, time, status):
    global audio_buffer, spectogram_data
    
    mono_data = indata[:, 0].flatten() # extract the first channel (mono) from the input data
    
    audio_buffer = np.roll(audio_buffer, -frames) # Shift Buffer Out
    audio_buffer[-frames:] = mono_data # Slide new audio data in
    
    # computing the magnitude spectrum (Short-Time Fourier Transform)
    windowed_data = audio_buffer * np.hanning(WINDOW_SIZE) # Apply Hanning Window to reduce spectral leakage
    fft_complex = np.fft.rfft(windowed_data) # Compute the FFT (real-valued input, so we use rfft which returns only the non-negative frequencies) (time domain to frequency domain)
    fft_mag = np.abs(fft_complex) # Get the magnitude of the FFT (discard phase information)
    
    """# Dynamic Normalization (prevents silent drops and caps data scale)
    max_val = np.max(fft_mag)
    if max_val > 0.05: # NOISE THRESHOLD : Only activates if the sound is louder than the baseline room hiss
        scaled_frame = np.log1p(fft_mag * 10) # log scale the magnitude for better visualization, log1p(x) = log(1 + x) to avoid log(0) issues
        normalized_frame = scaled_frame / (np.max(scaled_frame) + 1e-6) # Normalize after scaling to maintain a native 0.0 - 1.0 range
    else:
        # Absolute silence handling to prevent divide-by-zero exceptions
        normalized_frame = np.zeros_like(fft_mag)"""
        
    buffer_spike = np.max(mono_data) - np.min(mono_data) # simple measure of how much the audio signal is changing in this frame, high spike means a sudden loud sound, low spike means stable background noise
    
    frame_variance = np.var(fft_mag) # compute the mathematical variance of the frequency magnitudes
    
    if frame_variance > 1e-5 or buffer_spike > 0.1: # VARIANCE THRESHOLD GATE : Ignore stable noise, catch sudden changes 
        scaled_frame = np.log1p(fft_mag * 500) # log scale the magnitude for better visualization, log1p(x) = log(1 + x) to avoid log(0) issues
        scaled_frame[scaled_frame < 1.5] = 0.0 # NOISE GATE : Set low-level noise to zero to clean up the spectogram, threshold determined empirically
        max_scaled = np.max(scaled_frame)
        if max_scaled > 0:
            normalized_frame = scaled_frame / ((max_scaled + 1e-6))
        else:
            normalized_frame = np.zeros_like(fft_mag)
    else:
        normalized_frame = np.zeros_like(fft_mag) # If the sound texture is perfectly stable (just background hiss), keep it dark
        
        
    # roll the spectogram matrix and insert the new column
    spectogram_data = np.roll(spectogram_data, -1, axis=1)
    spectogram_data[:, -1] = normalized_frame
    
    
# this function updates the graph animation frame by frame
def update_plot(frame):
    im.set_array(spectogram_data) # update the image data with the latest spectogram data
    return [im] # returning the image object to let mplib know which objects were modified so as to update the plot efficiently


# starting the live mono-audio microphone stream
stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=HOP_SIZE, device=1, callback=audio_callback)
with stream:
    ani = FuncAnimation(fig, update_plot, interval=30, blit=True, cache_frame_data=False) # start the animation loop
    plt.show()