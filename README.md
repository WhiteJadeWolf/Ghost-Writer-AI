# Ghost-Writer AI : Audio-to-MIDI Transcription Pipeline

A machine learning and Digital Signal Processing (DSP) pipeline that transcribes raw polyphonic piano audio into standard MIDI files. 

This project explores the mathematical limits of linear neural networks on time-series audio by combining a PyTorch Autoencoder with a custom DSP transcription engine, specifically highlighting the critical data engineering shift from linear STFT (Short-Time Fourier Transform) to logarithmic CQT (Constant-Q Transform).

---

## The Architecture

### Phase 1 : Feature Extraction (Constant-Q Transform)
Standard STFT extracts audio in linear Hertz, which visually and mathematically crushes low-frequency bass notes together. To align the data with the physics of human hearing and Western music, this pipeline relies natively on CQT:
* **Nyquist Optimization:** Ingests standard `.wav` files downsampled to **22050 Hz**. This locks the Nyquist limit precisely above the highest acoustic piano frequencies (~4186 Hz), naturally filtering out high-frequency transient noise and room hiss before it ever hits the model.
* **Logarithmic Mapping:** Computes an 84-bin Constant-Q Transform (`n_bins=84`), perfectly mapping the raw audio matrix to the exact 84 physical keys of a piano (C1 to B7). 

### Phase 2 : Neural Denoising (Linear Autoencoder)
* A PyTorch-based linear autoencoder compresses the 84-node CQT array down to a 32-node bottleneck before reconstructing it.
* By intentionally bottlenecking the network, it acts as an extreme spatial and temporal noise filter. It strips out mic bumps, string resonances, and background bleed, forcing the network to output only the dominant fundamental frequencies.

### Phase 3 : The "Smart" Transcriber (DSP Engine)
Raw AI output lacks acoustic physics logic. The `3_transcribe.py` script bridges this gap by scanning the denoised matrix millisecond-by-millisecond and applying three hard-coded DSP filters:
1. **Spatial Peak Picking (Anti-Bleed):** Enforces local maxima logic (`val > neighbors`) to prevent loud keys from bleeding into adjacent CQT bins and triggering cluster chords.
2. **Harmonic Suppression (Anti-Ghost Notes):** Instantly muffles natural mathematical overtones (Octave +12, Perfect Fifth +7, +19) the exact millisecond a fundamental note is struck.
3. **Time Gating / Debouncing:** Tracks temporal duration, deleting any micro-spikes shorter than 150ms to ensure only deliberate keystrokes are written to the final `.mid` file.

---

## Repository Structure

The workspace is strictly segregated to separate active code from R&D sandboxes and heavy dataset matrices.

```text
Ghost-Writer/
├── src/                    # Production Pipeline
│   ├── dataset.py          # PyTorch DataLoader logic
│   ├── model.py            # Neural network architecture (SpectrogramAutoencoder)
│   ├── 1_generate_cqt.py   # Ingests WAV, applies CQT, outputs .npy matrices & plots
│   ├── 2_train.py          # Model training loop (MSE Loss, Adam Optimizer)
│   └── 3_transcribe.py     # DSP filters and MIDI generation
│
├── legacy/                 # R&D Archive
│   ├── legacy_model.py     # Original 513-bin model architecture
│   ├── 1_generate_spectrogram.py # Failed linear STFT attempts
│   └── 3_transcriber_legacy.py   # Legacy STFT-to-MIDI math
│
└── data/                   # Data directory (Contents excluded via .gitignore)
    ├── wavs/               # Raw audio input
    ├── matrices/           # Extracted .npy tensors
    ├── midis/              # Final transcription output
    └── saved_models/       # .pt weight files
```
---

## Quick Start & Execution

### 1. Install Dependencies
```
pip install -r requirements.txt
```
### 2. Generate Musical Features <br>
**Place your audio files in `data/wavs/` and run :**
```
python src/1_generate_cqt.py
```
*Outputs highly compressed .npy matrices and digital piano-roll visual plots.*


### 3. Train the Denoising Model
```
python src/2_train.py
```
*Trains the PyTorch Autoencoder and saves dynamically versioned weights to `data/saved_models/`.*


### 4. Generate MIDI
```
python src/3_transcribe.py
```
*Runs the AI reconstruction and DSP filters, outputting a tempo-synced .mid file to `data/midis/smart/`.*

---
