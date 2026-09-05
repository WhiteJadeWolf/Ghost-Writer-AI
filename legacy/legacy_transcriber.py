# FOR STFT
import os
import glob
import numpy as np
import torch
from midiutil import MIDIFile

from legacy.legacy_model import SpectrogramAutoencoder

MODEL_WEIGHTS = os.path.join("..", "data", "saved_models", "legacy", "autoencoder_legacy.pt")
npy_files = glob.glob(os.path.join("..", "data", "matrices", "legacy", "*.npy"))

SAMPLE_RATE = 44100
HOP_SIZE = 512
WINDOW_SIZE = 1024

def freq_to_midi(freq):
    if freq <= 0:
        return None
    midi_num = round(69 + 12 * np.log2(freq / 440.0))
    if 21 <= midi_num <= 108: # 88-key piano boundaries
        return midi_num
    return None

def transcribe():
    filtered_files = [f for f in npy_files if "verification" not in f]
    if not filtered_files:
        print(f"ERROR ! No data matrices (.npy) found to transcribe")
        return
    
    target_file = filtered_files[0]
    print(f"Target Track selected : {os.path.basename(target_file)}")
    matrix = np.load(target_file)
    
    model = SpectrogramAutoencoder(input_dim=513, bottleneck_dim=32)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print("Trained AI weights loaded successfully.")
    else:
        print(f"ERROR ! Weights file : {MODEL_WEIGHTS} not found")
        return
    
    model.eval()
    input_tensor = torch.from_numpy(matrix.T.astype(np.float32))
    with torch.no_grad():
        clean_matrix = model(input_tensor).numpy().T # Shape : (513, total_hops)
        
    sec_per_hop = HOP_SIZE / SAMPLE_RATE 
    total_hops = clean_matrix.shape[1]
    
    active_notes = {}    # { midi_note : (start_time, initial_velocity) }
    completed_notes = [] # (midi_note, start_time, duration, velocity)
    
    NOTE_THRESHOLD = 0.45    # base volume threshold
    TARGET_PLAYBACK_BPM = 80 # Adjusted down from 120 to match slow/romantic pacing
    
    for hop_idx in range(total_hops):
        hop_time = hop_idx * sec_per_hop
        column = clean_matrix[:, hop_idx]
        
        # peak finder with local suppression
        peaks = []
        for i in range(1, len(column) - 1):
            if column[i] > column[i - 1] and column[i] > column[i + 1] and column[i] > NOTE_THRESHOLD:
                peaks.append((i, column[i])) # Store bin index and its intensity value
                
        # Sort peaks from loudest to quietest
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        current_frame_notes = {}
        
        # Harmonic filtering : if a note is selected, don't let nearby rows double-fire
        used_midi_notes = set()
        for bin_idx, intensity in peaks:
            freq = (bin_idx * SAMPLE_RATE) / WINDOW_SIZE
            midi_note = freq_to_midi(freq)
            if midi_note is not None and midi_note not in used_midi_notes:
                velocity = int(intensity * 127) # map the 0-1 AI intensity directly to standard 0-127 midi volume
                velocity = max(40, min(velocity, 120)) # keep it in a natural human hearing range
                
                current_frame_notes[midi_note] = velocity
                used_midi_notes.add(midi_note)
                
                # suppress immediate octaves/harmonics (+12, +19, +24 notes up) to kill ghost tracking
                for ghost_offset in [12, 19, 24]:
                    used_midi_notes.add(midi_note + ghost_offset)
                    
        # handle notes that stopped playing
        for note in list(active_notes.keys()):
            if note not in current_frame_notes:
                start_time, vel = active_notes.pop(note)
                duration = hop_time - start_time
                if duration > 0.12: # debounce filter
                    completed_notes.append((note, start_time, duration, vel))
                    
        # handle notes that started playing
        for note, vel in current_frame_notes.items():
            if note not in active_notes:
                active_notes[note] = (hop_time, vel)
                
    # wrap up trailing notes
    for note, (start_time, vel) in active_notes.items():
        duration = (total_hops * sec_per_hop) - start_time
        if duration > 0.12:
            completed_notes.append((note, start_time, duration, vel))
            
    print(f"Writing {len(completed_notes)} clean, dynamic notes to MIDI file...")
    midi_file = MIDIFile(1)
    midi_file.addTrackName(0, 0, "Ghostwriter Output")
    midi_file.addTempo(0, 0, TARGET_PLAYBACK_BPM)
    
    # scale calculation matching our target tempo track
    # beats = seconds * (BPM / 60)
    time_scaler = TARGET_PLAYBACK_BPM / 60.0
    
    for note, start_time, duration, vel in completed_notes:
        beat_start = start_time * time_scaler
        beat_duration = duration * time_scaler
        midi_file.addNote(0, 0, note, beat_start, beat_duration, vel) 
        
    basename = os.path.splitext(os.path.basename(target_file))[0]
    output_path = os.path.join("..", "data", "midis", "legacy", f"{basename}.mid")
    with open(output_path, "wb") as output_file:
        midi_file.writeFile(output_file)
        
    print(f"Successfully generated clean transcription : {output_path}")
    
if __name__ == "__main__":
    transcribe()