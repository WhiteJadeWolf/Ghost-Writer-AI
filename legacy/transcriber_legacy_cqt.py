import os
import glob
import numpy as np
import torch
from midiutil import MIDIFile
from legacy_model import SpectrogramAutoencoder

MODEL_WEIGHTS = os.path.join("..", "data", "saved_models", "legacy", "autoencoder_legacy.pt")
npy_files = glob.glob(os.path.join("..", "data", "matrices", "legacy", "*.npy"))

# Audio constraints matching our new CQT setup
SAMPLE_RATE = 22050
HOP_SIZE = 512

def transcribe():
    filtered_files = [f for f in npy_files if "verification" not in f]
    if not filtered_files:
        print(f"ERROR! No data matrices found.")
        return
    
    target_file = filtered_files[1]
    print(f"Target Track selected: {os.path.basename(target_file)}")
    matrix = np.load(target_file)
    
    # Initialize model with 84 dimensions
    model = SpectrogramAutoencoder(input_dim=84, bottleneck_dim=32)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print("Trained AI weights loaded.")
    else:
        return
    
    model.eval()
    input_tensor = torch.from_numpy(matrix.T.astype(np.float32))
    with torch.no_grad():
        clean_matrix = model(input_tensor).numpy().T # Shape: (84, total_hops)
        
    sec_per_hop = HOP_SIZE / SAMPLE_RATE 
    total_hops = clean_matrix.shape[1]
    
    active_notes = {}    
    completed_notes = [] 
    
    NOTE_THRESHOLD = 0.50 # High confidence filter
    TARGET_PLAYBACK_BPM = 90
    
    for hop_idx in range(total_hops):
        hop_time = hop_idx * sec_per_hop
        column = clean_matrix[:, hop_idx]
        
        # Simple threshold tracking across the 84 absolute piano keys
        current_frame_notes = {}
        for bin_idx in range(len(column)):
            if column[bin_idx] > NOTE_THRESHOLD:
                # Row 0 maps directly to C1 (MIDI Note 24)
                midi_note = bin_idx + 24 
                velocity = int(column[bin_idx] * 127)
                current_frame_notes[midi_note] = max(50, min(velocity, 110))
                
        # Handle notes that stopped playing
        for note in list(active_notes.keys()):
            if note not in current_frame_notes:
                start_time, vel = active_notes.pop(note)
                duration = hop_time - start_time
                if duration > 0.10: 
                    completed_notes.append((note, start_time, duration, vel))
                    
        # Handle notes that started playing
        for note, vel in current_frame_notes.items():
            if note not in active_notes:
                active_notes[note] = (hop_time, vel)
                
    # Wrap up trailing notes
    for note, (start_time, vel) in active_notes.items():
        duration = (total_hops * sec_per_hop) - start_time
        if duration > 0.10:
            completed_notes.append((note, start_time, duration, vel))
            
    print(f"Writing {len(completed_notes)} mathematically locked notes to MIDI file...")
    midi_file = MIDIFile(1)
    midi_file.addTrackName(0, 0, "Ghostwriter Legacy CQT Output")
    midi_file.addTempo(0, 0, TARGET_PLAYBACK_BPM)
    
    time_scaler = TARGET_PLAYBACK_BPM / 60.0
    for note, start_time, duration, vel in completed_notes:
        beat_start = start_time * time_scaler
        beat_duration = duration * time_scaler
        midi_file.addNote(0, 0, note, beat_start, beat_duration, vel) 
        
    basename = os.path.splitext(os.path.basename(target_file))[0]
    output_path = os.path.join("..", "data", "midis", "legacy", f"{basename}_cqt_locked.mid")
    with open(output_path, "wb") as output_file:
        midi_file.writeFile(output_file)
        
    print(f"Success! Generated perfectly tracking transcription: {output_path}")
    
if __name__ == "__main__":
    transcribe()