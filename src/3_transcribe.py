import os
import glob
import numpy as np
import torch
from midiutil import MIDIFile

from model import SpectrogramAutoencoder

DATA_DIR = os.path.join("..", "data", "wavs")
MODEL_WEIGHTS = os.path.join("..", "data", "saved_models", "autoencoder.pt")
npy_files = glob.glob(os.path.join("..", "data", "matrices", "*.npy"))

SAMPLE_RATE = 22050
HOP_SIZE = 512

def transcribe():
    filtered_files = [f for f in npy_files if "verification" not in f]
    if not filtered_files:
        print(f"ERROR! No data matrices (.npy) found to transcribe.")
        return
    
    target_file = filtered_files[1] # SELECT TARGET FILE (by index)
    print(f"Target Track selected: {os.path.basename(target_file)}")
    matrix = np.load(target_file)
    
    model = SpectrogramAutoencoder(input_dim=84, bottleneck_dim=32)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print("Trained AI weights loaded successfully.")
    else:
        print(f"ERROR ! Weights file : {MODEL_WEIGHTS} not found")
        return
    
    model.eval()
    input_tensor = torch.from_numpy(matrix.T.astype(np.float32))
    with torch.no_grad():
        clean_matrix = model(input_tensor).numpy().T # Shape : (84, total_hops)
        
    sec_per_hop = HOP_SIZE / SAMPLE_RATE 
    total_hops = clean_matrix.shape[1]
    
    active_notes = {} # { midi_note : (start_time, initial_velocity) }
    completed_notes = [] # (midi_note, start_time, duration, velocity)
    
    NOTE_THRESHOLD = 0.45 # base volume threshold
    MIN_DURATION_SEC = 0.15 # ignore anything shorter than 150ms (kills clicks and sharp noises)
    TARGET_PLAYBACK_BPM = 90 # playback beats per minute
    
    for hop_idx in range(total_hops):
        hop_time = hop_idx * sec_per_hop
        column = clean_matrix[:, hop_idx]
        
        # peak finder
        peaks = []
        for i in range(1, len(column) - 1):
            if column[i] > NOTE_THRESHOLD and column[i] > column[i - 1] and column[i] > column[i + 1]: # trigger : > threshold and > neighbors
                peaks.append((i, column[i])) # store (bin_index, intensity)
                
        # sort from loudest to quietest in this exact millisecond
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        current_frame_notes = {}

        # harmonic suppresion to fix ghost notes
        used_midi_notes = set()
        for bin_idx, intensity in peaks:
            midi_note = bin_idx + 24 # row 0 -> C1 (midi 24)
            if midi_note not in used_midi_notes:
                velocity = int(intensity * 127) # map the 0-1 AI intensity directly to standard 0-127 midi volume
                current_frame_notes[midi_note] = max(40, min(velocity, 110))
                used_midi_notes.add(midi_note)
                
                # muffle the immediate ghost harmonics (Octave +12, Perfect Fifth +7 and +19)
                for ghost in [7, 12, 19, 24]:
                    used_midi_notes.add(midi_note + ghost)
                
        # handle notes that stopped playing
        for note in list(active_notes.keys()):
            if note not in current_frame_notes:
                start_time, vel = active_notes.pop(note)
                duration = hop_time - start_time

                # time gate / debounce filter (to remove micro spikes)
                if duration > MIN_DURATION_SEC: 
                    completed_notes.append((note, start_time, duration, vel))
                    
        # handle notes that started playing
        for note, vel in current_frame_notes.items():
            if note not in active_notes:
                active_notes[note] = (hop_time, vel)
                
    # wrap up trailing notes
    for note, (start_time, vel) in active_notes.items():
        duration = (total_hops * sec_per_hop) - start_time
        if duration > MIN_DURATION_SEC:
            completed_notes.append((note, start_time, duration, vel))
            
    print(f"Writing {len(completed_notes)} filtered notes to MIDI file...")
    midi_file = MIDIFile(1)
    midi_file.addTrackName(0, 0, "Ghostwriter CQT Output")
    midi_file.addTempo(0, 0, TARGET_PLAYBACK_BPM)
    # beats = seconds * (BPM / 60)
    time_scaler = TARGET_PLAYBACK_BPM / 60.0 
    for note, start_time, duration, vel in completed_notes:
        beat_start = start_time * time_scaler
        beat_duration = duration * time_scaler
        midi_file.addNote(0, 0, note, beat_start, beat_duration, vel) 
        
    basename = os.path.splitext(os.path.basename(target_file))[0]
    output_path = os.path.join("..", "data", "midis", f"{basename}_cqt_locked.mid")
    with open(output_path, "wb") as output_file:
        midi_file.writeFile(output_file)
        
    print(f"Successfully generated clean transcription : {output_path}")
    
if __name__ == "__main__":
    transcribe()