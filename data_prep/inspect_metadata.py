import os
import pandas as pd

METADATA_PATH = os.path.join("data", "maestro", "maestro-v3.0.0.json")

def inspect():
    
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR !! Metadata file not found at {METADATA_PATH}")
        return

    print(f"Metadata file found at {METADATA_PATH}.\nLoading Metadata file...")
    
    df = pd.read_json(METADATA_PATH)
        
    print(f"Metadata file loaded successfully.")
        
    total_tracks = len(df)
    print(f"Total performances indexed in the dataset : {total_tracks}")
    
    # Check the distribution of tracks across splits
    print("\nTrack distribution across splits :")
    split_counts = df['split'].value_counts()
    for split_name, count in split_counts.items():
        print(f"    * {split_name.upper()} || TRACKS : {count}")
    
    # Isolating a single sandox track to inspect its metadata
    # Considering the first training track
    train_df = df[df['split'] == 'train']
    sandbox_track = train_df.iloc[0].to_dict()
        
    if sandbox_track:
        print(f"\nSandbox Track Metadata (First Training Track) :")
        print(f"    * PIECE TITLE : {sandbox_track.get('canonical_title')}")
        print(f"    * COMPOSER : {sandbox_track.get('canonical_composer')}")
        print(f"    * YEAR : {sandbox_track.get('year')}")
        print(f"    * AUDIO FILENAME : {sandbox_track.get('audio_filename')}")
        print(f"    * MIDI FILENAME : {sandbox_track.get('midi_filename')}")
        print(f"    * DURATION : {sandbox_track.get('duration')}")
    else:
        print("ERROR !! Could not isolate a track sample.")
        
if __name__ == "__main__":
    inspect()