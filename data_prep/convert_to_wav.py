import os
import glob
import pandas as pd
import soundfile as sf
import librosa

TARGET_DIR = os.path.join("data", "sample_wavs")
SAMPLE_RATE = 44100

def convert_to_wav():
    mp3_pattern = os.path.join(TARGET_DIR, "*.mp3")
    mp3_files = glob.glob(mp3_pattern)
    
    if not mp3_files:
        print(f"No MP3 files discovered in target directory : {TARGET_DIR}")
        return
    
    df = pd.DataFrame([{"filename": os.path.basename(p), "path": p} for p in mp3_files])
    print(f"Found {len(df)} targets for conversion...\n")
    
    success = 0
    for idx, row in df.iterrows():
        mp3_path = row['path']
        basename = os.path.splitext(row['filename'])[0]
        wav_path = os.path.join(TARGET_DIR, f"{basename}.wav")
        
        try:
            audio_signal, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
            sf.write(wav_path, audio_signal, SAMPLE_RATE, subtype='PCM_16')
            print(f"[{idx+1}/{len(df)}] Soundfile Converted : {basename}.wav")
            success += 1
            # os.remove(mp3_path)
            
        except Exception as e:
            print(f"Extraction Error on {row['filename']} : {e}")
            
    print(f"\nSuccessfully processed {success}/{len(df)} files via Soundfile.")

if __name__ == '__main__':
    convert_to_wav()