import os
from huggingface_hub import hf_hub_download

REPO = "projectlosangeles/maestro-v3.0.0"
AUDIO_RELATIVE_PATH = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"
MIDI_RELATIVE_PATH = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"

print("Connecting to Hugging Face API via Hub client...")

midi_local_path = hf_hub_download(
    repo_id=REPO,
    repo_type="dataset",
    filename=MIDI_RELATIVE_PATH,
    local_dir=os.path.join("data", "maestro"),
    local_dir_use_symlinks=False
    )
print(f"MIDI file downloaded successfully and saved at : {midi_local_path}\n")

audio_local_path = hf_hub_download(
    repo_id=REPO,
    repo_type="dataset",
    filename=AUDIO_RELATIVE_PATH,
    local_dir=os.path.join("data", "maestro"),
    local_dir_use_symlinks=False
    )
print(f"Audio file downloaded successfully and saved at : {audio_local_path}\n")