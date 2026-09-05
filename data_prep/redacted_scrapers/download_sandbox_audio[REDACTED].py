import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
AUDIO_RELATIVE_PATH = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"

local_dest = os.path.join("data", "maestro", AUDIO_RELATIVE_PATH)
os.makedirs(os.path.dirname(local_dest), exist_ok=True)

full_url = BASE_URL + AUDIO_RELATIVE_PATH

print(f"Downloading sandbox audio file from Magenta Servers\nFROM : {full_url}\nTO : {local_dest}")

try:
    urllib.request.urlretrieve(full_url, local_dest)
    print("Sandbox audio file downloaded successfully.")
except Exception as e:
    print(f"Download Failed !! Error : {e}")