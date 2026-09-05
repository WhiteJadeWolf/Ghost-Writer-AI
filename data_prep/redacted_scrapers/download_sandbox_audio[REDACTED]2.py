import os
import opendatasets as od

# Using Kaggle API

DATASET_URL = "https://www.kaggle.com/datasets/alonhaviv/the-maestro-dataset-v3-0-0"
AUDIO_FILE_RELATIVE_PATH = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"

target_url = f"{DATASET_URL}/download/{AUDIO_FILE_RELATIVE_PATH}"

print(f"Downloading sandbox audio file from Kaggle\nTARGET LINK : {target_url}")

try:
    od.download(target_url, data_dir=os.path.join("data", "maestro"))
    print("Sandbox audio file downloaded successfully.")
    expected_path = os.path.join("data", "maestro", "the-maestro-dataset-v3-0-0", "2018")
    if os.path.exists(expected_path):
        print(f"File saved in : {expected_path}")
        print(os.listdir(expected_path))
except Exception as e:
    print(f"Download Failed !! Error : {e}")