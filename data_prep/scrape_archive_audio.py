import os
import pandas as pd
import requests

TARGET_DIR = os.path.join("data", "sample_wavs")
os.makedirs(TARGET_DIR, exist_ok=True)

def harvest(query_string, max_files = 10):
    """Queries the public archive.org database for clean solo instrumental tracks, isolates direct target URLs and downloads them natively"""
    
    search_url = "https://archive.org/advancedsearch.php"
    
    params = {
        "q" : f"mediatype:(audio) AND title:({query_string})",
        "fl[]" : "identifier,title",
        "rows" : max_files,
        "page" : 1,
        "output" : "json"
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        docs = response.json().get("response", {}).get("docs", [])
        if not docs:
            print(f"No collection matched the query string : {query_string}")
            return
        
        df_volumes = pd.DataFrame(docs) # loading results into a pandas DataFrame
        print(f"Isolated {len(df_volumes)} volume nodes :--")
        print(df_volumes[['title', 'identifier']].to_string(index=False, max_colwidth=40))
        
        for _, row in df_volumes.iterrows():
            vol_id = row['identifier']
            meta_url = f"https://archive.org/metadata/{vol_id}"
            meta_resp = requests.get(meta_url)
            meta_resp.raise_for_status()
            
            files_list = meta_resp.json().get("files", [])
            if not files_list:
                continue
            df_files = pd.DataFrame(files_list)
            if 'name' not in df_files.columns:
                continue
            df_files['name_lower'] = df_files['name'].str.lower()
            wav_match = df_files[df_files['name_lower'].str.endswith('.wav', na=False)]
            mp3_match = df_files[df_files['name_lower'].str.endswith('.mp3', na=False)]
            target_track = None
            if not wav_match.empty:
                target_track = wav_match.iloc[0]['name']
            elif not mp3_match.empty:
                target_track = mp3_match.iloc[0]['name']
            if not target_track:
                print(f"Skipping Volume [{vol_id}] : Incompatible audio track")
                continue
            
            download_url = f"https://archive.org/download/{vol_id}/{target_track}"
            clean_name = f"ia_{vol_id}_{target_track.replace('/', '_')}"
            local_dest = os.path.join(TARGET_DIR, clean_name)
            print(f"Syncing Asset : {clean_name}")
            
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(local_dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
            print("Synchronized successfully.\n")
            
        print(f"xtraction Finished. Files loaded into {TARGET_DIR}/\n")
        
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__":
    # Query for solo instrument recitals
    harvest(query_string="solo piano", max_files=10)
    # harvest(query_string="acoustic guitar solo", max_files=10)
            
    