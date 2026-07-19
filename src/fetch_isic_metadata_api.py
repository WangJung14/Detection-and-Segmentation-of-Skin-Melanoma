import pandas as pd
import requests
import concurrent.futures
import time
import argparse

def fetch_metadata(image_id):
    url = f"https://api.isic-archive.com/api/v2/images/search?query=isic_id:{image_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['count'] > 0:
                meta = data['results'][0].get('metadata', {}).get('clinical', {})
                # Try to get the diagnosis. Sometimes it's in diagnosis, diagnosis_1, diagnosis_3, etc.
                diag = meta.get('diagnosis_1') or meta.get('diagnosis') or meta.get('benign_malignant') or meta.get('diagnosis_3') or 'unknown'
                return {'image_id': image_id, 'diagnosis': diag}
    except Exception as e:
        print(f"Error fetching {image_id}: {e}")
    return {'image_id': image_id, 'diagnosis': 'unknown'}

def fetch_all(features_csv, output_csv):
    df_features = pd.read_csv(features_csv)
    image_ids = df_features['image_id'].tolist()
    
    print(f"Fetching metadata for {len(image_ids)} images via ISIC API...")
    results = []
    
    start_time = time.time()
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {executor.submit(fetch_metadata, img_id): img_id for img_id in image_ids}
        for count, future in enumerate(concurrent.futures.as_completed(future_to_id), 1):
            results.append(future.result())
            if count % 100 == 0:
                print(f"Fetched {count}/{len(image_ids)}...")
                
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")
    
    df_meta = pd.DataFrame(results)
    df_meta.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()
    
    fetch_all(args.features_csv, args.output_csv)
