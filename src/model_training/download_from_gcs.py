import os
from google.cloud import storage
from dotenv import load_dotenv
import shutil

load_dotenv()

def download_blob(bucket_name, folder_path):
    """Downloads a file to the bucket."""
    print("download")

    # Clear
    shutil.rmtree(folder_path, ignore_errors=True, onerror=None)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f"{folder_path}/test", exist_ok=True)
    os.makedirs(f"{folder_path}/train", exist_ok=True)
    os.makedirs(f"{folder_path}/validation", exist_ok=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=f"{folder_path}/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)

# Usage:
bucket_name = 'ac215-sciencetutor'
folder_path = 'ScienceQA-LLAVA'

download_blob(bucket_name, folder_path)