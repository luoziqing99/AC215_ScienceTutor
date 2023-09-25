import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def download_blob(bucket_name, gcs_folder_path, local_folder_path):
    """Downloads a file to the bucket."""
    print("download")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs()
    print(blobs)

#     for root, dirs, files in os.walk(gcs_folder_path):
#         for file_name in files:
#             local_file_path = os.path.join(root, file_name)
#             blob_destination_path = os.path.join(destination_folder_gcs, os.path.relpath(local_file_path, source_folder_path))
#             blob = bucket.blob(blob_destination_path)
#             blob.upload_from_filename(local_file_path, timeout=600) # 10 minutes timeout
#             print(f'File {local_file_path} uploaded to {blob_destination_path}.')

# def download():
#     print("download")

#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)

#     blobs = bucket.list_blobs(prefix=input_audios + "/")
#     for blob in blobs:
#         print(blob.name)
#         if not blob.name.endswith("/"):
#             blob.download_to_filename(blob.name)

# Usage:
bucket_name = 'ac215-sciencetutor'
gcs_folder_path = 'ScienceQA-LLAVA'
local_folder_path = './ScienceQA-LLAVA'

print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

download_blob(bucket_name, gcs_folder_path, local_folder_path)