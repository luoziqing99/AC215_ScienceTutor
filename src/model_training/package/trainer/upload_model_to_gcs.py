import os
from google.cloud import storage

def upload_blob(bucket_name, source_folder_path, destination_folder_gcs):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_destination_path = os.path.join(destination_folder_gcs, os.path.relpath(local_file_path, source_folder_path))
            blob = bucket.blob(blob_destination_path)
            blob.upload_from_filename(local_file_path, timeout=600) # 10 minutes timeout
            print(f'File {local_file_path} uploaded to {blob_destination_path}.')


BUCKET_NAME = "ac215-sciencetutor-trainer"
def upload_model_checkpoint(checkpoint_name):
    source_folder_path = f'./checkpoints/{checkpoint_name}'
    destination_folder_gcs = f'checkpoints/{checkpoint_name}'
    upload_blob(BUCKET_NAME, source_folder_path, destination_folder_gcs)
    print(f"Uploaded model checkpoint {checkpoint_name} to {BUCKET_NAME}")