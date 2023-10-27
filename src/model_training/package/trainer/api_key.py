from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

BUCKET_NAME = 'ac215-sciencetutor-trainer2'
def get_apikey(type) -> str:
    # Download the file
    src_name = f'{type}_api.txt'
    dest_name = f'./{type}_api.txt'
    download_blob(BUCKET_NAME, src_name, dest_name)
    # Read the downloaded file
    file_content = read_file(dest_name).strip()
    return file_content