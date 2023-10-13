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

# Set these parameters accordingly
BUCKET_NAME = 'ac215-sciencetutor-trainer'
SOURCE_BLOB_NAME = 'wandb_api.txt'
DESTINATION_FILE_NAME = './wandb_api.txt'  # adjust the path accordingly

def wandb_apikey() -> str:
    # Download the file
    download_blob(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)
    # Read the downloaded file
    file_content = read_file(DESTINATION_FILE_NAME)
    return file_content