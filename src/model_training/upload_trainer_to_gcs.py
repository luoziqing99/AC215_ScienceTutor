from google.cloud import storage
storage_client = storage.Client()

bucket_name = "ac215-sciencetutor-trainer2"
filepath = "trainer-yp.tar.gz"
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(filepath)
blob.upload_from_filename(filepath, timeout=300)
