rm -f trainer2.tar trainer2.tar.gz
tar cvf trainer2.tar package
gzip trainer2.tar
python3 upload_trainer_to_gcs.py
