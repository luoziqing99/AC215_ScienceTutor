rm -f trainer-yp.tar trainer-yp.tar.gz
tar cvf trainer-yp.tar package
gzip trainer-yp.tar
python3 upload_trainer_to_gcs.py
