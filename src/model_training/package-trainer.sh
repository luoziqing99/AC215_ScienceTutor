rm -f trainer.tar trainer.tar.gz
tar cvf trainer.tar package
gzip trainer.tar
python3 upload_trainer_to_gcs.py
