import os
import random
import string
import google.cloud.aiplatform as aip

# GCP_PROJECT = os.environ["GCP_PROJECT"]
# GCP_REGION = os.environ["GCP_REGION"]
# GCS_BUCKET_URI = os.environ["GCS_BUCKET_URI"]
GCP_PROJECT = "ac215project-398401"
GCP_REGION = "us-central1"
GCS_BUCKET_URI = "gs://ac215-sciencetutor-trainer"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


# Initialize Vertex AI SDK for Python
aip.init(
    # project=GCP_PROJECT,
         location=GCP_REGION, staging_bucket=GCS_BUCKET_URI)

job_id = generate_uuid()
DISPLAY_NAME = "lsj-" + job_id

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"

job = aip.CustomPythonPackageTrainingJob(
    display_name=DISPLAY_NAME,
    python_package_gcs_uri=f"{GCS_BUCKET_URI}/trainer.tar.gz",
    python_module_name="trainer.llava.train.train_mem",
    container_uri=TRAIN_IMAGE,
    project=GCP_PROJECT,
)

CMDARGS = ["--epochs=1"]
MODEL_DIR = GCS_BUCKET_URI
TRAIN_COMPUTE = "a2-highgpu-1g"
TRAIN_GPU = "NVIDIA_TESLA_A100"
TRAIN_NGPU = 1

# Run the training job on Vertex AI
# sync=True, # If you want to wait for the job to finish
job.run(
    model_display_name=None,
    args=CMDARGS,
    replica_count=1,
    machine_type=TRAIN_COMPUTE,
    accelerator_type=TRAIN_GPU,
    accelerator_count=TRAIN_NGPU,
    base_output_dir=MODEL_DIR,
    # sync=True,
)

print("done")