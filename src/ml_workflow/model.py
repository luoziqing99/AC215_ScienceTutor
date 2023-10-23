from kfp import dsl


# Define a Container Component
@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_training(
    project: str = "",
    location: str = "",
    staging_bucket: str = "",
    bucket_name: str = "",
    epochs: int = 15,
    batch_size: int = 16,
    model_name: str = "llava-sqa",
    train_base: bool = False,
):
    print("Model Training Job")

    import google.cloud.aiplatform as aip

    # Initialize Vertex AI SDK for Python
    aip.init(project=project, location=location, staging_bucket=staging_bucket)

    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
    python_package_gcs_uri = f"{staging_bucket}/trainer.tar.gz"

    job = aip.CustomPythonPackageTrainingJob(
        display_name="sciencetutor-app-training",
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name="trainer.task",
        container_uri=container_uri,
        project=project,
    )

    CMDARGS = [
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--model_name={model_name}",
        f"--bucket_name={bucket_name}",
    ]
    if train_base:
        CMDARGS.append("--train_base")

    MODEL_DIR = staging_bucket
    TRAIN_COMPUTE = "n1-standard-4"
    TRAIN_GPU = "NVIDIA_TESLA_V100"
    TRAIN_NGPU = 4

    print(python_package_gcs_uri)

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
        sync=True,
    )


# # Define a Container Component
@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_deploy(
    bucket_name: str = "",
):
    print("Model Training Job")

    import google.cloud.aiplatform as aip

    # List of prebuilt containers for prediction
    # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    serving_container_image_uri = ("13052423200/scienceqa_llava")
    serving_container_ports = (7860, 5000, 5005)

    display_name = "ScienceTutor App Model"
    # ARTIFACT_URI = f"gs://{bucket_name}/model"

    # Upload and Deploy model to Vertex AI
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
    deployed_model = aip.Model.upload(
        display_name=display_name,
        # artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_ports=serving_container_ports,
    )

    DEPLOY_COMPUTE = "n1-standard-16"
    DEPLOY_GPU = "NVIDIA_TESLA_V100"
    DEPLOY_NGPU = 1

    print("deployed_model:", deployed_model)
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
    endpoint = deployed_model.deploy(
        deployed_model_display_name=display_name,
        traffic_split={"0": 100},
        machine_type=DEPLOY_COMPUTE,
        accelerator_count=DEPLOY_NGPU,
        accelerator_type=DEPLOY_GPU,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )
    print("endpoint:", endpoint)
