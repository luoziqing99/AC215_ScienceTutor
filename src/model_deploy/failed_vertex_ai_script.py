APP_NAME = "ST"
CUSTOM_PREDICTOR_IMAGE_URI="gcr.io/ac215project-398401/sciencetutor"
from google.cloud import aiplatform
VERSION = 2
model_display_name = f"{APP_NAME}-v{VERSION}"
model_description = "PyTorch based text classifier with custom container"
# health_route = "/ping"
predict_route = f"/api/v1/stream"
serving_container_ports = [5005, ] #7860, 5000]
# List all models you have
existing_models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
if len(existing_models) > 0:
    assert len(existing_models) == 1
    model = existing_models[0]
else:
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        description=model_description,
        serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
        serving_container_predict_route=predict_route,
        #serving_container_health_route=health_route,
        serving_container_ports=serving_container_ports,
    )
    model.wait()
print(model.display_name)
print(model.resource_name)

existing_endpoints = aiplatform.Endpoint.list(filter=f'display_name="{APP_NAME}-endpoint"')
if len(existing_endpoints) > 0:
    assert len(existing_endpoints) == 1
    endpoint = existing_endpoints[0]
else:
    endpoint_display_name = f"{APP_NAME}-endpoint"
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

traffic_percentage = 100
deployed_model_display_name = model_display_name
min_replica_count = 1
max_replica_count = 3
sync = True
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=deployed_model_display_name,
    machine_type="n1-highmem-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
    traffic_percentage=traffic_percentage,
    sync=sync,
)
model.wait()
print(model.display_name)