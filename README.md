# AC215 - ScienceTutor

## Application Pipeline Flow

<img width="1362" alt="image" src="pictures/science_tutor_app_pipeline.png">

## Project Organization
      .
      ├── LICENSE
      ├── README.md
      ├── notebooks
      │   └── AC215_milestone3_model_training.ipynb
      ├── pictures
      │   ├── science_tutor_app_pipeline.png
      │   ├── gcs_model_bucket.png
      │   ├── vertex_ai_model_training.png
      │   ├── wandb_system.png
      │   └── wandb_train.png
      ├── references
      ├── reports
      └── src
            ├── data_processing
            │   ├── Dockerfile
            │   ├── convert_scienceqa_to_llava.py
            │   ├── ScienceQA-LLAVA.dvc
            │   ├── upload_to_gcs.py
            │   ├── upload_to_hf.py
            │   └── requirements.txt
            ├── model_training
            │   ├── package
            │   │   ├── trainer
            │   │   │   ├── __init__.py
            │   │   │   ├── task.py
            │   │   │   └── wandb_api.py
            │   │   ├── PKG-INFO
            │   │   ├── setup.cfg
            │   │   └── setup.py
            │   ├── cli.py
            │   ├── cli.sh
            │   ├── docker-entrypoint.sh
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── download_from_gcs.py
            │   ├── download_from_hf.py
            │   ├── package-trainer.sh
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   └──upload_trainer_to_gcs.py
            ├── chatbot_logic
            │   ├── Dockerfile
            │   ├── docker-shell.sh
            │   ├── Pipfile
            │   └── Pipfile.lock
            └── web_server
                ├── Dockerfile
                ├── docker-shell.sh
                ├── Pipfile
                └── Pipfile.lock

## AC215 - Milestone3 - ScienceTutor

**Team Members** Sijia (Nancy) Li, Ziqing Luo, Yuqing Pan, Jiashu Xu, Xiaohan Zhao

**Group Name** Science Tutor

**Project** In this project we aim to develop an educational application that provides instant and expert answers to science questions that children have in different domains such as natural, social and language science.

### Milestone3

We further refined our data pipeline process for milestone 3 by forking the LLaVA repository and [updating the code for passing in the ScienceQA that we preprocessed](https://github.com/cnut1648/LLaVA). By doing this, we customized the model to take into our own preprocessed ScienceQA dataset. 

Regarding the modeling process, we tried several optimization techniques to reduce memory usage: 
- bf16
- deepspeed ZERO-2 for multi-GPU
- gradient checkpointing
- gradient accumulation
- tf32

In our colab version, we use all those optimization techniques with A100 GPU except deepspeed as we can only access 1 GPU. For Vertex AI, Google approved our request for 4 NVIDIA_TESLA_V100 GPU but we do not have NVIDIA_TESLA_A100 GPU:
V100 unfortunately does not support bf16. We tried fp16 but due to Huggingface implementation of LLaMA (model that LLaVA is based on), there is a data type conversion error in attention computation with fp16. Moreover, tf32 is also not supported. We found that we cannot load the model into the memory using 4 V100 on Vertex AI, let alone training it.

## Experiment Tracking

The images below show the training output from our Weights & Biases Page. The Weights & Biases Page tracks our model training process. This is done by using the `wandb` library that we included in our `task.py` Python script.

Train Tracking:
<img width="1362" alt="image" src="pictures/wandb_train.png">

System Tracking:
<img width="1362" alt="image" src="pictures/wandb_system.png">

## Serverless Training

There are three main steps to launch training instances in the cloud: (1) run and build model training container, (2) package model training code and upload into a bucket in Google Cloud Storage, and (3) run the model on Vertex AI. 

To create a new serverless job we did the following commands:
```shell
cd src/model_training
sh docker-shell.sh
sh package-trainer.sh
sh cli.sh
```

Google Cloud Storage Bucket with our training code stored in `trainer.tar.gz`:
<img width="1362" alt="image" src="pictures/gcs_model_bucket.png">

Vertex AI showing our attempts for model training (currently we are still restricted by Vertex AI's GPU quota and cannot load our model into memory):
<img width="1362" alt="image" src="pictures/vertex_ai_model_training.png">

## Dataset Evaluation

We have trained the model on ScienceQA, and to evaluate our model performance on science domain, we provice code to evalaute on the testset of ScienceQA, which contains 4241 instances.

```shell
cd src/model_inference;
git clone https://github.com/cnut1648/LLaVA.git # our forked repo

# this will cache inference results in `src/model_inference/scienceqa-eval.jsonl`
PYTHONPATH=LLaVA python -m model_vqa_science \
    --model-path <your llava model> \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python compute_metric.py
```
Our code supports multi-GPU inference, simply set your `CUDA_VISIBLE_DEVICES` environment variable to the GPUs you want to use. For example, to use GPU 0 and 1, run `CUDA_VISIBLE_DEVICES=0,1 python -m model_vqa_science ...`.

For our 7B model, the performance is
```
Total: 4241, Correct: 2779, Accuracy: 65.53%, IMG-Accuracy: 63.86%
```
which is pretty close to the performance reported by the LLaVA 13B (~70%). Note that that is a larger model with a possibly more careful hyperparameter tuning while we only train for one epoch with a default hyperparameter.

## Code Structure

### notebooks
This folder contains code that is not part of container, for example, model training testing code for debugging purposes.

[`notebooks/AC215_milestone3_model_training.ipynb`](notebooks/AC215_milestone3_model_training.ipynb): This notebook includes the code we used for training our model on Colab.

### src
This folder contains the development code for the ScienceTutor application.

#### (1) Data Processing Container
- This container loads the dataset from huggingface, and convert each data instance into LLaVA format to enforce format consistency as LLaVA training format.
- This container will store the reformatted dataset, so that user can retrieve the dataset to (1) use for training (2) upload to GCP, huggingface etc, your choice.

(1) [`src/data_processing/convert_scienceqa_to_llava.py`](src/data_processing/convert_scienceqa_to_llava.py): This script converts the ScienceQA dataset downloaded from Huggingface into the data format that can be passed into the LLaVA model.

(2) [`src/data_processing/requirements.txt`](src/data_processing/requirements.txt): This file specifies the packages required to be installed.

(3) [`src/data_processing/Dockerfile`](src/data_processing/Dockerfile): This is the Dockerfile to build the container.

(4) [`src/data_processing/upload_to_hf.py`](src/data_processing/upload_to_hf.py): This script uploads the data to Huggingface as a private dataset.

(5) [`src/data_processing/upload_to_gcs.py`](src/data_processing/upload_to_gcs.py): This script uploads the data to Google Cloud Storage.

However, as mentioned in [Data Versioning](#data-versioning), we use `dvc` to version control the dataset. You can simply `dvc pull` to obtain the processed dataset, and can safely skip the rest of this section.

To run Dockerfile:
```shell
cd src/data_processing;
# build docker
docker build .
# ...
# Successfully built b0d701fb573e

# run container from image
docker run -it -d b0d701fb573e
# get container id
docker ps
# to explore dataset and use dataset
docker exec -it <container_id> bash
# OR
# to copy to host
# reformatted dataset
docker cp <container_id>:/usr/src/app/ScienceQA-LLAVA ./ScienceQA-LLAVA
# original dataset
docker cp <container_id>:/usr/src/app/ScienceQA ./ScienceQA
```

To upload to huggingface/GCS, first create a `.env` as follows:
```
HF_TOKEN=<YOUR HUGGINGFACE TOKEN>
GOOGLE_APPLICATION_CREDENTIALS=<PATH TO SERVICE ACCOUNT CREDENTIALS>
```
Then `python upload_to_hf.py` to upload to huggingface as a private dataset; or 
`python upload_to_gcs.py` to upload to GCS.

To ease development, we have uploaded the reformatted dataset to

- Huggingface: [`cnut1648/ScienceQA-LLAVA`](https://huggingface.co/datasets/cnut1648/ScienceQA-LLAVA/).
- GCS: [`gs://ac215-sciencetutor/ScienceQA-LLAVA`](https://console.cloud.google.com/storage/browser/ac215-sciencetutor/ScienceQA-LLAVA). For TA, please contact us for access.

##### Data Versioning
We additionally use `dvc` to version control the dataset. Specifically, `src/data_processing/ScienceQA-LLAVA.dvc` is the dvc file that tracks the reformatted dataset. The data is remotely tracked in GCS. To download the dataset, run `dvc pull` after cloning the repo.

#### (2) Model Training Container
This container will download the processed dataset and train the LLaVA model. The trained LLaVA model will be used in the chatbot logic component to perform the visual question answering (VQA) task. 

To build and run the container, package the model training code, and send job to Vertex AI:
```shell
cd src/model_training
sh docker-shell.sh
sh package-trainer.sh
sh cli.sh
```

Files for downloading the datasets:

(1) [`src/model_training/download_from_hf.py`](src/model_training/download_from_hf.py): This script downloads the dataset from Huggingface.

(2) [`src/model_training/download_from_gcs.py`](src/model_training/download_from_gcs.py): This script downloads the dataset from Google Cloud Storage.

(3) [`src/model_training/Dockerfile`](src/model_training/Dockerfile), [`src/model_training/Pipfile`](src/model_training/Pipfile), [`src/model_training/Pipfile.lock`](src/model_training/Pipfile.lock), [`src/model_training/docker-entrypoint.sh`](src/model_training/docker-entrypoint.sh), [`src/model_training/docker-shell.sh`](src/model_training/docker-shell.sh): These are the files to build the container.

(4) [`src/model_training/package/`](src/model_training/package/): This is the folder that contains the model training code and wandb_api_key upload code.

(5) [`src/model_training/package-trainer.sh`](src/model_training/package-trainer.sh): This is the script for packaging the model training code into `trainer.tar.gz`.

(6) [`src/model_training/upload_trainer_to_gcs.py`](src/model_training/upload_trainer_to_gcs.py): This script uploads the `trainer.tar.gz` containing the model training code to Google Cloud Storage Bucket.

(7) [`src/model_training/upload_model_to_gcs.py`](src/model_training/upload_model_to_gcs.py): This script uploads the `checkpoints` folder containing the trained model checkpoints to Google Cloud Storage Bucket. 

(8) [`src/model_training/cli.py`](src/model_training/cli.py), [`src/model_training/cli.sh`](src/model_training/cli.sh): These are the scripts for command-line interface (CLI) to create custom model training jobs on Vertex AI. 


#### (3) Web Server Container
This container serves as the frontend of our Science Tutor chatbot application. It handles HTTP requests, provides a user interface, and communicates with the chatbot logic component.

To build and run the container:
```shell
cd src/web_server
sh docker-shell.sh
```

In this milestone, it is a placeholder for future implementation.

#### (4) Chatbot Logic Container
This container contains the core chatbot logic. It processes user messages received from the web server container, conducts inference with the model API and generates responses.

To build and run the container:
```shell
cd src/chatbot_logic
sh docker-shell.sh
```

In this milestone, it is a placeholder for future implementation.

#### (5) Other Containers
In addition to the existing containers, we may consider incorporating additional containers as the need arises. 
This may include a database container for the storage of user message data, and a recommendation engine container housing the logic for recommending posts or videos based on the questions user asked.

