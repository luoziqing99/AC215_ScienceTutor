AC215-Template (Milestone2)
==============================

AC215 - Milestone2

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      └── src
            ├── chatbot_logic
            │   ├── Dockerfile
            │   ├── chatbot_logic.py
            │   └── requirements.txt
            ├── data_processing
            │   ├── Dockerfile
            │   ├── data_processing.py
            │   └── requirements.txt
            ├── model_training
            │   ├── Dockerfile
            │   ├── model_training.py
            │   └── requirements.txt
            └── web_server
                  ├── Dockerfile
                  ├── web_server.py
                  └── requirements.txt


--------
# AC215 - Milestone2 - ScienceTutor

**Team Members**
Sijia (Nancy) Li, Ziqing Luo, Yuqing Pan, Jiashu Xu, Xiaohan Zhao

**Group Name**
Science Tutor

**Project**
In this project we aim to develop an educational application that provides instant and expert answers to science questions that children have in different domains such as natural, social and language science.

### Milestone2 ###

We will use [ScienceQA](https://scienceqa.github.io/#dataset), which is a public dataset that consists of ~21k multimodal multiple choice questions covering a diverse set of science topics. The dataset is available at [Hugging Face](https://huggingface.co/datasets/derek-thomas/ScienceQA).

#### Preprocess container
- The container load dataset from huggingface, and convert each data instance into LLaVA format to enforce format consistency as LLaVA training format.
- The container will store the reformatted dataset, so that user can retrieve the dataset to (1) use for training (2) upload to GCP, huggingface etc, your choice.

(1) [`src/data_processing/convert_scienceqa_to_llava.py`](src/data_processing/convert_scienceqa_to_llava.py): conversion code

(2) [`src/data_processing/requirements.txt`](src/data_processing/requirements.txt): required packages

(3) [`src/data_processing/Dockerfile`](src/data_processing/Dockerfile): Dockerfile to build the container

(4) [`src/data_processing/upload_to_hf.py`](src/data_processing/upload_to_hf.py): upload to huggingface as private dataset

(5) [`src/data_processing/upload_to_gcs.py`](src/data_processing/upload_to_gcs.py): upload to Google Cloud Storage.

However, as mentioned in [Data Versioning](#data-versioning), we use `dvc` to version control the dataset.
You can simply `dvc pull` to obtain the processed dataset, and can safely skip the rest of this section.

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

#### Data Versioning

We additionally use `dvc` to version control the dataset.
Specifically, `src/data_processing/ScienceQA-LLAVA.dvc` is the dvc file that tracks the reformatted dataset. 
The data is remotely tracked in GCS.
To download the dataset, run `dvc pull` after cloning the repo.