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

**Preprocess container**
- This container reads 100GB of data and resizes the image sizes and stores it back to GCP
- Input to this container is source and destincation GCS location, parameters for resizing, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we do preprocessing on our dataset of 100GB, we reduce the image sizes (a parameter that can be changed later) to 128x128 for faster iteration with our process. Now we have dataset at 10GB and saved on GCS. 

(2) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `special butterfly package` 

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Cross validation, Data Versioning**
- This container reads preprocessed dataset and creates validation split and uses dvc for versioning.
- Input to this container is source GCS location, parameters if any, secrets needed - via docker
- Output is flat file with cross validation splits
  
(1) `src/validation/cv_val.py` - Since our dataset is quite large we decided to stratify based on species and kept 80% for training and 20% for validation. Our metrics will be monitored on this 20% validation set. 

(2) `requirements.txt` - We used following packages to help us with cross validation here - `iterative-stratification` 

(3) `src/validation/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 

----
You may adjust this template as appropriate for your project.
