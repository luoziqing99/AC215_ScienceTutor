# AC215 - ScienceTutor: An Educational Application For Children

### Presentation Video
* https://drive.google.com/file/d/1y41Zua5000fInmUBlQKhANr39Wn5bQJf/view?usp=sharing

### Blog Post Link
* https://medium.com/@lsjnancy/94a5af6b1b74

------------

## Project Organization
```
.
├── .github
│   └── workflows
|       └── ci-cd.yml
├── LICENSE
├── README.md
├── notebooks
│   └── AC215_milestone3_model_training.ipynb
├── pictures
│   ├── apidoc.png
│   ├── chatbot-v2.png
│   ├── chatbot.png
│   ├── compute_engine.png
│   ├── gcs_model_bucket.png
│   ├── k8s-v2.png
│   ├── k8s.png
│   ├── ml_workflow.png
│   ├── ml_workflow_pipeline_run.png
│   ├── postman.png
│   ├── science_tutor_app_pipeline.png
│   ├── science_tutor_app_pipeline2.png
│   ├── solution_architecture.png
│   ├── technical_architecture.png
│   ├── vertex_ai_model_training.png
│   ├── wandb_system.png
│   ├── wandb_train.png
│   └── web_server_demo.png
├── presentations
│   ├── AC215-final-presentation.mp4
│   ├── AC215-final-presentation.pdf
│   ├── AC215-midterm-demo.mp4
│   └── AC215-midterm.pdf
├── references
├── reports
│   ├── milestone2.md
│   ├── milestone3.md
│   ├── milestone4.md
│   └── milestone5.md
└── src
    ├── api-service                 <-- Code for app backend APIs
    │   ├── Dockerfile
    │   ├── api
    │   │   └── model_backend.py
    │   ├── docker-shell.sh
    │   └── requirements.txt
    ├── app_deploy                  <-- Code for app deployment to GCP
    │   ├── Dockerfile
    │   ├── deploy-create-instance.yml
    │   ├── deploy-docker-images.yml
    │   ├── deploy-k8s-cluster.yml
    │   ├── deploy-provision-instance.yml
    │   ├── deploy-setup-containers.yml
    │   ├── deploy-setup-webserver.yml
    │   ├── update-k8s-cluster.yml
    │   ├── deploy-app-init.sh
    │   ├── deploy-app.sh
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── update-deploy-app.sh
    │   ├── inventory.yml
    │   ├── inventory-prod.yml
    │   └── nginx-conf
    │       └── nginx
    │           └── nginx.conf
    ├── data_processing             <-- Code for data processing
    │   ├── Dockerfile
    │   ├── ScienceQA-LLAVA.dvc
    │   ├── convert_scienceqa_to_llava.py
    │   ├── docker-shell.sh
    │   ├── requirements.txt
    │   ├── upload_to_gcs.py
    │   ├── upload_to_hf.py
    │   └── utils.py
    ├── frontend                    <-- Code for app frontend
    │   ├── Dockerfile
    │   ├── Dockerfile.dev
    │   ├── docker-shell.sh
    │   ├── index.html
    │   ├── node_modules
    │   ├── package-lock.json
    │   ├── package.json
    │   ├── public
    │   │   ├── send.png
    │   │   ├── student.png
    │   │   ├── teacher.png
    │   ├── src
    │   │   ├── App.css
    │   │   ├── App.jsx
    │   │   ├── index.css
    │   │   └── main.jsx
    │   └── vite.config.js
    ├── ml_workflow                 <-- Scripts for automating data processing and modeling
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cli.py
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── model.py
    │   ├── model_training.yaml
    │   └── pipeline.yaml
    ├── model_deploy                <-- Model deployment
    │   ├── Dockerfile
    │   ├── api_example
    │   │   ├── req.json
    │   │   └── websocket_streaming.py
    │   ├── docker-shell.sh
    │   └── failed_vertex_ai_script.py
    ├── model_inference             <-- Model inference
    │   ├── compute_metric.py
    │   └── model_vqa_science.py
    └── model_training              <-- Model training
        ├── Dockerfile
        ├── Pipfile
        ├── Pipfile.lock
        ├── cli.sh
        ├── docker-entrypoint.sh
        ├── docker-shell.sh
        ├── download_from_gcs.py
        ├── download_from_hf.py
        ├── package
        │   ├── PKG-INFO
        │   ├── setup.cfg
        │   ├── setup.py
        │   └── trainer
        │       ├── __init__.py
        │       ├── task.py
        │       └── wandb_api.py
        ├── package-trainer.sh
        ├── trainer-yp.tar.gz
        ├── upload_model_to_gcs.py
        └── upload_trainer_to_gcs.py
```
------------

## AC215 - Final Project

**Team Members** Sijia (Nancy) Li, Ziqing Luo, Yuqing Pan, Jiashu Xu, Xiaohan Zhao

**Group Name** Science Tutor

**Project - Problem Definition** In this project we aim to develop an educational application that provides instant and expert answers to science questions that children have in different domains such as natural, social and language science.

### Data Description
We will use [ScienceQA](https://scienceqa.github.io/#dataset), which is a public dataset that consists of ~21k multimodal multiple choice questions covering a diverse set of science topics. The dataset is available at [Hugging Face](https://huggingface.co/datasets/derek-thomas/ScienceQA). 


After completions of building a robust ML Pipeline in our previous milestone we have built a backend api service using Flask and a frontend web app using React. 
This will be our user-facing application that ties together the various components built in previous milestones.

### Proposed Solution
Before we start implementing the app we built a detailed design document outlining the application’s architecture. 
We built a Solution Architecture abd Technical Architecture to ensure all our components work together.

### Solution Architecture
<img width="1362" alt="image" src="../pictures/solution_architecture.png">

### Technical Architecture
<img width="1362" alt="image" src="../pictures/technical_architecture.png">

### Backend API
We built backend api service using Flask to expose model functionality to the frontend. 

We provide a `/chat` endpoint with `POST` method. You can check `/apidocs` for Swagger UI API docs.
<img width="1362" src="../pictures/apidoc.png">

We also used postman to test the API.
<img width="1362" src="../pictures/postman.png">

### Frontend
A user friendly React app was built to interact with the Science Tutor chatbot in the web browser using the Llava-7b model finetuned on ScienceQA. Using the app, a user can type a question and upload an image, and then send the messages to the chatbot. The app will send the text and image (if an image is uploaded) to the backend api to get the model's output on what the answer will be to the given question (and image). Once the app gets the response from the backend api, the app will then reply to the user in the chat. 

Here is a screenshot of our app:
<img width="1362" alt="image" src="../pictures/chatbot.png">


### Deployment
We used Ansible and Kubernetes to create, provision, and deploy our frontend and backend to GCP in an automated fashion.

We successfully created the Kubernetes cluster for our app in GCP:
<img width="1362" alt="image" src="../pictures/k8s.png">


## Code Structure

The following are the folders from the previous milestones:
```
- data_processing
- model_training
- model_inference
- model_deploy
- ml_workflow
```

### API Service Container

This container has the python file `api/model_backend.py` to run and expose the backend apis.

To run the container locally:
* Open a terminal and go to the location where `src/api-service`
* Run `sh docker-shell.sh`
* The backend server is launched at `http://localhost:5000/` and `http://127.0.0.1:5000`
* Go to `http://127.0.0.1:5000/chat` to interact with the endpoint
* Go to `http://127.0.0.1:5000/apidocs` to view the APIs

### Frontend Container
This container contains all the files to develop and build a react app. There are dockerfiles for both development and production. 

To run the container locally:
* Open a terminal and go to the location where `src/frontend`
* Run `sh docker-shell.sh`
* Once inside the docker container, run `npm install`
* Once `npm` is installed, run `npm start`
* Go to `http://localhost:8080` to access the app locally

### Deployment Container
This container helps manage building and deploying all our app containers. This can be achieved with Ansible, with or without Kubernetes.

To run the container locally:
* Open a terminal and go to the location `AC215_ScienceTutor/src/app_deploy`
* Run `sh docker-shell.sh`

#### Deploy with Ansible and Kubernetes

* Build and Push Docker Containers to GCR
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

* Create and Deploy Cluster
```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
```
Once the command runs go to `http://<YOUR INGRESS IP>.sslip.io`

#### Deploy with Ansible

* Build and Push Docker Containers to GCR
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

* Create Compute Instance (VM) Server in GCP
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```

* Provision Compute Instance in GCP
Install and setup all the required things for deployment.
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

* Setup Docker Containers in the Compute Instance
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```

* Setup Webserver on the Compute Instance
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```
Once the command runs go to `http://<External IP>` 

---
