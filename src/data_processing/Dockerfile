FROM python:3.9-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt .
COPY convert_scienceqa_to_llava.py .

RUN pip install -r requirements.txt

RUN python convert_scienceqa_to_llava.py
#
#COPY ScienceQA docker_processed/ScienceQA
#COPY ScienceQA-LLAVA docker_processed/ScienceQA-LLAVA

CMD ["bash"]
