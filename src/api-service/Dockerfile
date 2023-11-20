# ENV
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y git wget jq

# APP
RUN \
  pip --no-cache-dir install transformers==4.34.0 datasets einops pyyaml colorama markdown pandas Pillow requests scipy \
    sentencepiece flask flasgger flask-cors tqdm bitsandbytes protobuf marshmallow apispec accelerate;
WORKDIR /app
RUN git clone https://github.com/cnut1648/LLaVA.git
COPY api/model_backend.py .
EXPOSE 5000
CMD ["python", "model_backend.py"]