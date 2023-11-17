# ENV
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y git wget jq

#CMD ["bash"]

# APP
WORKDIR /app
RUN \
  pip --no-cache-dir install transformers Pillow requests scipy \
    sentencepiece tqdm torch-grammar bitsandbytes protobuf flask flasgger;
RUN \
   git clone https://github.com/cnut1648/LLaVA.git && \
    cd /LLaVA;
WORKDIR /app/LLaVA
EXPOSE 7860
ENTRYPOINT ["/bin/bash"]