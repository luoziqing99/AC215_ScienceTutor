# ENV
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git wget jq

# APP
WORKDIR /app
RUN \
  pip install transformers datasets einops gradio peft pyyaml colorama markdown pandas Pillow requests scipy \
    sentencepiece tqdm torch-grammar bitsandbytes protobuf;
RUN \
   git clone https://github.com/cnut1648/text-generation-webui.git && \
    cd text-generation-webui && \
     git checkout llavav1.5-7b;
WORKDIR /app/text-generation-webui
RUN \
    python download-model.py cnut1648/llava-v1.5-7b;
RUN \
   cd models/cnut1648_llava-v1.5-7b && \
    jq '.model_type |= if . == "llava" then "llama" else . end' config.json > tmp.json && mv tmp.json config.json
EXPOSE 7860 5000 5005
CMD ["python", "server.py", "--model", "cnut1648_llava-v1.5-7b", "--multimodal-pipeline", "llava-v1.5-7b", "--load-in-4bit", "--listen", "--listen-host", "0.0.0.0"]