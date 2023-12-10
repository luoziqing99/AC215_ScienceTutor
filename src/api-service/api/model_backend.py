from pathlib import Path

from flask import Flask, request, jsonify
from flasgger import Swagger, SwaggerView, Schema, fields
from io import BytesIO
from flask_cors import CORS
import torch
import requests
pwd = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, "LLaVA")
# print(sys.path)
# print(os.environ.get("PYTHONPATH"))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
from PIL import Image

# Load pre-trained model and tokenizer
disable_torch_init()

class Prompt(Schema):
    prompt = fields.Str(required=True)
    temperature = fields.Float(required=True)
    image = fields.Str(required=False)
    history = fields.List(fields.Str, required=False)

class ChatView(SwaggerView):
    parameters = [{
        "name": "body",
        "in": "formData",
        "schema": Prompt
    }]
    consumes = ["multipart/form-data"]
    responses = {200: {"description": "A successful response from the model"},
                 400: {"description": "Invalid request format"}}

    def post(self):
        """Chat with our model!"""
        # Extract text data from form-data
        qs = request.form.get("prompt")
        temperature = float(request.form.get("temperature", 1.0))

        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        # Check if history is provided and process it
        if 'history[]' in request.form:
            history = request.form.getlist("history[]")
            assert len(history) % 2 == 0, f"History must be a list of alternating user and assistants messages, but got {history}"
            print("Provided history: ", history)
            # # assume first history always starts with user
            for i in range(0, len(history), 2):
                conv.append_message(conv.roles[0], history[i])
                conv.append_message(conv.roles[1], history[i + 1])
            # for prompt in history:
            #     role, sentence = prompt.strip(')()').split(', ')
            #     conv.append_message(role, sentence)

        # Check if an image is provided and process it
        if 'image' in request.files:
            image_file = request.files['image']
            # response = requests.get(image_file)
            # image = Image.open(BytesIO(response.content)).convert('RGB')
            image = Image.open(BytesIO(image_file.read()))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            images = None

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                top_p=1.0,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        conv.messages[-1][-1] = outputs
        
        # Return the response
        return jsonify({"response": outputs})

# Initialize the Flask application
app = Flask(__name__)

swagger = Swagger(app)
app.add_url_rule('/chat', view_func=ChatView.as_view('chat'), methods=['POST'])

# @app.route('/status', methods=['GET'])
# def get_api_status():
#     return jsonify({
#         "version": "1.0",
#         "torch_version": torch.__version__
#     })

CORS(app)

# Run the Flask app
if __name__ == '__main__':

    model_path = "cnut1648/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)  # "llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_name=model_name,
        model_base=None,
        load_8bit=False,
        load_4bit=True,
        device_map="auto",
    )

    app.run(host='0.0.0.0', port=5000, debug=False)