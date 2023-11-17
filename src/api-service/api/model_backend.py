import io
from flask import Flask, request, jsonify

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from flasgger import Swagger
from PIL import Image
from flask import Flask, jsonify
from flasgger import Swagger, SwaggerView, Schema, fields
import torch
import requests

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


# Load pre-trained model and tokenizer
disable_torch_init()

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_name=model_name, 
        model_base=None, 
        load_8bit=False, 
        load_4bit=True
)

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
        cur_prompt = qs
        temperature = float(request.form.get("temperature", 1.0))
        print(request.form)
        history = request.form.getlist("history")
        print("Provided history: ", history)

        # Check if an image is provided and process it
        if 'image' in request.files:
            image_file = request.files['image']
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
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
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        
        # Return the response
        return jsonify({"response": outputs})

# Initialize the Flask application
app = Flask(__name__)

swagger = Swagger(app)

app.add_url_rule('/chat', view_func=ChatView.as_view('chat'), methods=['POST'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)