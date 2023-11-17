import io
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
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

# Load pre-trained model and tokenizer
tokenizer, model, image_processor, context_len = load_pretrained_model(
        "cnut1648/llava-v1.5-7b", 
        model_name="llava-v1.5-7b", 
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
        cur_prompt = request.form.get("prompt")
        temperature = float(request.form.get("temperature", 1.0))
        print(request.form)
        history = request.form.getlist("history")
        print("Provided history: ", history)

        # Check if an image is provided and process it
        if 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
            print(image)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
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

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        outputs_reasoning = outputs
        input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=64,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        outputs = outputs_reasoning + '\n The answer is ' + outputs

        # # Tokenize the prompt
        # inputs = tokenizer.encode(prompt, return_tensors='pt')

        # # Generate a response from the model
        # outputs = model.generate(inputs, max_length=100, temperature=temperature)
        # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # # remove prompt from response
        # response_text = response_text.replace(prompt, "")

        # Return the response
        return jsonify({"response": outputs})

# Initialize the Flask application
app = Flask(__name__)

swagger = Swagger(app)

app.add_url_rule('/chat', view_func=ChatView.as_view('chat'), methods=['POST'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)