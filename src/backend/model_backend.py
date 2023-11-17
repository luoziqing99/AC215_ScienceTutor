import io

from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flasgger import Swagger
from PIL import Image
from flask import Flask, jsonify
from flasgger import Swagger, SwaggerView, Schema, fields


# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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
        prompt = request.form.get("prompt")
        temperature = float(request.form.get("temperature", 1.0))
        print(request.form)
        history = request.form.getlist("history")
        print("Provided history: ", history)

        # Check if an image is provided and process it
        if 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(io.BytesIO(image_file.read()))
            print(image)

        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate a response from the model
        outputs = model.generate(inputs, max_length=100, temperature=temperature)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # remove prompt from response
        response_text = response_text.replace(prompt, "")

        # Return the response
        return jsonify({"response": response_text})


# Initialize the Flask application
app = Flask(__name__)

swagger = Swagger(app)

app.add_url_rule('/chat', view_func=ChatView.as_view('chat'), methods=['POST'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)