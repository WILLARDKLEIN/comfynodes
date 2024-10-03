import openai
import base64
from PIL import Image
from io import BytesIO
import requests
from comfy.model_management import register_node


class OpenAIChatGPT4Dalle3Node:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    # ChatGPT-4 completion function
    def generate_chat_completion(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 150):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    # DALL·E 3 image generation function
    def generate_image(self, prompt: str, model: str = "dalle-3"):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            # Get the image URL
            image_url = response['data'][0]['url']

            # Download the image
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            return image
        except Exception as e:
            return f"Error: {str(e)}"


# Registering the node with ComfyUI
@register_node
class ComfyUINode(OpenAIChatGPT4Dalle3Node):
    def __init__(self):
        super().__init__(api_key="your-openai-api-key")

    # Define node inputs/outputs
    def inputs(self):
        return {
            "api_key": ("STRING", "Enter your OpenAI API Key"),
            "prompt": ("STRING", "Enter your prompt"),
            "model_type": ("STRING", "Choose 'chat' for ChatGPT or 'image' for DALL·E"),
            "temperature": ("FLOAT", "Temperature for text responses", 0.7),
            "max_tokens": ("INT", "Max tokens for text completion", 150)
        }

    def outputs(self):
        return {
            "chat_output": ("STRING", "Chat Completion"),
            "image_output": ("IMAGE", "Generated Image")
        }

    def process(self, inputs):
        prompt = inputs.get('prompt')
        model_type = inputs.get('model_type')
        api_key = inputs.get('api_key')
        self.api_key = api_key  # Update API key for this execution

        if model_type == "chat":
            result = self.generate_chat_completion(
                prompt=prompt,
                temperature=inputs.get('temperature', 0.7),
                max_tokens=inputs.get('max_tokens', 150)
            )
            return {"chat_output": result}

        elif model_type == "image":
            image = self.generate_image(prompt)
            return {"image_output": image}

        else:
            return {"chat_output": "Invalid model type specified.", "image_output": None}
