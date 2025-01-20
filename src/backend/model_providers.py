from enum import Enum
import base64
import io
from PIL import Image
import time
from openai import OpenAI
import fireworks.client
from abc import ABC, abstractmethod
import cv2
import json


class ModelProvider(Enum):
    OPENAI = "OpenAI"
    FIREWORKS = "Fireworks"


class VisionModel(ABC):
    @staticmethod
    def _encode_image(frame):
        """Convert CV2 frame to base64 string"""
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @abstractmethod
    def analyze_frame(self, frame, prompt):
        pass


class OpenAIVisionModel(VisionModel):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_name = None  # Will be set when model is selected
        
    @property
    def available_models(self):
        return {
            "GPT-4o Mini": "gpt-4o-mini",
            "GPT-4o": "gpt-4o"
        }
    
    def analyze_frame(self, frame, prompt):
        base64_image = self._encode_image(frame)
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=300,
                response_format={ "type": "json_object" },  # Enable JSON mode
                seed=42  # Optional: for consistent outputs
            )
            inference_time = time.time() - start_time
            
            # Response will already be in JSON format
            return json.loads(response.choices[0].message.content), inference_time
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class FireworksVisionModel(VisionModel):
    def __init__(self, api_key):
        fireworks.client.api_key = api_key
        self.model_name = None  # Will be set when model is selected
        
    @property
    def available_models(self):
        return {
            "Llama-3 11B Vision": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
            "Llama-3 90B Vision": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            "Phi-3 Vision": "accounts/fireworks/models/phi-3-vision-128k-instruct",
        }
    
    def analyze_frame(self, frame, prompt):
        base64_image = self._encode_image(frame)
        start_time = time.time()
        
        try:
            response = fireworks.client.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing video frames."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                response_format={ "type": "json_object" }  # Enable JSON mode for Fireworks
            )
            inference_time = time.time() - start_time
            
            # Parse the response content as JSON
            content = response.choices[0].message.content
            # Remove any leading/trailing whitespace and handle potential string formatting
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:-3]  # Remove ```json and ``` markers
            elif content.startswith('{'):
                content = content  # Already clean JSON
                
            return json.loads(content), inference_time
            
        except Exception as e:
            raise Exception(f"Fireworks API error: {str(e)}")


def get_model_provider(provider: ModelProvider, api_key: str) -> VisionModel:
    if provider == ModelProvider.OPENAI:
        return OpenAIVisionModel(api_key)
    elif provider == ModelProvider.FIREWORKS:
        return FireworksVisionModel(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}") 