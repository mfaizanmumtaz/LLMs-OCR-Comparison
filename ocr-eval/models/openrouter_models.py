import base64
import io
import json
import os
import time
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

def load_timing_data(filename="llm_timing_data.json"):
    """Load existing timing data from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_timing_data(data, filename="llm_timing_data.json"):
    """Save timing data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
plugins = [
    {
        "id": "file-parser",
        "pdf": {
            "engine": "mistral-ocr"
        }
    }
]
class OpenRouterModel:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        self.model_name = kwargs["model_name"]
        self.api_key = os.getenv("API_KEY_REF")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timing_data = []  # Store timing data for this session
    
    def _enc(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_messages(self, image: Image, prompt):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_bytes = buf.read()
        image_base64 = self._enc(image_bytes)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

    def save_timing_stats(self):
        """Save timing statistics to JSON file"""
        if not self.timing_data:
            return
            
        # Calculate average
        avg_time = sum(self.timing_data) / len(self.timing_data)
        
        # Load existing data
        all_timing_data = load_timing_data()
        
        # Update with current model data
        all_timing_data[self.model_name] = {
            "call_times": self.timing_data,
            "average_time": round(avg_time, 3),
            "total_calls": len(self.timing_data)
        }
        
        # Save updated data
        save_timing_data(all_timing_data)
        
        print(f"\nTiming stats for {self.model_name}:")
        print(f"Total calls: {len(self.timing_data)}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Call times: {self.timing_data}")
        print(f"Data saved to llm_timing_data.json")

    def __call__(self, prompt, image):
        messages = self._get_messages(image, prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            # "plugins": plugins
        }
        
        # Time the LLM call
        start_time = time.time()
        response = requests.post(self.url, headers=self.headers, json=payload)
        end_time = time.time()
        
        # Calculate and store timing
        call_time = round(end_time - start_time, 3)  # Keep in seconds with 3 decimal places
        self.timing_data.append(call_time)
        print(f"LLM call took: {call_time}s")
        
        # Try to parse JSON and handle unexpected responses
        try:
            response_data = response.json()
        except Exception as e:
            # Non-JSON response (or empty). Include raw text for debugging.
            raise Exception(f"OpenRouter returned non-JSON response: {e}; text={response.text}")

        # If the API returned an error status, include the body for debugging
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error {response.status_code}: {response_data}")

        # Extract content from OpenRouter response format
        if not isinstance(response_data, dict) or 'choices' not in response_data or not response_data['choices']:
            # Unexpected format - surface the whole response so caller can log it
            raise Exception(f"Unexpected OpenRouter response format (missing 'choices'): {response_data}")

        out_text = response_data['choices'][0].get('message', {}).get('content')
        if out_text is None:
            # If content is absent, show full response for debugging
            raise Exception(f"OpenRouter response has no message content: {response_data}")

        print(out_text)
        return out_text

    def __del__(self):
        """Save timing stats when the object is destroyed"""
        try:
            self.save_timing_stats()
        except:
            pass  # Ignore errors during cleanup