from PIL import Image
import os
import requests
import json
import re
import time
from pdf2image import convert_from_path
from dotenv import load_dotenv, find_dotenv
import base64
from io import BytesIO

load_dotenv(find_dotenv())

def remove_html_comment(text):
    cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return cleaned_text

def remove_tables(md_text: str) -> str:
    table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
    cleaned_text = re.sub(table_pattern, '', md_text)
    return cleaned_text.strip()

def extract_html_tables(html):
    table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
    tables = table_regex.findall(html)
    return tables

def image_to_base64(image: Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def load_timing_data(filename="llm_timing_data.json"):
    """Load existing timing data from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_timing_data(data, filename="llm_timing_data.json"):
    """Save timing data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

class OpenRouterOCR:
    def __init__(self, max_tokens=2000, model_name="google/gemini-2.5-flash"):
        self.max_tokens = max_tokens
        self.tmp = f"{os.getcwd()}/openroutertmp"
        self.api_key = os.environ["API_KEY_REF"]
        self.model_name = model_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timing_data = []  # Store timing data for this session
        os.makedirs(self.tmp, exist_ok=True)

    def _answer(self, prompt: str, image: Image) -> str:
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        messages = [
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
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            # "max_tokens": self.max_tokens
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

    def __call__(self, prompt: str, pdf_path: str):
        images = convert_from_path(pdf_path)
        label_path = pdf_path.replace(".pdf", ".md").replace("pdfs/", "labels/")
        pred_str = ""
        for idx, img in enumerate(images):
            try:
                pred = self._answer(prompt, img)
                pred = pred.replace("```markdown", "").replace("```html", "").replace("```", "")
                pred = remove_html_comment(pred)
            except Exception as e:
                print(f"Skipping sample {pdf_path}:{idx} due to error: {e}")
                pred = ""
            pred_str += pred
        
        # Save timing statistics after processing all pages
        self.save_timing_stats()
        
        pred_text, pred_tables = remove_tables(pred_str), extract_html_tables(pred_str)
        return pred_text, pred_tables