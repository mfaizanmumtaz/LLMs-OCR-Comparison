from PIL import Image
import os
import requests
import json
import re
from pdf2image import convert_from_path
from dotenv import load_dotenv, find_dotenv
import base64
from io import BytesIO

load_dotenv(find_dotenv())

# def remove_html_comment(text):
#     cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
#     return cleaned_text

# def remove_tables(md_text: str) -> str:
#     table_pattern = re.compile(r"(^\|.*\|$\n(^\|[-:]+\|$\n)?(\|.*\|$\n)*)", re.MULTILINE)
#     cleaned_text = re.sub(table_pattern, '', md_text)
#     return cleaned_text.strip()

# def extract_html_tables(html):
#     table_regex = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
#     tables = table_regex.findall(html)
#     return tables

# def image_to_base64(image: Image) -> str:
#     """Convert PIL Image to base64 string"""
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     return img_str

# class OpenRouterOCR:
#     def __init__(self, max_tokens=2000, model_name="google/gemini-2.5-flash"):
#         self.max_tokens = max_tokens
#         self.tmp = f"{os.getcwd()}/openroutertmp"
#         self.api_key = os.environ["API_KEY_REF"]
#         self.model_name = model_name
#         self.url = "https://openrouter.ai/api/v1/chat/completions"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         os.makedirs(self.tmp, exist_ok=True)

#     def _answer(self, prompt: str, image: Image) -> str:
#         # Convert image to base64
#         base64_image = image_to_base64(image)
        
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{base64_image}"
#                         }
#                     }
#                 ]
#             }
#         ]

#         payload = {
#             "model": self.model_name,
#             "messages": messages,
#             # "max_tokens": self.max_tokens
#         }

#         response = requests.post(self.url, headers=self.headers, json=payload)
#         response_data = response.json()
        
#         # Extract content from OpenRouter response format
#         out_text = response_data['choices'][0]['message']['content']
#         print(out_text)
#         return out_text

#     def __call__(self, prompt: str, pdf_path: str):
#         images = convert_from_path(pdf_path)
#         label_path = pdf_path.replace(".pdf", ".md").replace("pdfs/", "labels/")
#         pred_str = ""
#         for idx, img in enumerate(images):
#             try:
#                 pred = self._answer(prompt, img)
#                 pred = pred.replace("```markdown", "").replace("```html", "").replace("```", "")
#                 pred = remove_html_comment(pred)
#             except Exception as e:
#                 print(f"Skipping sample {pdf_path}:{idx} due to error: {e}")
#                 pred = ""
#             pred_str += pred
#         pred_text, pred_tables = remove_tables(pred_str), extract_html_tables(pred_str)
#         return pred_text, pred_tables
    


import os
import requests
import json

API_KEY_REF = os.getenv("API_KEY_REF")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "HI"
            },
            {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": "https://arxiv.org/pdf/1706.03762"
                }
            },
        ]
    }
]



payload = {
    "model": "mistralai/mistral-saba",
    "messages": messages,
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())