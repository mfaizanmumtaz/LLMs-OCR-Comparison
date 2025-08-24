import base64
import io
from PIL import Image
import requests

class GPT4oOCR:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        self.model_name = kwargs["model_name"]

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



    def __call__(self, prompt, image):
        messages = self._get_messages(image, prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            # "max_tokens": self.max_tokens
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        out_text = response.choices[0].message.content
        print(out_text)
        return out_text
