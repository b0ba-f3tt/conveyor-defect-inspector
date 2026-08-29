import base64
import requests


class QwenVLMProvider:
    def __init__(self, base_url, model_name, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def _image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def ask(self, question, image_path=None):
        content = []

        if question:
            content.append({
                "type": "text",
                "text": question
            })

        if image_path:
            image_base64 = self._image_to_base64(image_path)

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.2
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]