import requests
from config import API_KEYS, BASE_URLS
import anthropic

class LLMClient:
    def __init__(self, model="openai"):
        self.model = model
        self.api_key = API_KEYS.get(model)
        self.base_url = BASE_URLS.get(model)
        self.system_prompt = "You are a chatbot tasked with assisting users understand what is written in a dataset. You will be given data in the form of a table and you will be asked questions about the data. You will need to answer the questions based on the data you have been given. Please refrain from doing anything else than answering the questions about the data."

        if not self.api_key or not self.base_url:
            raise ValueError(f"Invalid or missing configuration for model: {model}")

    def chat(self, messages, **kwargs):
        """Send chat requests to the selected LLM."""
        if self.model == "openai":
            return self._chat_openai(messages, **kwargs)
        elif self.model == "anthropic":
            return self._chat_anthropic(messages, **kwargs)
        elif self.model == "llama":
            return self._chat_llama(messages, **kwargs)
        else:
            raise ValueError("Unsupported LLM model")

    def _chat_openai(self, messages, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        #Add system prompt if no messages are present yet
        if messages == []:
            messages = [{"role": "system", "content": self.system_prompt}]
        payload = {"model": "gpt-4", "messages": messages, **kwargs}
        response = requests.post(self.base_url, json=payload, headers=headers)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content")

    def _chat_anthropic(self, messages, **kwargs):
        client = anthropic.Anthropic(
            api_key=self.api_key,
        )
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system = self.system_prompt,
            messages=messages
        )
        return message.content[0].text

    def _chat_llama(self, messages, **kwargs):
        payload = {"messages": messages, **kwargs}
        #Add system prompt if no messages are present yet
        if messages == []:
            messages = [{"role": "system", "content": self.system_prompt}]
        response = requests.post(self.base_url, json=payload)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content")
