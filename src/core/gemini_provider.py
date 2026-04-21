import os
import time
from openai import OpenAI
from typing import Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model_name, api_key)
        # Ưu tiên tham số truyền vào, nếu không có mới lấy từ .env
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        
        print(f"[GeminiProvider] Initializing with Base URL: {self.base_url}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        
        # Prepare messages in OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Retry with backoff for rate limits
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                break
            except Exception as e:
                # Retain old retry logic
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 20 * (attempt + 1)
                    print(f"[Gemini] Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Map OpenAI-compatible response back to the required structure
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
