import asyncio
import json
import logging
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
from abc import ABC, abstractmethod
from anthropic import AsyncAnthropic

# Cấu hình Logging
logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. MULTI-PROVIDER INTERFACE
# ==========================================
class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, system_prompt: str, user_prompt: str) -> Dict:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    def _extract_json(self, text: str) -> Dict:
        """Hàm dùng chung để bóc tách JSON từ chuỗi phản hồi của LLM"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("Không tìm thấy cấu trúc JSON trong phản hồi.")
            return json.loads(text[start:end])
        except Exception as e:
            logging.error(f"Lỗi phân tách JSON: {e} | Nội dung: {text}")
            return {"error": "json_parse_error", "raw": text}

# --- OpenAI Provider ---
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def health_check(self):
        print(f"\n--- Checking OpenAI ({self.model}) ---")
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10
            )
            print("✅ OpenAI API OK")
            print(f"Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ OpenAI Error: {e}")
            return False

    async def generate_response(self, system_prompt: str, user_prompt: str) -> Dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # OpenAI hỗ trợ ép kiểu JSON
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

# --- Gemini Provider ---
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    async def health_check(self):
        print(f"\n--- Checking Gemini ({self.model_name}) ---")
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, "ping")
            print("✅ Gemini API OK")
            print(f"Response: {response.text}")
            return True
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            return False

    async def generate_response(self, system_prompt: str, user_prompt: str) -> Dict:
        loop = asyncio.get_event_loop()
        combined_prompt = f"{system_prompt}\n\nUser Input: {user_prompt}"
        response = await loop.run_in_executor(None, self.model.generate_content, combined_prompt)
        return self._extract_json(response.text)

# --- Anthropic Provider ---
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def health_check(self):
        print(f"\n--- Checking Anthropic ({self.model}) ---")
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}]
            )
            print("✅ Anthropic API OK")
            print(f"Response: {response.content[0].text}")
            return True
        except Exception as e:
            print(f"❌ Anthropic Error: {e}")
            return False

    async def generate_response(self, system_prompt: str, user_prompt: str) -> Dict:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt, 
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1
        )
        return self._extract_json(response.content[0].text)

# ==========================================
# RUN HEALTH CHECK
# ==========================================
async def run_checks_1():
    # OpenAI
    if os.environ.get('OPENAI_API_KEY'):
        openai_llm = OpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        await openai_llm.health_check()
    
    # Anthropic
    if os.environ.get('ANTHROPIC_API_KEY'):
        anthropic_llm = AnthropicProvider(api_key=os.environ['ANTHROPIC_API_KEY'])
        await anthropic_llm.health_check()

async def run_checks_2():
    system_p = "You are a helpful assistant. Always respond in valid JSON format."
    user_p = "Return a JSON object with a 'status': 'connected' and 'model': 'your_name'"

    print("\n" + "="*50)
    print("TESTING GENERATE_RESPONSE (JSON OUTPUT)")
    print("="*50)

    # --- OpenAI ---
    if os.environ.get('OPENAI_API_KEY'):
        try:
            openai_llm = OpenAIProvider(api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o-mini")
            res = await openai_llm.generate_response(system_p, user_p)
            print(f"✅ OpenAI Result: {res} | Type: {type(res)}")
        except Exception as e:
            print(f"❌ OpenAI Gen Error: {e}")

    # --- Anthropic ---
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            anthropic_llm = AnthropicProvider(api_key=os.environ['ANTHROPIC_API_KEY'])
            res = await anthropic_llm.generate_response(system_p, user_p)
            print(f"✅ Anthropic Result: {res} | Type: {type(res)}")
        except Exception as e:
            print(f"❌ Anthropic Gen Error: {e}")

if __name__ == '__main__':
    print("TEST LLM PROVIDER")
    asyncio.run(run_checks_1())
    asyncio.run(run_checks_2())
    print('=' * 10 + "ALL TESTS PASSED !!" + "="*10)