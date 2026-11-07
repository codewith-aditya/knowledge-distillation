"""
Multi-provider LLM client with streaming support
Supports OpenAI-compatible APIs
"""

import asyncio
import time
from typing import AsyncIterator, Optional, List, Dict, Any
import httpx
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
        max_retries: int = 5,
        retry_base_delay: float = 2.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        self.total_usage = TokenUsage()
        self.request_count = 0

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            trust_env=False,  # ignore HTTP_PROXY/HTTPS_PROXY env vars — raw-IP endpoints
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat completion request

        Returns:
            {
                "content": "response text",
                "usage": TokenUsage,
                "model": "model_name",
                "finish_reason": "stop"
            }
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream
        }

        for attempt in range(self.max_retries):
            try:
                if stream:
                    return await self._stream_chat(payload)
                else:
                    return await self._non_stream_chat(payload)

            except Exception as e:
                # Only retry transient transport errors (timeouts, connection
                # resets, 5xx via raise_for_status). Request-construction bugs
                # (TypeError, LocalProtocolError, JSON serialization) fail the
                # same way every time — retrying just burns 30s of sleeps.
                transient = isinstance(e, httpx.TransportError) or (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code >= 500
                )
                if not transient or attempt == self.max_retries - 1:
                    raise

                delay = self.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    def _headers(self) -> dict:
        """Build request headers — omit Authorization entirely when no API key
        is set ("Bearer " with empty key is an illegal header in httpx)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _non_stream_chat(self, payload: dict) -> Dict[str, Any]:
        """Non-streaming chat completion"""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()

        data = response.json()

        self.request_count += 1

        usage = TokenUsage(
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0)
        )

        self.total_usage += usage

        return {
            # Reasoning models (MiniMax M3, DeepSeek-R1 style) may return
            # content=null when all tokens went to the reasoning field.
            "content": data["choices"][0]["message"].get("content") or "",
            "reasoning": data["choices"][0]["message"].get("reasoning") or "",
            "usage": usage,
            "model": data.get("model", self.model),
            "finish_reason": data["choices"][0].get("finish_reason", "stop")
        }

    async def _stream_chat(self, payload: dict) -> Dict[str, Any]:
        """Streaming chat completion"""
        full_content = []

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers()
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or line.strip() == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        import json
                        chunk = json.loads(line[6:])

                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                full_content.append(delta["content"])
                    except:
                        continue

        self.request_count += 1

        content = "".join(full_content)

        estimated_usage = TokenUsage(
            input_tokens=len(payload["messages"]) * 100,
            output_tokens=len(content) // 4,
            total_tokens=(len(payload["messages"]) * 100) + (len(content) // 4)
        )

        self.total_usage += estimated_usage

        return {
            "content": content,
            "usage": estimated_usage,
            "model": self.model,
            "finish_reason": "stop"
        }

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "total_requests": self.request_count,
            "total_input_tokens": self.total_usage.input_tokens,
            "total_output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total_tokens
        }
