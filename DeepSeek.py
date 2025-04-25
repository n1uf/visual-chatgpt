import os
import json
import logging
import requests
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Generator, AsyncGenerator, Set
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from pydantic import Field, root_validator

logger = logging.getLogger(__name__)


def update_token_usage(keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]) -> None:
    _keys_to_use = keys.intersection(response.get("usage", {}))
    for _key in _keys_to_use:
        token_usage[_key] = token_usage.get(_key, 0) + response["usage"][_key]


def _streaming_response_template() -> Dict[str, Any]:
    return {
        "choices": [
            {
                "text": "",
                "finish_reason": None,
                "logprobs": None,
            }
        ]
    }


def _update_response(response: Dict[str, Any], stream_response: Dict[str, Any]) -> None:
    response["choices"][0]["text"] += stream_response["choices"][0]["text"]
    response["choices"][0]["finish_reason"] = stream_response["choices"][0]["finish_reason"]
    response["choices"][0]["logprobs"] = stream_response["choices"][0]["logprobs"]


class BaseDeepSeek(BaseLLM):
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    deepseek_api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    n: int = 1
    best_of: int = 1
    request_timeout: Optional[Union[float, Tuple[float, float]]] = 30.0
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    streaming: bool = False
    max_retries: int = 6
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "ignore"

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "deepseek_api_key" not in values or not values["deepseek_api_key"]:
            env_key = os.getenv("DEEPSEEK_API_KEY")
            if env_key:
                values["deepseek_api_key"] = env_key
            else:
                raise ValueError("未提供 DeepSeek API Key，请设置 `deepseek_api_key` 或环境变量 `DEEPSEEK_API_KEY`")

        all_required_field_names = {field.alias for field in cls.__fields__.values()}
        extra = values.get("model_kwargs", {})

        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(f"WARNING! {field_name} is not a default parameter. Moving to model_kwargs.")
                extra[field_name] = values.pop(field_name)

        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "best_of": self.best_of,
            "logit_bias": self.logit_bias,
            **self.model_kwargs,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self._default_params
        }
        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            res = response.json()
            return res["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise RuntimeError(f"DeepSeek API 请求失败: {str(e)}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}

        for prompt in prompts:
            if self.streaming:
                response = _streaming_response_template()
                for stream_resp in self._stream_completion(prompt, stop):
                    self.callback_manager.on_llm_new_token(
                        stream_resp["choices"][0]["text"],
                        verbose=self.verbose,
                        logprobs=stream_resp["choices"][0]["logprobs"],
                    )
                    _update_response(response, stream_resp)
                generations.append([Generation(text=response["choices"][0]["text"])])
            else:
                result = self._call(prompt, stop)
                generations.append([Generation(text=result)])

        return LLMResult(generations=generations, llm_output={"token_usage": token_usage})

    def _stream_completion(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **self._default_params
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.request_timeout
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.strip() == "data: [DONE]":
                    break
                if decoded.startswith("data:"):
                    yield json.loads(decoded[len("data:"):])

    async def _async_call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self._default_params
        }
        if stop:
            payload["stop"] = stop

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def _async_stream_completion(self, prompt: str, stop: Optional[List[str]] = None) -> AsyncGenerator:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **self._default_params
        }
        if stop:
            payload["stop"] = stop

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.strip() == "data: [DONE]":
                            break
                        if decoded.startswith("data:"):
                            yield json.loads(decoded[len("data:"):])

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}

        for prompt in prompts:
            if self.streaming:
                response = _streaming_response_template()
                async for stream_resp in self._async_stream_completion(prompt, stop):
                    if self.callback_manager.is_async:
                        await self.callback_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    else:
                        self.callback_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    _update_response(response, stream_resp)
                generations.append([Generation(text=response["choices"][0]["text"])])
            else:
                result = await self._async_call(prompt, stop)
                generations.append([Generation(text=result)])

        return LLMResult(generations=generations, llm_output={"token_usage": token_usage})

    @property
    def _llm_type(self) -> str:
        return "deepseek"


class DeepSeekAI(BaseDeepSeek):
    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._default_params}
