from typing import Dict, Any, List, Optional, Mapping, Union, Tuple, Generator, AsyncGenerator, Set
from langchain.llms.base import BaseLLM
from pydantic import Field, root_validator
import asyncio
from langchain.schema import LLMResult, Generation
import requests
import logging
import json

logger = logging.getLogger(__name__)


def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _streaming_response_template() -> Dict[str, Any]:
    """Template for streaming response."""
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
    """Update response from the stream response."""
    response["choices"][0]["text"] += stream_response["choices"][0]["text"]
    response["choices"][0]["finish_reason"] = stream_response["choices"][0][
        "finish_reason"
    ]
    response["choices"][0]["logprobs"] = stream_response["choices"][0]["logprobs"]


class BaseDeepSeek(BaseLLM):
    """Base class for DeepSeek API 模型封装，兼容 OpenAI 的接口"""

    model_name: str = "deepseek-chat"
    """默认使用 deepseek-chat"""
    base_url: str = "https://api.deepseek.com/"
    """DeepSeek API 地址"""
    deepseek_api_key: Optional[str] = None
    """API Key"""
    temperature: float = 0.7
    max_tokens: int = 1024
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
        """Pydantic 配置"""
        extra = "ignore"

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """构造 model_kwargs 以兼容 OpenAI 格式"""

            
        all_required_field_names = {field.alias for field in cls.__fields__.values()}
        extra = values.get("model_kwargs", {})

        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"WARNING! {field_name} is not a default parameter. "
                    f"Moving {field_name} to model_kwargs. "
                    f"Ensure this is intended."
                )
                extra[field_name] = values.pop(field_name)

        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """DeepSeek API 调用默认参数"""
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
        """单次 API 请求"""
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
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            return f"DeepSeek API 请求失败: {str(e)}"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """支持批量 generate()，并返回符合 langchain 规范的 LLMResult"""
        generations = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}

        for prompt in prompts:
            if self.streaming:
                # 流式处理逻辑
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
                # 非流式处理逻辑
                response = self._call(prompt, stop)
                generations.append([Generation(text=response)])
                if isinstance(response, dict) and "usage" in response:
                    update_token_usage(_keys, response, token_usage)

        return LLMResult(generations=generations, llm_output={"token_usage": token_usage})

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """异步批量请求 DeepSeek API"""
        generations = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}

        for prompt in prompts:
            if self.streaming:
                # 异步流式处理逻辑
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
                # 非流式处理逻辑
                response = await self._async_call(prompt, stop)
                generations.append([Generation(text=response)])
                if isinstance(response, dict) and "usage" in response:
                    update_token_usage(_keys, response, token_usage)

        return LLMResult(generations=generations, llm_output={"token_usage": token_usage})

    def _stream_completion(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        """流式生成文本"""
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
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.request_timeout
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode("utf-8"))

    async def _async_stream_completion(self, prompt: str, stop: Optional[List[str]] = None) -> AsyncGenerator:
        """异步流式生成文本"""
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
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    yield json.loads(line.decode("utf-8"))

    async def _async_call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """异步单次 API 请求"""
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
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "deepseek"


class DeepSeekAI(BaseDeepSeek):
    """DeepSeek AI LLM 适配 OpenAI"""

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._default_params}