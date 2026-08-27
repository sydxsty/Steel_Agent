from openai import OpenAI, AsyncOpenAI
from env_utils import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
)


def _to_openai_messages(messages):
    result = []
    for m in messages:
        if isinstance(m, str):
            result.append({"role": "user", "content": m})
            continue
        role = "user"
        cls_name = type(m).__name__
        if cls_name == "SystemMessage":
            role = "system"
        elif cls_name == "AIMessage":
            role = "assistant"
        result.append({"role": role, "content": getattr(m, "content", str(m))})
    return result


class OfficialLLMResult:
    def __init__(self, content, reasoning_content, raw_metadata=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.raw_metadata = raw_metadata or {}


class OfficialDeepSeekSync:
    def __init__(self, api_key=None, base_url=None):
        self._client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
            timeout=180,
        )
        self._model = DEEPSEEK_MODEL

    def invoke(self, messages, **kwargs):
        openai_messages = _to_openai_messages(messages)
        # OpenAI 兼容 SDK 默认会自动重试网络错误。调用方可按业务阶段覆盖，
        # 避免一次长超时被 SDK 静默重复多次后表现为十几分钟无响应。
        max_retries = kwargs.pop("max_retries", None)
        client = (
            self._client.with_options(max_retries=max_retries)
            if max_retries is not None
            else self._client
        )
        resp = client.chat.completions.create(model=self._model, messages=openai_messages, **kwargs)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        content = getattr(msg, 'content', '') or ''
        reasoning = getattr(msg, 'reasoning_content', '') or ''
        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens}
        return OfficialLLMResult(
            content=content,
            reasoning_content=reasoning,
            raw_metadata={
                "model": resp.model,
                "usage": usage,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_id": getattr(resp, "id", None),
            },
        )


class OfficialDeepSeekAsync:
    def __init__(self, api_key=None, base_url=None):
        self._client = AsyncOpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
            timeout=180,
        )
        self._model = DEEPSEEK_MODEL

    async def invoke(self, messages, **kwargs):
        openai_messages = _to_openai_messages(messages)
        resp = await self._client.chat.completions.create(model=self._model, messages=openai_messages, **kwargs)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        content = getattr(msg, 'content', '') or ''
        reasoning = getattr(msg, 'reasoning_content', '') or ''
        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens}
        return OfficialLLMResult(
            content=content,
            reasoning_content=reasoning,
            raw_metadata={
                "model": resp.model,
                "usage": usage,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_id": getattr(resp, "id", None),
            },
        )

    async def astream(self, messages, **kwargs):
        openai_messages = _to_openai_messages(messages)
        stream = await self._client.chat.completions.create(model=self._model, messages=openai_messages, stream=True, **kwargs)
        full_reasoning = ''
        full_content = ''
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            r = getattr(delta, 'reasoning_content', None)
            c = getattr(delta, 'content', None)
            if r:
                full_reasoning += r
                yield {'type': 'reasoning', 'text': r}
            if c:
                full_content += c
                yield {'type': 'content', 'text': c}
        yield {'type': 'done', 'full_reasoning': full_reasoning, 'full_content': full_content}


class OfficialQwenSync:
    def __init__(self, api_key=None, base_url=None):
        self._client = OpenAI(
            api_key=api_key or QWEN_API_KEY,
            base_url=base_url or QWEN_BASE_URL,
            timeout=180,
        )
        self._model = QWEN_MODEL

    def invoke(self, messages, **kwargs):
        openai_messages = _to_openai_messages(messages)
        # max_retries is an OpenAI client option, not a chat completion
        # request parameter. Remove it before calling Completions.create().
        max_retries = kwargs.pop("max_retries", None)
        client = (
            self._client.with_options(max_retries=max_retries)
            if max_retries is not None
            else self._client
        )
        resp = client.chat.completions.create(model=self._model, messages=openai_messages, **kwargs)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        content = getattr(msg, 'content', '') or ''
        reasoning = getattr(msg, 'reasoning_content', '') or ''
        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens}
        return OfficialLLMResult(
            content=content,
            reasoning_content=reasoning,
            raw_metadata={
                "model": resp.model,
                "usage": usage,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_id": getattr(resp, "id", None),
            },
        )


class OfficialQwenAsync:
    def __init__(self, api_key=None, base_url=None):
        self._client = AsyncOpenAI(
            api_key=api_key or QWEN_API_KEY,
            base_url=base_url or QWEN_BASE_URL,
            timeout=180,
        )
        self._model = QWEN_MODEL

    async def invoke(self, messages, **kwargs):
        openai_messages = _to_openai_messages(messages)
        resp = await self._client.chat.completions.create(model=self._model, messages=openai_messages, **kwargs)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        content = getattr(msg, 'content', '') or ''
        reasoning = getattr(msg, 'reasoning_content', '') or ''
        usage = None
        if hasattr(resp, 'usage') and resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens}
        return OfficialLLMResult(
            content=content,
            reasoning_content=reasoning,
            raw_metadata={
                "model": resp.model,
                "usage": usage,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_id": getattr(resp, "id", None),
            },
        )


official_deepseek_sync = OfficialDeepSeekSync()
official_deepseek_async = OfficialDeepSeekAsync()
official_qwen_sync = OfficialQwenSync()
official_qwen_async = OfficialQwenAsync()
