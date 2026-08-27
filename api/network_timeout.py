"""
network_timeout.py — 强制所有 HTTP 请求超时的底层拦截模块
============================================================

必须在所有 HuggingFace / sentence-transformers / requests 导入之前执行！

为什么需要这个模块：
  1. Python 的 socket.setdefaulttimeout() 对 requests 库无效
     — requests.Session 有自己独立的 timeout 参数，默认 None（无限等）
  2. 环境变量 HF_HUB_DOWNLOAD_TIMEOUT 被 sentence-transformers 绕过
     — sentence-transformers 内部下载不走 huggingface_hub 标准路径
  3. Python 线程无法被强制杀死
     — ThreadPoolExecutor.future.result(timeout) 只能停止等待，杀不死 C 层阻塞 I/O

解决方案：
  猴子补丁（Monkey Patch）requests.Session.request()，在每次 HTTP 调用时强制
  注入 timeout=5，确保网络不可达时 5 秒内必定抛出异常。

用法：
  from network_timeout import enforce_timeout
  enforce_timeout(timeout=5)  # 全局生效，设置后所有 HTTP 请求 5 秒超时
"""

import os
import types
from typing import Optional
from urllib.parse import urlparse

_ENFORCED = False
_DEFAULT_TIMEOUT = 5
_LLM_API_TIMEOUT = 180
_LLM_API_HOST_KEYWORDS = (
    "api.deepseek.com",
    "dashscope.aliyuncs.com",
)


def _is_llm_api_url(url) -> bool:
    """判断请求是否发往 DeepSeek/Qwen 官方兼容接口，避免被全局 5 秒超时误伤。"""
    try:
        host = urlparse(str(url)).netloc.lower()
    except Exception:
        host = str(url).lower()
    return any(keyword in host for keyword in _LLM_API_HOST_KEYWORDS)


def enforce_timeout(timeout: Optional[float] = 5):
    """
    猴子补丁 requests 库，强制所有 HTTP 请求带上超时。

    必须在所有使用 requests 的第三方库（huggingface_hub、sentence_transformers 等）
    导入之前调用。

    Args:
        timeout: 超时秒数，None 表示不强制（恢复默认行为）
    """
    global _ENFORCED, _DEFAULT_TIMEOUT

    if _ENFORCED:
        return

    _DEFAULT_TIMEOUT = timeout

    # 方案1：设置环境变量（能拦截一部分路径）
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout)
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ.get("REQUESTS_CA_BUNDLE", "")

    # 方案2：猴子补丁 requests — 强制 timeout + 禁用重试
    try:
        import requests
        from requests.adapters import HTTPAdapter

        _original_request = requests.Session.request

        def _patched_request(self, method, url, **kwargs):
            kwargs.setdefault("timeout", _LLM_API_TIMEOUT if _is_llm_api_url(url) else timeout)
            return _original_request(self, method, url, **kwargs)

        requests.Session.request = _patched_request

        # 拦截 Session.__init__，禁用适配器重试
        _original_init = requests.Session.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            for prefix, adapter in self.adapters.items():
                adapter.max_retries = 0
            _orig_mount = self.mount
            def _patched_mount(prefix, adapter):
                adapter.max_retries = 0
                return _orig_mount(prefix, adapter)
            self.mount = _patched_mount

        requests.Session.__init__ = _patched_init

        # 禁用 urllib3 全局重试
        try:
            import urllib3
            urllib3.util.Retry.DEFAULT = 0
        except Exception:
            pass

        print(f"[network_timeout] 已拦截 requests（timeout={timeout}s, retries=0）")
        _ENFORCED = True

    except ImportError:
        print("[network_timeout] requests 未安装，跳过拦截")
    except Exception as e:
        print(f"[network_timeout] requests 拦截失败: {e}")

    # 方案3：猴子补丁 httpx — huggingface_hub 新版本已迁移到 httpx
    try:
        import httpx

        _original_httpx_request = httpx.Client.request

        def _patched_httpx_request(self, method, url, **kwargs):
            kwargs.setdefault("timeout", _LLM_API_TIMEOUT if _is_llm_api_url(url) else timeout)
            return _original_httpx_request(self, method, url, **kwargs)

        httpx.Client.request = _patched_httpx_request
        print(f"[network_timeout] 已拦截 httpx（timeout={timeout}s）")
    except ImportError:
        print("[network_timeout] httpx 未安装，跳过拦截")
    except Exception as e:
        print(f"[network_timeout] httpx 拦截失败: {e}")


def get_timeout() -> float:
    """获取当前的强制超时值"""
    return _DEFAULT_TIMEOUT
