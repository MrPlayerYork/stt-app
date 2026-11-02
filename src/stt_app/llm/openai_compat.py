"""OpenAI and OpenAI-compatible (local server) backend via chat completions."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import openai

from .base import TEMPERATURE, TIMEOUT_S, LLMError, LLMResult, split_think


def _wrap(e: Exception, base_url: str | None) -> LLMError:
    where = base_url or "OpenAI"
    if isinstance(e, openai.APITimeoutError):
        return LLMError(f"Request to {where} timed out")
    if isinstance(e, openai.APIConnectionError):
        return LLMError(f"Cannot connect to {where} (is the server running?)")
    if isinstance(e, openai.AuthenticationError):
        return LLMError("Authentication failed: check the API key")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError("Permission denied by the API")
    if isinstance(e, openai.NotFoundError):
        return LLMError("Model or endpoint not found: check the model name")
    if isinstance(e, openai.RateLimitError):
        return LLMError("Rate limited or quota exceeded")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"API error {e.status_code}: {str(e.message)[:200]}")
    return LLMError(f"LLM request failed: {e}")


PROBE_TIMEOUT_S = 1.5
UNLOAD_TIMEOUT_S = 3.0
DEFAULT_PORTS = {1234: "lmstudio", 11434: "ollama"}


def _root(base_url: str) -> str:
    """Server root (scheme://host:port) for a base URL such as http://localhost:1234/v1."""
    u = urlsplit(base_url)
    return f"{u.scheme}://{u.netloc}"


def _http(method: str, url: str, api_key: str | None, body: dict | None, timeout: float) -> tuple[int, str]:
    """Tiny request helper returning (status, text); status 0 means the server was unreachable."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # 401 etc. still carries a useful body
        try:
            return e.code, e.read(65536).decode("utf-8", "replace")
        except Exception:
            return e.code, ""
    except Exception:
        return 0, ""


def detect_server(base_url: str, api_key: str | None = None) -> str:
    """Return "lmstudio", "ollama" or "other" using cheap GET probes; never raises."""
    try:
        root = _root(base_url)
        # Ollama: GET / -> "Ollama is running"; GET /api/version -> {"version": ...}
        status, text = _http("GET", root + "/", api_key, None, PROBE_TIMEOUT_S)
        if "Ollama is running" in text:
            return "ollama"
        if "LM Studio" in text:
            return "lmstudio"
        status, text = _http("GET", root + "/api/version", api_key, None, PROBE_TIMEOUT_S)
        if status == 200 and '"version"' in text:
            return "ollama"
        if "LM Studio" in text:
            return "lmstudio"
        # LM Studio native REST (answers 401 with an LM Studio message when a token is required)
        status, text = _http("GET", root + "/api/v0/models", api_key, None, PROBE_TIMEOUT_S)
        if "LM Studio" in text or (status == 200 and '"data"' in text):
            return "lmstudio"
        if status:  # something answered but is neither: vLLM, llama-server, ...
            return "other"
        return DEFAULT_PORTS.get(urlsplit(base_url).port or 0, "other")
    except Exception:
        return "other"


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        idle_ttl_seconds: int = 0,
        reasoning: str = "default",
    ) -> None:
        self.model = model
        self.reasoning = reasoning or "default"
        self.base_url = base_url or None
        self.api_key = api_key
        self.idle_ttl_seconds = max(0, int(idle_ttl_seconds or 0))
        self._kind: str | None = None
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed", base_url=self.base_url, timeout=TIMEOUT_S, max_retries=1
        )

    @property
    def server_kind(self) -> str:
        """"lmstudio" | "ollama" | "other" (detected once per backend; always "other" without a base URL)."""
        if self._kind is None:
            self._kind = detect_server(self.base_url, self.api_key) if self.base_url else "other"
        return self._kind

    def _set_ollama_keep_alive(self) -> None:
        # Ollama's /v1 endpoint does not document keep_alive; the native API does. An /api/generate call
        # without a prompt only updates the loaded model's expiry (best effort, errors ignored).
        _http("POST", _root(self.base_url) + "/api/generate", self.api_key,
              {"model": self.model, "keep_alive": self.idle_ttl_seconds}, UNLOAD_TIMEOUT_S)

    def _efforts(self) -> list[str | None]:
        # reasoning_effort candidates, tried in order. Mapping: low/medium/high pass through; "off" -> "none"
        # (OpenAI gpt-5.1+, Ollama; official OpenAI also tries "minimal" for older gpt-5 models). Servers
        # that reject the field fall through to None (nothing sent) and the result notes it.
        if self.reasoning == "default":
            return [None]
        if self.reasoning == "off":
            return ["none"] if self.base_url else ["none", "minimal"]
        return [self.reasoning]

    def _create(self, kwargs: dict):
        try:
            return self._client.chat.completions.create(temperature=TEMPERATURE, **kwargs)
        except openai.BadRequestError as e:
            # some models (e.g. reasoning ones) only accept the default temperature
            if "temperature" not in str(e).lower():
                raise
            return self._client.chat.completions.create(**kwargs)

    def complete(self, system_prompt: str, text: str) -> LLMResult:
        if not self.model:
            raise LLMError("No model selected")
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        )
        use_ttl = self.idle_ttl_seconds > 0 and self.base_url is not None and self.server_kind == "lmstudio"
        extra = {"ttl": self.idle_ttl_seconds} if use_ttl else {}  # seconds; LM Studio applies it to JIT loads
        note = ""
        try:
            candidates = self._efforts()
            if candidates != [None]:
                candidates.append(None)
            for i, effort in enumerate(candidates):
                k = dict(kwargs)
                if effort is not None:
                    k["reasoning_effort"] = effort
                if extra:
                    k["extra_body"] = extra
                try:
                    r = self._create(k)
                    break
                except openai.BadRequestError as e:
                    if effort is None or "reasoning" not in str(e).lower() and "effort" not in str(e).lower():
                        raise
                    if candidates[i + 1] is None:
                        note = "server ignored reasoning setting"
        except Exception as e:
            raise _wrap(e, self.base_url) from e
        if self.idle_ttl_seconds > 0 and self.base_url is not None and self.server_kind == "ollama":
            self._set_ollama_keep_alive()
        return self._parse(r, note)

    @staticmethod
    def _parse(r, note: str = "") -> LLMResult:
        msg = r.choices[0].message if r.choices else None
        content = (getattr(msg, "content", None) or "") if msg else ""
        # LM Studio / vLLM / llama-server: reasoning_content; Ollama /v1 and others: reasoning
        reasoning = ""
        for name in ("reasoning_content", "reasoning"):
            v = getattr(msg, name, None) if msg else None
            if isinstance(v, str) and v.strip():
                reasoning = v.strip()
                break
        answer, inline = split_think(content)
        if inline:
            reasoning = f"{reasoning}\n\n{inline}".strip() if reasoning else inline
        tokens = None
        details = getattr(getattr(r, "usage", None), "completion_tokens_details", None)
        if details is not None and getattr(details, "reasoning_tokens", None) is not None:
            tokens = int(details.reasoning_tokens)
        reasoned = True if reasoning or (tokens or 0) > 0 else (False if tokens == 0 else None)
        return LLMResult(answer, reasoning, tokens, reasoned, note)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as e:
            raise _wrap(e, self.base_url) from e

    def unload(self) -> None:
        """Ask LM Studio / Ollama to unload the model now. Other servers: no-op. Raises LLMError on failure."""
        if not self.base_url or not self.model:
            return
        kind = self.server_kind
        root = _root(self.base_url)
        if kind == "lmstudio":
            status, text = _http("POST", root + "/api/v1/models/unload", self.api_key,
                                 {"instance_id": self.model}, UNLOAD_TIMEOUT_S)
        elif kind == "ollama":
            status, text = _http("POST", root + "/api/generate", self.api_key,
                                 {"model": self.model, "keep_alive": 0}, UNLOAD_TIMEOUT_S)
        else:
            return
        if status == 0:
            raise LLMError(f"Could not reach {root} to unload the model")
        if status >= 400:
            raise LLMError(f"Unload failed ({kind}, HTTP {status})")
