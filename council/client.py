"""stdlib-only chat completions client for any openai-compatible api."""

import json
import os
import time
import urllib.error
import urllib.request


class ApiError(Exception):
    pass


RETRYABLE = (429, 500, 502, 503, 504)


class ChatClient:
    def __init__(self, base_url=None, api_key=None, model="gpt-4o-mini",
                 timeout=60, max_retries=3):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                         "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages, model=None, temperature=None, max_tokens=None):
        """send messages, return (text, usage dict). retries 429/5xx."""
        text, usage, _ = self.chat_tools(messages, None, model=model,
                                         temperature=temperature,
                                         max_tokens=max_tokens)
        return text, usage

    def chat_tools(self, messages, tools, model=None, temperature=None,
                   max_tokens=None):
        """like chat, but the model may request tool calls.

        tools is a list of openai-style tool specs (Tool.schema()).
        returns (text, usage, tool_calls) where each call is a dict with
        id, name, arguments, and arguments_raw."""
        body = {"model": model or self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
                return self._parse_tools(payload)
            except urllib.error.HTTPError as e:
                last_err = ApiError("http %d: %s" % (e.code, e.read()[:200]))
                if e.code not in RETRYABLE or attempt == self.max_retries:
                    raise last_err
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = ApiError("connection failed: %s" % e)
                if attempt == self.max_retries:
                    raise last_err
            time.sleep(0.5 * (2 ** attempt))
        raise last_err

    @staticmethod
    def _parse(payload):
        text, usage, _ = ChatClient._parse_tools(payload)
        return text, usage

    @staticmethod
    def _parse_tools(payload):
        try:
            msg = payload["choices"][0]["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ApiError("unexpected response shape")
        usage = payload.get("usage") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                arguments = (json.loads(raw) if isinstance(raw, str)
                             else raw)
            except (ValueError, TypeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append({
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": arguments,
                "arguments_raw": (raw if isinstance(raw, str)
                                  else json.dumps(raw)),
            })
        return text, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }, calls
