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
        body = {"model": model or self.model, "messages": messages}
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
                return self._parse(payload)
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
        try:
            msg = payload["choices"][0]["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ApiError("unexpected response shape")
        usage = payload.get("usage") or {}
        return text, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
