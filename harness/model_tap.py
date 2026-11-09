"""Local pass-through HTTP tap between an agent (Codex CLI, Claude Code, Gemini CLI, OpenCode, Minimal, Biomni, MLEvolve) and the API upstream
(OpenAI-compatible, Anthropic or Gemini; bx_config.ROUTES).
Every request is forwarded unchanged to UPSTREAM (https) and the response streamed back. From each response the tap records
(a) the model the SERVER reports it served and (b) the token usage the server reports, so both are exact for every run,
including runs stopped at the deadline (the agents' own end-of-run summaries are missing then).
Gemini native API (generateContent / streamGenerateContent?alt=sse; google-genai SDK, Gemini CLI, @ai-sdk/google):
modelVersion, usageMetadata{promptTokenCount, cachedContentTokenCount, candidatesTokenCount, thoughtsTokenCount,
toolUsePromptTokenCount}; safety blocks = candidates[].finishReason in GEMINI_BLOCK or promptFeedback.blockReason.
Gemini OpenAI-compatible route (/v1beta/openai, Chat Completions): model, usage{prompt_tokens,
prompt_tokens_details.cached_tokens, completion_tokens, completion_tokens_details.reasoning_tokens}; finish_reason
content_filter = safety block. Token convention (same as the GPT runs): input_tokens includes cached tokens, output_tokens
includes reasoning ("thoughts") tokens, reasoning_output_tokens is the reasoning part.
Runs in the runner process (outside the agent sandbox) and listens on 127.0.0.1 only."""
import http.client, json, threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "upgrade", "proxy-connection", "te", "trailer", "host",
       "accept-encoding"}
KEYS = ["input_tokens", "cached_input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens",
        "reasoning_output_tokens"]


POLICY = ("policy", "safety", "cyber", "bio", "content_filter", "flagged", "refus")
GEMINI_BLOCK = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def parse_event(j, acc):
    """Update acc (model, usage dict, refusal flags) from one decoded JSON event / body."""
    t = j.get("type", "")
    # safety signals: OpenAI refusal content (response.refusal.*, content part type "refusal"), Anthropic stop_reason
    # "refusal", and error events whose code/type/message names a policy / safety category
    if "refusal" in t or (j.get("part") or {}).get("type") == "refusal" or (j.get("item") or {}).get("type") == "refusal":
        acc["refusal"] = True
    if ((j.get("delta") or {}).get("stop_reason") == "refusal") or ((j.get("message") or {}).get("stop_reason") == "refusal") \
            or j.get("stop_reason") == "refusal":  # non-streamed Anthropic message body
        acc["refusal"] = True
    err = j.get("error") or ((j.get("response") or {}).get("error") if isinstance(j.get("response"), dict) else None)
    if isinstance(err, dict):
        code = " ".join(str(err.get(k, "")) for k in ("code", "type", "message")).lower()
        if any(k in code for k in POLICY): acc["refusal"] = True; acc["policy_error"] = code[:200]
    r = j.get("response") if isinstance(j.get("response"), dict) else None
    if r:
        for it in r.get("output") or []:
            for c in (it.get("content") or []) if isinstance(it, dict) else []:
                if isinstance(c, dict) and c.get("type") == "refusal": acc["refusal"] = True
    if "candidates" in j or "usageMetadata" in j or "promptFeedback" in j:  # Gemini native response / SSE chunk
        acc["model"] = j.get("modelVersion") or acc["model"]
        for c in j.get("candidates") or []:
            if isinstance(c, dict) and c.get("finishReason") in GEMINI_BLOCK: acc["refusal"] = True
        if (j.get("promptFeedback") or {}).get("blockReason"): acc["refusal"] = True
        u = j.get("usageMetadata") or {}
        if u:  # cumulative in every SSE chunk: the last one wins
            th = u.get("thoughtsTokenCount") or 0
            acc["u"]["input_tokens"] = (u.get("promptTokenCount") or 0) + (u.get("toolUsePromptTokenCount") or 0)
            acc["u"]["cached_input_tokens"] = u.get("cachedContentTokenCount") or 0
            acc["u"]["output_tokens"] = (u.get("candidatesTokenCount") or 0) + th
            acc["u"]["reasoning_output_tokens"] = th
        return
    obj = j.get("object", "")
    if obj in ("chat.completion", "chat.completion.chunk") or ("choices" in j and "model" in j):  # OpenAI Chat Completions (Minimal, OpenCode, Biomni via OpenAI SDK)
        acc["model"] = j.get("model") or acc["model"]
        for ch in j.get("choices") or []:
            m = ch.get("message") or ch.get("delta") or {}
            if m.get("refusal") or ch.get("finish_reason") == "content_filter": acc["refusal"] = True
        u = j.get("usage") or {}
        if u:
            acc["u"]["input_tokens"] = u.get("prompt_tokens") or 0
            acc["u"]["cached_input_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            acc["u"]["output_tokens"] = u.get("completion_tokens") or 0
            acc["u"]["reasoning_output_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return
    if t in ("response.completed", "response.incomplete", "response.failed") or ("usage" in j and "output" in j and "model" in j):
        r = j.get("response", j); acc["model"] = r.get("model") or acc["model"]; u = r.get("usage") or {}
        if u:
            acc["u"]["input_tokens"] = u.get("input_tokens") or 0
            acc["u"]["cached_input_tokens"] = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            acc["u"]["output_tokens"] = u.get("output_tokens") or 0
            acc["u"]["reasoning_output_tokens"] = (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    elif t == "response.created":
        acc["model"] = (j.get("response") or {}).get("model") or acc["model"]
    elif t == "message_start" or (t == "message" and "usage" in j):
        m = j.get("message", j); acc["model"] = m.get("model") or acc["model"]; u = m.get("usage") or {}
        for k in ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]:
            if u.get(k) is not None: acc["u"][k] = u[k]
    elif t == "message_delta":
        u = j.get("usage") or {}
        for k in ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]:
            if u.get(k) is not None: acc["u"][k] = u[k]


class Tap:
    def __init__(self, upstream):
        u = urlsplit(upstream); self.host, self.base = u.netloc, u.path.rstrip("/")
        self.served = Counter(); self.usage = Counter(); self.usage_by_model = {}; self.responses = 0; self.errors = 0
        self.refusals = 0; self.policy_errors = Counter(); self.http_status = Counter(); self.paths = Counter()
        self.lock = threading.Lock(); tap = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a): pass

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else None
                hdr = {k: v for k, v in self.headers.items() if k.lower() not in HOP}; hdr["Accept-Encoding"] = "identity"
                path = self.path if (not tap.base or self.path.startswith(tap.base)) else tap.base + self.path
                try:
                    c = http.client.HTTPSConnection(tap.host, timeout=900); c.request(self.command, path, body=body, headers=hdr)
                    r = c.getresponse()
                except Exception:
                    with tap.lock: tap.errors += 1
                    self.send_error(502); return
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP: self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                acc = {"model": None, "u": {}, "refusal": False, "policy_error": None}; buf = b""; whole = []
                with tap.lock: tap.http_status[r.status] += 1; tap.paths[f"{self.command} {path.split('?')[0]}"] += 1
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk: break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
                        buf += chunk; whole.append(chunk) if len(whole) < 4096 else None
                        *lines, buf = buf.split(b"\n")
                        for ln in lines:
                            if ln.startswith(b"data:"):
                                try: parse_event(json.loads(ln[5:].strip()), acc)
                                except Exception: pass
                    self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    c.close()
                    if acc["model"] is None and whole:  # non-streaming JSON body (a JSON array for non-SSE Gemini streams)
                        try:
                            body = json.loads(b"".join(whole))
                            for ev in (body if isinstance(body, list) else [body]): parse_event(ev, acc)
                        except Exception: pass
                    if acc["refusal"] or acc["policy_error"]:
                        with tap.lock:
                            tap.refusals += 1
                            if acc["policy_error"]: tap.policy_errors[acc["policy_error"]] += 1
                    if self.command == "POST" and acc["model"]:
                        with tap.lock:
                            tap.responses += 1; tap.served[acc["model"]] += 1
                            bm = tap.usage_by_model.setdefault(acc["model"], Counter())
                            for k, v in acc["u"].items(): tap.usage[k] += int(v or 0); bm[k] += int(v or 0)

            do_POST = do_GET = _go

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H); self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def url(self): return f"http://127.0.0.1:{self.port}{self.base}"

    def report(self):
        with self.lock:
            return {"responses": self.responses, "served_models": dict(self.served), "upstream_errors": self.errors,
                    "refusal_responses": self.refusals, "policy_errors": dict(self.policy_errors),
                    "http_status": {str(k): v for k, v in self.http_status.items()}, "paths": dict(self.paths),
                    "usage_total": {k: self.usage.get(k, 0) for k in KEYS},
                    "usage_by_served_model": {m: dict(c) for m, c in self.usage_by_model.items()}}

    def close(self): self.srv.shutdown()
