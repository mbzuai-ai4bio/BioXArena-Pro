"""Minimal scaffold for the unified harness: one LLM, no tools. The model returns one complete solution.py, the harness runs it
and, on failure, sends the error back for a repaired version (training/run_general_llm_agents.py, max 3 attempts in the
7200 s budget). The generated solution.py runs inside the Landlock sandbox (sandbox.py); the core-package pip guard is
disabled (the env is read-only for agents); the API is reached through model_tap.
Backends by route: "openai" = OpenAI Chat Completions (streamed, reasoning enabled where the endpoint supports it);
"anthropic" = Anthropic Messages API with adaptive thinking (an OpenAI-compatible proxy route may silently drop Claude
thinking, which would make Minimal + Claude a non-reasoning baseline); "gemini" = Gemini native API (google-genai SDK, streamed,
the model's default thinking; thought parts are not resent). All three return the same response shape to the generation loop."""
import os, subprocess, sys, types
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H.parent / "training"))
sys.modules["_core_pkg_guard"] = types.SimpleNamespace(restore_core_packages=lambda *a, **k: None)
import run_general_llm_agents as G  # noqa: E402

SANDBOX = [sys.executable, "-I", str(H / "sandbox.py"), sys.executable]
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SEC = 1800


def _run_sandboxed(solution_path, cwd, timeout_sec):
    return subprocess.run(SANDBOX + [str(solution_path)], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout_sec, check=False)


G.run_solution_file = _run_sandboxed


class AnthropicClient:
    """Drop-in for G.LLMClient.chat(): Anthropic Messages API, adaptive thinking, streamed; thinking blocks are not resent."""
    def __init__(self, api_key, base_url, model, timeout_sec):
        import anthropic
        self.model = model; self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=timeout_sec, max_retries=2)

    def chat(self, messages):
        msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])} for m in messages]
        with self.client.messages.stream(model=self.model, max_tokens=64000, thinking={"type": "adaptive"}, messages=msgs) as st:
            r = st.get_final_message()
        text = "".join(b.text for b in r.content if b.type == "text"); u = r.usage
        inp = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
        return {"id": r.id, "model": r.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": r.stop_reason}],
                "usage": {"prompt_tokens": inp, "completion_tokens": u.output_tokens or 0}}


class GeminiClient:
    """Drop-in for G.LLMClient.chat(): Gemini generateContent (streamed); system messages -> system_instruction."""
    def __init__(self, api_key, base_url, model, timeout_sec):
        from google import genai
        from google.genai import types as gt
        self.model, self.gt = model, gt
        self.client = genai.Client(api_key=api_key, http_options=gt.HttpOptions(base_url=base_url, timeout=timeout_sec * 1000))

    def chat(self, messages):
        gt = self.gt
        txt = lambda c: c if isinstance(c, str) else str(c)
        system = "\n\n".join(txt(m["content"]) for m in messages if m["role"] == "system")
        contents = [gt.Content(role="model" if m["role"] == "assistant" else "user", parts=[gt.Part(text=txt(m["content"]))])
                    for m in messages if m["role"] != "system"]
        cfg = gt.GenerateContentConfig(system_instruction=system or None)
        text, usage, model_version, finish, rid = [], None, None, None, None
        for ch in self.client.models.generate_content_stream(model=self.model, contents=contents, config=cfg):
            rid = rid or getattr(ch, "response_id", None); model_version = getattr(ch, "model_version", None) or model_version
            usage = ch.usage_metadata or usage
            for c in ch.candidates or []:
                finish = c.finish_reason or finish
                for p in (c.content.parts if c.content and c.content.parts else []):
                    if p.text and not p.thought: text.append(p.text)
        u = usage
        th = (u.thoughts_token_count or 0) if u else 0
        return {"id": rid or "streamed-response", "model": model_version or self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                             "finish_reason": str(finish) if finish else None}],
                "usage": {"prompt_tokens": (u.prompt_token_count or 0) if u else 0,
                          "completion_tokens": ((u.candidates_token_count or 0) + th) if u else 0, "reasoning_tokens": th}}


def run_single_task(*, domain, task, public, out, prompt_template, model, route, base_url, api_key, wall_clock_sec, print_lock):
    os.environ["BIOXARENA_STREAM_API"] = "1"
    ts = G.TaskSpec(domain=domain, task=task, grade_path=H.parent / "tasks" / domain / task / "grade.py", task_dir=public,
                    description_path=public / "description.md", output_dir=out)
    if route == "anthropic":
        client = AnthropicClient(api_key, base_url, model, REQUEST_TIMEOUT_SEC)
    elif route == "gemini":
        client = GeminiClient(api_key, base_url, model, REQUEST_TIMEOUT_SEC)
    else:
        client = G.LLMClient(api_key=api_key, api_url=base_url.rstrip("/") + "/v1", model=model, timeout_sec=REQUEST_TIMEOUT_SEC,
                             enable_reasoning=True, temperature=None, max_completion_tokens=None, seed=None)
    return G.run_single_task(task_spec=ts, prompt_template=prompt_template, client=client, max_attempts=MAX_ATTEMPTS,
                             run_timeout_sec=wall_clock_sec, dry_run=False, print_lock=print_lock, task_timeout_sec=wall_clock_sec)
