"""Single place for every path, API route, endpoint and key of the BioXArena-Pro harness. Everything is set by environment
variables (see config.env.example at the repository root); nothing here is specific to the machine the benchmark was run on.
Keys are read from the environment only and are never written to disk by the harness."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BX_ROOT") or Path(__file__).resolve().parents[1])            # repository root
DATA_PUBLIC = Path(os.environ.get("BX_DATA_PUBLIC") or ROOT / "data" / "BioXArena-Pro-Data-Public")
DATA_PRIVATE = Path(os.environ.get("BX_DATA_PRIVATE") or ROOT / "data" / "BioXArena-Pro-Data-Private")  # scoring only
OUTPUT_ROOT = Path(os.environ.get("BX_OUTPUT_ROOT") or ROOT / "outputs")                 # <scaffold>__<model>/<domain>/<task>
ARCHIVE_ROOT = Path(os.environ.get("BX_ARCHIVE_ROOT") or ROOT / "archive")               # failed / refused attempts (never deleted)
WALL_SEC = int(os.environ.get("BX_WALL_SEC", "7200"))                                    # 2 h per task; change only for smoke tests

# API routes. The harness places a local pass-through (model_tap.py) between the agent and the upstream so the served model
# and the exact token usage of every response are recorded. "host" may carry a path prefix (e.g. https://openrouter.ai/api);
# the runners append the API path they need (/v1, /v1/messages, /v1beta, ...).
ROUTES = {
    "openai": {  # any OpenAI-compatible endpoint: Chat Completions under /v1/chat/completions, Responses under /v1/responses
        "host": os.environ.get("BX_OPENAI_BASE", "https://openrouter.ai/api"),
        "keys": ("OPENROUTER_API_KEY", "OPENAI_API_KEY")},
    "anthropic": {  # Anthropic Messages API (Claude Code, and the Claude backends of Minimal / OpenCode / Biomni)
        "host": os.environ.get("BX_ANTHROPIC_BASE", "https://api.anthropic.com"),
        "keys": ("ANTHROPIC_API_KEY",)},
    "gemini": {  # Google Gemini API (Gemini CLI, and the Gemini backends of Minimal / OpenCode / Biomni / MLEvolve)
        "host": os.environ.get("BX_GOOGLE_API_HOST", "https://generativelanguage.googleapis.com"),
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY")},
}
OPENAI_COMPAT_PATH = "/v1beta/openai"        # Gemini's OpenAI-compatible Chat Completions route (Biomni on the gemini route)
CODEX_WIRE_API = os.environ.get("BX_CODEX_WIRE_API", "responses")  # Codex CLI wire protocol on the openai route ("responses" | "chat")

SCAFFOLDS = {"codex": "Codex CLI", "claude": "Claude Code", "geminicli": "Gemini CLI", "minimal": "Minimal",
             "opencode": "OpenCode", "biomni": "Biomni", "mlevolve": "MLEvolve"}

# External tools (read-only inside the sandbox). Set the ones you use.
NODE_HOME = os.environ.get("BX_NODE_HOME", "")              # e.g. ~/.nvm/versions/node/v22.22.0 (Codex, Gemini CLI, OpenCode)
CODEX_BIN = os.environ.get("BX_CODEX_BIN", "")               # codex binary, or <npm prefix>/lib/node_modules/@openai/codex/bin/codex.js
CLAUDE_BIN = os.environ.get("BX_CLAUDE_BIN", "")             # Claude Code binary (e.g. ~/.local/share/claude/versions/<version>)
GEMINI_CLI_JS = os.environ.get("BX_GEMINI_CLI_JS", "")       # <npm prefix>/lib/node_modules/@google/gemini-cli/bundle/gemini.js
OPENCODE_BIN = os.environ.get("BX_OPENCODE_BIN", "")         # <npm prefix>/lib/node_modules/opencode-ai/bin/opencode(.exe)
BIOMNI_DATA_ROOT = os.environ.get("BIOMNI_DATA_ROOT", "")    # dir that holds biomni_data/{data_lake,benchmark} (~15 GB)
SANDBOX_EXTRA_RO = [p for p in os.environ.get("BX_SANDBOX_EXTRA_RO", "").split(":") if p]  # e.g. cluster module / CUDA trees


def default_route(cli: str, model: str) -> str:
    """Route used when a run item has no explicit "route": the vendor CLIs use their own API; the model-agnostic scaffolds
    use the Anthropic / Gemini API for bare claude-* / gemini-* ids and an OpenAI-compatible endpoint otherwise (a
    vendor-prefixed id such as deepseek/deepseek-v4.1-flash-1017 is an OpenRouter-style id and always goes to "openai")."""
    if cli == "claude": return "anthropic"
    if cli == "codex": return "openai"
    if cli == "geminicli": return "gemini"
    if "/" in model: return "openai"
    if model.startswith("claude"): return "anthropic"
    if model.startswith("gemini"): return "gemini"
    return "openai"


def upstream(route: str) -> str:
    return ROUTES[route]["host"].rstrip("/")


def api_key(route: str) -> str:
    for k in ROUTES[route]["keys"]:
        if os.environ.get(k):
            return os.environ[k]
    raise SystemExit(f"route {route}: none of {ROUTES[route]['keys']} is set in the environment")


def model_dir(scaffold: str, model: str) -> str:
    return f"{scaffold}__{model.split('/')[-1]}"


def same_model(served: str, requested: str) -> bool:
    """Server-reported model == requested model, tolerating a vendor prefix (models/, openai/) and a dated / numbered suffix
    (e.g. gemini-3.8-flash-001, claude-opus-5-5-20261001). Any other model counts as a different model."""
    s = (served or "").split("/")[-1]; r = requested.split("/")[-1]
    return s == r or s.startswith(r + "-")


if __name__ == "__main__":  # bx_config.py upstream|key <route>  (used by worker.sh)
    import sys
    print(upstream(sys.argv[2]) if sys.argv[1] == "upstream" else api_key(sys.argv[2]))
