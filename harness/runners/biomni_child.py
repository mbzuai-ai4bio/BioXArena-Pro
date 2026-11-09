"""Child process of biomni_runner.py: runs ONE Biomni A1 task (training/run_biomni_agent.py::run_single_task, single attempt)
inside the Landlock sandbox. Usage: biomni_child.py <config.json>; the API key comes from env BX_LLM_API_KEY.
Harness changes vs the standalone training runner: no core-package pip guard (env is read-only), no RLIMIT_AS 60 GB cap (it
makes CUDA initialisation fail; the job's 96 GB cgroup bounds memory for every scaffold alike), all Biomni LLM calls (agent,
tool retriever, database tools via default_config) use the requested model through model_tap (source "Anthropic" =
ChatAnthropic, source "OpenAI" = ChatOpenAI with explicit base_url / key)."""
import json, os, resource, sys, threading, types
from pathlib import Path

# Biomni runs agent code in a worker thread (utils.run_with_timeout). Slurm jobs often have `ulimit -s unlimited`, which leaves
# new threads with glibc's small default stack, and BLAS/torch code then segfaults there. 256 MB thread stacks fix it.
threading.stack_size(256 * 1024 * 1024)

cfg = json.loads(Path(sys.argv[1]).read_text())
CODE = Path(cfg["code_root"])
sys.path[:0] = [str(CODE / "agents" / "Biomni"), str(CODE / "training")]
import importlib.util  # noqa: E402  benchmark_layout.py is readable as a single file only (the code root is not in the sandbox)
_spec = importlib.util.spec_from_file_location("benchmark_layout", CODE / "benchmark_layout.py")
sys.modules["benchmark_layout"] = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sys.modules["benchmark_layout"])
sys.modules["_core_pkg_guard"] = types.SimpleNamespace(restore_core_packages=lambda *a, **k: None)
_setrlimit = resource.setrlimit
resource.setrlimit = lambda which, lim: None if which == resource.RLIMIT_AS else _setrlimit(which, lim)
key = os.environ.pop("BX_LLM_API_KEY")
os.environ.update(BIOMNI_LLM=cfg["model"], BIOMNI_SOURCE=cfg["source"], BIOMNI_CUSTOM_BASE_URL=cfg["base_url"],
                  BIOMNI_CUSTOM_API_KEY=key, BIOMNI_TIMEOUT_SECONDS=str(cfg["exec_timeout_sec"]), BX_BIOMNI_DATA_LAKE=cfg["data_lake"])
if cfg["source"] == "Anthropic":
    os.environ.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL=cfg["base_url"])
else:
    os.environ.update(OPENAI_API_KEY=key, OPENAI_BASE_URL=cfg["base_url"])

import run_biomni_agent as B  # noqa: E402
from biomni.config import default_config  # noqa: E402

default_config.llm = cfg["model"]; default_config.source = cfg["source"]; default_config.base_url = cfg["base_url"]
default_config.api_key = key; default_config.timeout_seconds = cfg["exec_timeout_sec"]
ts = B.TaskSpec(domain=cfg["domain"], task=cfg["task"], task_dir=Path(cfg["public"]),
                description_path=Path(cfg["public"]) / "description.md", output_dir=Path(cfg["output"]))
res = B.run_single_task(task_spec=ts, prompt_template=cfg["prompt_template"], llm_model=cfg["model"], source=cfg["source"],
                        base_url=cfg["base_url"], api_key=key, temperature=0.7, timeout_seconds=cfg["exec_timeout_sec"],
                        print_lock=threading.Lock())
Path(cfg["result_path"]).write_text(json.dumps({"status": res.status, "error": res.error, "duration_sec": res.duration_sec}))
