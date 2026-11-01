import json
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError 
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
import time
import re
import pickle
import subprocess
import sys

# ---------- Data structures ----------
@dataclass
class Subtask:
    id: str
    description: str
    role: str
    depends_on: list[str] = field(default_factory=list)

@dataclass
class Result:
    subtask_id: str
    output: str
    status: str = "done"
# ---------- LLM wrapper ----------
# MODEL = os.environ.get("MODEL")
# client = Groq(api_key=os.environ["GROQ_API_KEY"])
client = OpenAI(
  base_url = os.environ.get("base_url"),
  api_key = os.environ.get("api_key"),
  timeout=300,
)

model=os.environ.get("MODEL")
# completion = client.chat.completions.create(
#   messages=[{"role":"user","content":"Which number is larger, 9.11 or 9.8?"}],
# #   temperature=1,
# #   top_p=1,
# #   max_tokens=4096,
# #   stream=False
# )

# reasoning = getattr(model.choices[0].message, "reasoning_content", None)
# if reasoning:
#   print(reasoning)
# print(model.choices[0].message.content)

def call_llm(system: str, user: str, retries: int = 6, max_tokens: int = 16000) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                # extra_body={"reasoning_effort": "low"},   # the fix
            )
            choice = response.choices[0]
            text = choice.message.content

            if not text or not text.strip():
                print(f"  empty reply (finish_reason={choice.finish_reason}), retrying...")
                if choice.finish_reason == "length" and attempt == 0:
                    max_tokens *= 2        # one bigger retry only
                elif choice.finish_reason == "length":
                    raise RuntimeError("Model used all tokens on reasoning; check the model or prompt")
                continue

            if choice.finish_reason == "length":
                print("  reply was cut off, retrying with a bigger budget...")
                max_tokens *= 2
                continue
            return text
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            wait = min(2 ** attempt, 30)
            print(f"  {type(e).__name__}, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("No usable reply after retries")

# ---------- Planner ----------
PLANNER_SYSTEM = """You are an orchestrator that decomposes coding tasks.
Split the task into the smallest useful subtasks. Return ONLY valid JSON:
{
  "package_name": "one_snake_case_name",
  "subtasks": [
    {"id": "T1", "description": "...", "role": "You are an expert in ...", "depends_on": []},
    {"id": "T3", "description": "...", "role": "...", "depends_on": ["T1", "T2"]}
  ]
}
Rules: ids are unique; depends_on lists only ids that exist; no cycles;
independent subtasks must have empty depends_on so they can run in parallel.
ALL code must live in ONE package named package_name. Tests go in tests/."""

def plan(task: str):
    raw = call_llm(PLANNER_SYSTEM, task)
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(raw)
    subtasks = [Subtask(**s) for s in data["subtasks"]]
    return data["package_name"], subtasks

# ---------- DAG validation + ordering ----------
def validate_and_layer(subtasks: list[Subtask]) -> list[list[Subtask]]:
    """Check the plan and group subtasks into 'waves'.
    Every subtask in a wave can run in parallel, because all of
    its dependencies are in earlier waves."""
    by_id = {s.id: s for s in subtasks}

    # Check 1: ids are unique
    if len(by_id) != len(subtasks):
        raise ValueError("Duplicate subtask ids in the plan")

    # Check 2: every dependency points to a real subtask
    for s in subtasks:
        for dep in s.depends_on:
            if dep not in by_id:
                raise ValueError(f"{s.id} depends on unknown subtask '{dep}'")

    # Check 3 + ordering: peel the graph layer by layer (Kahn's algorithm)
    done = set()
    remaining = dict(by_id)
    waves = []

    while remaining:
        ready = [s for s in remaining.values()
                 if all(dep in done for dep in s.depends_on)]
        if not ready:
            raise ValueError(f"Cycle detected among: {list(remaining)}")
        waves.append(ready)
        for s in ready:
            done.add(s.id)
            del remaining[s.id]

    return waves
# ---------- Sub-agent factory ----------
class SubAgent:
    def __init__(self, subtask: Subtask, package_name: str):
        self.subtask = subtask
        self.system_prompt = (
            f"{subtask.role}\n"
            "Write complete, working code for your assigned part only. "
            "Return only code (with brief comments), no long explanations.\n"
            f'Use ONLY the package "{package_name}". '
            f"Put code in {package_name}/ and tests in tests/. "
            "Never create any other package or folder. "
            f"Inside the package, import with relative imports "
            f"(from .models import X), never from {package_name} itself."
        )

    def run(self, dep_results: dict[str, Result]) -> Result:
        # Build the prompt: my task + the outputs of the subtasks I depend on
        context = ""
        for dep_id in self.subtask.depends_on:
            context += f"\n--- Output of {dep_id} ---\n{dep_results[dep_id].output}\n"

        user_prompt = f"Your task: {self.subtask.description}\n"
        if context:
            user_prompt += f"\nWork already done by other agents (use it, stay consistent with it):\n{context}"

        output = call_llm(self.system_prompt, user_prompt)
        return Result(subtask_id=self.subtask.id, output=output)


def create_agent(subtask: Subtask, package_name: str) -> SubAgent:
    return SubAgent(subtask, package_name)
# ---------- Executor ----------
def execute(waves, package_name: str) -> dict[str, Result]:
    results: dict[str, Result] = {}

    for i, wave in enumerate(waves, 1):
        print(f"Running wave {i}: {[s.id for s in wave]}")
        agents = [create_agent(s, package_name) for s in wave]

        def run_one(agent: SubAgent) -> Result:
            try:
                return agent.run(results)
            except Exception as e:
                return Result(agent.subtask.id, f"ERROR: {e}", status="failed")

        with ThreadPoolExecutor(max_workers=2) as pool:
            wave_results = list(pool.map(run_one, agents))

        for r in wave_results:
            results[r.subtask_id] = r
            print(f"  {r.subtask_id}: {r.status}")

    return results
# ---------- Integrator ----------
INTEGRATOR_SYSTEM = """You are a senior engineer integrating code written by several agents.
Merge everything into one coherent, working solution:
- remove duplicated code and duplicated definitions
- keep names, imports and function signatures consistent
- put tests in a tests/ folder and make them import from the app files
Output format (strict): for every file, write its path in bold on its own line,
then the full code in a fenced block, exactly like this:

**path/to/module.py**
```python
<code>
```

**tests/test_module.py**
```python
<code>
```

Return only files in this format, with no other text."""

def merge_step(task: str, merged: str, new_pieces: str, package_name: str) -> str:
    user = f"Original task: {task}\n\n"
    user += (f'Output ONE package only: "{package_name}". '
             "If two files do the same job, merge them into one. "
             f"Use relative imports inside the package.\n\n")
    if merged:
        user += f"Current merged solution:\n{merged}\n\n"
    user += f"New pieces to merge into it:\n{new_pieces}"
    return call_llm(INTEGRATOR_SYSTEM, user, max_tokens=24000)

def integrate(task, subtasks, results, package_name, max_chars: int = 5000) -> str:
    merged, batch = "", ""
    for s in subtasks:
        r = results[s.id]
        if r.status != "done":
            print(f"  skipping {s.id} (failed)")
            continue
        piece = f"\n### {s.id}: {s.description}\n{r.output}\n"
        if batch and len(batch) + len(piece) > max_chars:
            merged = merge_step(task, merged, batch, package_name)
            batch = ""
        batch += piece
    if batch:
        merged = merge_step(task, merged, batch, package_name)
    return merged
def save_files(markdown: str, out_dir: str = "output") -> list[str]:
    # matches:  **filename**  followed by a fenced code block
    pattern = re.compile(
       r"(?:^|\n)[#*\s`]*(?:File:\s*)?([\w./\\-]+\.(?:py|toml|md|txt))[`*\s:]*\n\s*```\w*\n(.*?)```",
       re.DOTALL,)
    saved = []
    for filename, code in pattern.findall(markdown):
        path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        saved.append(path)
    return saved

# ---------- Test-and-fix loop ----------
def run_pytest(project_dir: str = "output") -> tuple[bool, str]:
    if os.path.exists(os.path.join(project_dir, "pyproject.toml")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."],
                       cwd=project_dir, capture_output=True, text=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"],
        cwd=project_dir, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    return proc.returncode == 0, out[-8000:]      # keep only the end, where the errors are

def read_project(project_dir: str = "output") -> str:
    parts = []
    for root, _, files in os.walk(project_dir):
        if any(skip in root for skip in (".pytest_cache", "__pycache__", ".egg-info")):
            continue
        for name in files:
            if name.endswith((".py", ".toml")):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, project_dir).replace("\\", "/")
                with open(path, encoding="utf-8") as f:
                    parts.append(f"**{rel}**\n```\n{f.read()}\n```")
    return "\n\n".join(parts)

FIXER_SYSTEM = """You are a senior engineer fixing a Python project whose tests fail.
You get the pytest output, a diagnostic probe output (the real behavior of the code),
and all project files.
- Find the ROOT CAUSE. Many failures often come from one bug.
- Fix the source code. Only edit a test if the test itself is wrong.
- Never delete or weaken tests just to make them pass.
- Read helper functions literally.

First write a section starting with "DIAGNOSIS:" (max 6 lines): for each failing test,
state what the code returned versus what was expected, and the exact line responsible.

Then return ONLY the files you changed, each one complete, in this exact format:

**path/to/file.py**
```````python
<full file content>
``````"""

PROBE_SYSTEM = """You are debugging a Python project with failing tests.
Write ONE short Python script that prints the actual behavior of the suspicious
functions (call them with the failing inputs, print results and intermediate values,
for example the result of each helper function on a single character).
The script runs with the project root as the current directory, so import the package
directly (for example: from expense_tracker.models import add_expense).
Never use __file__ or sys.path.
Return only a python code block."""

def probe(output: str, project_dir: str) -> str:
    reply = call_llm(PROBE_SYSTEM,
                     f"Failing tests:\n{output}\n\nProject files:\n{read_project(project_dir)}")
    match = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    if not match:
        return "(no probe)"
    try:
        proc = subprocess.run([sys.executable, "-c", match.group(1)],
                              cwd=project_dir, capture_output=True, text=True, timeout=20)
        return (proc.stdout + proc.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        return "(probe timed out)"

def fix_loop(task: str, project_dir: str = "output", max_rounds: int = 4) -> bool:
    history = []
    for round_no in range(1, max_rounds + 1):
        passed, output = run_pytest(project_dir)
        summary = [l for l in output.splitlines() if " passed" in l or " failed" in l]
        print("  status:", summary[-1] if summary else "unknown")
        if passed:
            print(f"All tests pass (after {round_no - 1} fix rounds)")
            return True

        print(f"Round {round_no}: tests failing, asking the fixer...")
        evidence = probe(output, project_dir)
        print("  probe output:", evidence[:200].replace("\n", " | "))

        user = (f"Original task: {task}\n\n"
                f"Previous attempts (do not repeat them):\n" + "\n".join(history) + "\n\n"
                f"Pytest output:\n{output}\n\n"
                f"Diagnostic probe output (real behavior):\n{evidence}\n\n"
                f"Project files:\n{read_project(project_dir)}")
        answer = call_llm(FIXER_SYSTEM, user, max_tokens=24000)
        changed = save_files(answer, project_dir)
        if not changed:
            with open("fixer_last.md", "w", encoding="utf-8") as f:
                f.write(answer)
            print("  no files parsed, reply starts with:", answer[:300].replace("\n", " | "))
            retry = call_llm(
                FIXER_SYSTEM,
                user + "\n\nYour previous reply contained no files in the required format. "
                       "Reply again. Each changed file must be a bold path on its own line, "
                       "followed by the FULL file in a fenced code block.",
                max_tokens=24000,
            )
            changed = save_files(retry, project_dir)
        print("  fixer changed:", changed)
        history.append(f"Round {round_no}: changed {changed}")
        if not changed:
            print("  fixer returned no files twice, stopping")
            return False

    passed, _ = run_pytest(project_dir)
    print("Final result:", "PASS" if passed else "still failing")
    return passed
# ---------- Streamlit UI ----------
UI_SYSTEM = """You are a senior engineer who builds Streamlit apps.
You get a finished Python project. Write ONE file, app.py, that gives it a Streamlit UI.
Rules:
- Import from the existing package (the folder that contains __init__.py). Do not change the package.
- Use only the real function and class names you see in the code. Never invent any.
- One section per main feature (use st.tabs if there are several).
- Show results clearly (st.metric, st.progress, st.success, st.error, st.code).
- Keep it under 150 lines.
Return ONLY this format:

**app.py**
`````python
<code>
````"""

def build_ui(task: str, project_dir: str = "output"):
    user = (f"Original task: {task}\n\n"
            f"Project files:\n{read_project(project_dir)}")
    answer = call_llm(UI_SYSTEM, user, max_tokens=16000)

    saved = save_files(answer, project_dir)
    if not saved:
        m = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        code = m.group(1) if m else answer
        path = os.path.join(project_dir, "app.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        saved = [path]

    print("UI saved:", saved)
    with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
        f.write("streamlit\n")
# ---------- Test ----------
# FIX_ONLY = True    # skip planning and integration, just fix the existing output/ folder
FIX_ONLY = False     # run the whole pipeline, not just the fix loop
USE_CACHE = False    # don't reuse the old sub-agent results
# import shutil
# shutil.rmtree("output", ignore_errors=True)
if __name__ == "__main__":
    # TASK = ("Build a Python password toolkit that scores password strength with clear reasons, "
    #     "generates secure passwords with configurable length and character sets, "
    #     "and generates passphrases from a built-in word list, with a CLI and unit tests. "
    #     "Use only the standard library. Functions must return plain dicts so a UI can display them.")
    TASK = ("Build a Python text analyzer that counts words, finds the most frequent words,"
    " computes reading time and sentence statistics from a text file, with a CLI and unit tests")
    if not FIX_ONLY:
        if USE_CACHE and os.path.exists("results.pkl"):
            with open("results.pkl", "rb") as f:
                package_name, subtasks, results = pickle.load(f)
        else:
            package_name, subtasks = plan(TASK)
            waves = validate_and_layer(subtasks)
            results = execute(waves, package_name)
            with open("results.pkl", "wb") as f:
                pickle.dump((package_name, subtasks, results), f)

        print("\nIntegrating...")
        final = integrate(TASK, subtasks, results, package_name)
        with open("final_solution.md", "w", encoding="utf-8") as f:
            f.write(final)
        print("Saved:", save_files(final))

    ok = fix_loop(TASK)
    if not ok:
        print("Some tests still fail, building the UI anyway")
    build_ui(TASK)
    # if fix_loop(TASK):
        # build_ui(TASK)
