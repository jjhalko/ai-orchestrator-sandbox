"""
Rung 2 of the mini orchestrator: the CODE stage.

Flow:
  1. Load the issue and the plan JSON the plan stage saved.
  2. Create a branch: ai/issue-<number>.
  3. Run an AGENT LOOP: Claude gets tools (list_files, read_file,
     write_file, delete_file, finish) and works step by step. Each turn we
     run the tools it asked for and send back the results, until it calls
     `finish` or hits MAX_TURNS.
  4. Commit, push, open a pull request, and comment on the issue.

This is the key new idea vs. rung 1: instead of one request/response,
the model acts in a loop, and *your code* executes its actions.

Run locally (writes files, but no git push / PR):
    python orchestrator/code.py --issue-file orchestrator/sample_issue.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anthropic
import requests

MODEL = os.environ.get("ORCH_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 15                 # hard stop for the loop (cost + runaway guard)
MAX_OUTPUT_TOKENS = 8000
MAX_READ_CHARS = 20_000
REPO_ROOT = Path.cwd().resolve()
OUTPUT_DIR = Path("orchestrator_runs")
# Paths the agent may never touch
PROTECTED = (".git", ".github", "orchestrator", "orchestrator_runs")


# ---------------------------------------------------------------------------
# Tools. "strict": True guarantees inputs match the schema.
# ---------------------------------------------------------------------------
def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": required,
        },
    }


TOOLS = [
    _tool("list_files", "List all files in the repository.", {}, []),
    _tool("read_file", "Read a text file from the repository.",
          {"path": {"type": "string"}}, ["path"]),
    _tool("write_file",
          "Create or fully overwrite a file with the given content.",
          {"path": {"type": "string"}, "content": {"type": "string"}},
          ["path", "content"]),
    _tool("delete_file", "Delete a file from the repository.",
          {"path": {"type": "string"}}, ["path"]),
    _tool(
        "finish",
        "Call this once the work is complete (or you cannot proceed).",
        {
            "summary": {"type": "string",
                        "description": "What you changed, for the PR description."},
            "how_to_test": {"type": "array", "items": {"type": "string"},
                            "description": "Steps a human can follow to verify."},
            "notes": {"type": "array", "items": {"type": "string"},
                      "description": "Deviations from the plan, caveats, follow-ups."},
            "confidence": {"type": "integer",
                           "description": "0-100: confidence the change meets every acceptance criterion."},
            "confidence_reason": {"type": "string"},
        },
        ["summary", "how_to_test", "notes", "confidence", "confidence_reason"],
    ),
]

SYSTEM_PROMPT = """You are the CODING stage of a software delivery pipeline.
You are given a GitHub issue and an approved implementation plan.
Implement the plan using your tools, then call `finish`.

Rules:
- Follow the plan. If you must deviate, explain why in finish.notes.
- Make the smallest change that satisfies every acceptance criterion.
- Read a file before overwriting it, unless you are creating it.
- write_file replaces the whole file: always send the complete content.
- You cannot modify .git, .github, or orchestrator/ — those are off limits.
- Do not add dependencies or files the plan doesn't call for.
- Calibrate confidence honestly; you cannot run the code, so say what you
  could not verify.
Always act through tool calls; finish by calling `finish`."""


# ---------------------------------------------------------------------------
# Tool execution — this is the part *you* control. The model only asks;
# your code decides what actually happens.
# ---------------------------------------------------------------------------
def safe_path(rel: str) -> Path:
    p = (REPO_ROOT / rel).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"Path escapes the repository: {rel}")
    first = p.relative_to(REPO_ROOT).parts[:1]
    if first and first[0] in PROTECTED:
        raise ValueError(f"Path is protected: {rel}")
    return p


def list_files() -> str:
    files = []
    for p in sorted(REPO_ROOT.rglob("*")):
        rel = p.relative_to(REPO_ROOT)
        if (p.is_file() and rel.parts[0] not in (".git", "orchestrator_runs")
                and "__pycache__" not in rel.parts):
            files.append(str(rel))
    return "\n".join(files) or "(repository is empty)"


def run_tool(name: str, args: dict, changed: set[str]) -> str:
    try:
        if name == "list_files":
            return list_files()
        if name == "read_file":
            p = (REPO_ROOT / args["path"]).resolve()
            if REPO_ROOT not in p.parents:
                return "Error: path escapes the repository."
            if not p.is_file():
                return f"Error: {args['path']} does not exist."
            text = p.read_text(errors="replace")
            return text[:MAX_READ_CHARS] + (
                "\n...(truncated)" if len(text) > MAX_READ_CHARS else "")
        if name == "write_file":
            p = safe_path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            changed.add(args["path"])
            return f"Wrote {args['path']} ({len(args['content'])} chars)."
        if name == "delete_file":
            p = safe_path(args["path"])
            if not p.is_file():
                return f"Error: {args['path']} does not exist."
            p.unlink()
            changed.add(args["path"])
            return f"Deleted {args['path']}."
        return f"Error: unknown tool {name}."
    except Exception as e:  # report errors back to the model, don't crash
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
def run_agent(issue: dict, plan: dict) -> tuple[dict, set[str], dict]:
    client = anthropic.Anthropic()
    changed: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
    messages = [{
        "role": "user",
        "content": (
            f"# Issue #{issue.get('number')}: {issue.get('title')}\n\n"
            f"{issue.get('body') or '(no description)'}\n\n"
            f"# Approved plan\n```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            "Start by calling list_files."
        ),
    }]

    for turn in range(1, MAX_TURNS + 1):
        usage["turns"] = turn
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            print(f"Turn {turn}: no tool call; nudging.")
            messages.append({"role": "user", "content":
                             "Please continue using your tools, and call `finish` when done."})
            continue

        results = []
        for call in tool_calls:
            if call.name == "finish":
                result = dict(call.input)
                result["confidence"] = max(0, min(100, int(result["confidence"])))
                print(f"Turn {turn}: finish (confidence {result['confidence']})")
                return result, changed, usage
            print(f"Turn {turn}: {call.name} {call.input.get('path', '')}")
            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": run_tool(call.name, call.input, changed),
            })
        messages.append({"role": "user", "content": results})

    return ({"summary": "Stopped: hit the turn limit before finishing.",
             "how_to_test": [], "notes": [f"Reached MAX_TURNS={MAX_TURNS}."],
             "confidence": 0, "confidence_reason": "Did not finish."},
            changed, usage)


# ---------------------------------------------------------------------------
# Git + GitHub
# ---------------------------------------------------------------------------
def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def gh(method: str, path: str, payload: dict) -> dict:
    r = requests.request(
        method,
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}{path}",
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept": "application/vnd.github+json"},
        json=payload, timeout=30,
    )
    if r.status_code >= 400:
        sys.exit(f"GitHub API {method} {path} failed: {r.status_code} {r.text}")
    return r.json()


def badge(score: int) -> str:
    return f"🟢 {score}/100" if score >= 80 else (
        f"🟡 {score}/100" if score >= 50 else f"🔴 {score}/100")


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "_None_"


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Code stage")
    parser.add_argument("--issue-file", help="Local issue JSON (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write files locally; skip git push, PR and comments")
    args = parser.parse_args()

    if args.issue_file:
        issue = json.loads(Path(args.issue_file).read_text())
    else:
        issue = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["issue"]
    number = issue.get("number", "local")
    plan_file = OUTPUT_DIR / f"issue-{number}-plan.json"
    if not plan_file.exists():
        sys.exit(f"No plan found at {plan_file}; run the plan stage first.")
    plan = json.loads(plan_file.read_text())["plan"]

    branch = f"ai/issue-{number}"
    if not args.dry_run:
        git("config", "user.name", "github-actions[bot]")
        git("config", "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com")
        git("checkout", "-B", branch)

    result, changed, usage = run_agent(issue, plan)
    (OUTPUT_DIR / f"issue-{number}-code.json").write_text(json.dumps(
        {"stage": "code", "model": MODEL, "issue": number, "result": result,
         "changed": sorted(changed), "usage": usage}, indent=2))

    footer = (f"<sub>Model: {MODEL} · turns: {usage['turns']} · tokens in/out: "
              f"{usage['input_tokens']}/{usage['output_tokens']}</sub>")
    files_md = bullets([f"`{f}`" for f in sorted(changed)])

    if args.dry_run:
        print(f"\nChanged: {sorted(changed)}\n{json.dumps(result, indent=2)}")
        return

    if not changed:
        gh("POST", f"/issues/{number}/comments", {"body":
           f"## 🤖 Code stage — no changes made {badge(result['confidence'])}\n\n"
           f"{result['summary']}\n\n### Notes\n{bullets(result['notes'])}\n\n{footer}"})
        return

    git("add", "-A", "--", *sorted(changed))
    git("commit", "-m", f"AI: {issue.get('title')} (closes #{number})")
    git("push", "--force", "-u", "origin", branch)

    base = os.environ.get("BASE_BRANCH", "main")
    body = f"""Closes #{number}

## 🤖 Code stage — confidence {badge(result['confidence'])}

**Summary:** {result['summary']}

**Why this confidence:** {result['confidence_reason']}

### Files changed
{files_md}

### How to test
{bullets(result['how_to_test'])}

### Notes
{bullets(result['notes'])}

{footer}
"""
    existing = requests.get(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/pulls",
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
        params={"head": f"{os.environ['GITHUB_REPOSITORY'].split('/')[0]}:{branch}",
                "state": "open"}, timeout=30).json()
    if existing:
        pr = gh("PATCH", f"/pulls/{existing[0]['number']}", {"body": body})
    else:
        pr = gh("POST", "/pulls", {"title": f"AI: {issue.get('title')}",
                                   "head": branch, "base": base, "body": body})

    gh("POST", f"/issues/{number}/comments", {"body":
       f"## 🤖 Code stage — confidence {badge(result['confidence'])}\n\n"
       f"Opened pull request #{pr['number']}: {pr['html_url']}\n\n"
       f"{result['summary']}\n\n{footer}"})
    print(f"Pull request: {pr['html_url']}")


if __name__ == "__main__":
    main()
