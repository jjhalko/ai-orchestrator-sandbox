"""
Rung 1 of the mini orchestrator: the PLAN stage.

Flow:
  1. Read the GitHub issue that was labeled `ai-ready`.
  2. Gather a slice of the repo as context (file list + small text files).
  3. Ask Claude for a structured plan, forced through a "tool" so the
     output always matches our JSON schema.
  4. Post the plan as a comment on the issue and save the raw JSON
     (later rungs and the dashboard will read it).

Run modes:
  - In GitHub Actions: env vars are set by the workflow.
  - Locally (no GitHub needed):
      python orchestrator/plan.py --issue-file orchestrator/sample_issue.json --dry-run
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

# ---------------------------------------------------------------------------
# Settings — tweak these freely
# ---------------------------------------------------------------------------
MODEL = os.environ.get("ORCH_MODEL", "claude-sonnet-5-5")
MAX_OUTPUT_TOKENS = 4000          # cost cap on the response
CONTEXT_CHAR_BUDGET = 60_000      # cost cap on how much repo we send
MAX_FILE_CHARS = 12_000           # skip giant files
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".html", ".css", ".md", ".yml", ".yaml",
    ".json", ".toml", ".txt", ".sh", ".cs", ".sql",
}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".venv"}
OUTPUT_DIR = Path("orchestrator_runs")

# ---------------------------------------------------------------------------
# The "tool" Claude should call. With "strict": True the API guarantees the
# tool's input matches this schema exactly. (Sonnet 5.5 doesn't allow
# *forcing* a tool call, so the prompt tells it to use the tool, and
# request_plan() retries if it answers in plain text instead.)
# Strict mode needs "additionalProperties": False on every object.
# ---------------------------------------------------------------------------
PLAN_TOOL = {
    "name": "submit_plan",
    "description": (
        "Submit the implementation plan for this issue. "
        "Always use this tool to respond; never reply in plain text."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {
                "type": "string",
                "description": "One or two sentences: what will be done.",
            },
            "files_to_change": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "change": {"type": "string"},
                    },
                    "required": ["path", "change"],
                },
            },
            "steps": {"type": "array", "items": {"type": "string"}},
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How a human could verify the work is done.",
            },
            "risks": {"type": "array", "items": {"type": "string"}},
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Anything ambiguous that a human should answer.",
            },
            "confidence": {
                "type": "integer",
                "description": "0-100: how confident you are this plan is correct and complete.",
            },
            "confidence_reason": {"type": "string"},
        },
        "required": [
            "summary", "files_to_change", "steps", "acceptance_criteria",
            "risks", "open_questions", "confidence", "confidence_reason",
        ],
    },
}

SYSTEM_PROMPT = """You are the PLANNING stage of a software delivery pipeline.
You receive a GitHub issue and a snapshot of the repository.
Produce a concrete, minimal implementation plan that a separate coding agent
will follow. Do not write the code itself.

Rules:
- Prefer the smallest change that satisfies the issue.
- Only reference files that exist in the snapshot, unless you say a file must be created.
- If the issue is vague, list open_questions and lower your confidence.
- Calibrate confidence honestly:
    90-100 = trivial, unambiguous, all relevant code visible
    70-89  = clear, minor unknowns
    40-69  = notable ambiguity or missing context
    0-39   = should not proceed without human input
Always respond by calling the submit_plan tool."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_issue(issue_file: str | None) -> dict:
    """Get the issue from a local JSON file or from the GitHub event payload."""
    if issue_file:
        return json.loads(Path(issue_file).read_text())
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        sys.exit("No --issue-file given and GITHUB_EVENT_PATH not set.")
    event = json.loads(Path(event_path).read_text())
    return event["issue"]


def list_repo_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout
        files = out.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [str(p) for p in Path(".").rglob("*") if p.is_file()]
    return [
        f for f in files
        if not any(part in SKIP_DIRS for part in Path(f).parts)
        and not f.startswith(str(OUTPUT_DIR))
    ]


def build_repo_context(files: list[str]) -> str:
    """File tree + as many small text files as fit in the budget."""
    parts = ["## File tree", *files, "", "## File contents"]
    used = sum(len(p) for p in parts)
    skipped = []
    for f in files:
        path = Path(f)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if len(text) > MAX_FILE_CHARS or used + len(text) > CONTEXT_CHAR_BUDGET:
            skipped.append(f)
            continue
        block = f"\n### {f}\n```\n{text}\n```"
        parts.append(block)
        used += len(block)
    if skipped:
        parts.append("\n(Not included for size: " + ", ".join(skipped) + ")")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The Claude call
# ---------------------------------------------------------------------------
def request_plan(issue: dict, repo_context: str) -> tuple[dict, dict]:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    user_msg = (
        f"# Issue #{issue.get('number', '?')}: {issue.get('title', '')}\n\n"
        f"{issue.get('body') or '(no description)'}\n\n"
        f"# Repository snapshot\n{repo_context}"
    )
    messages = [{
        "role": "user",
        "content": user_msg + "\n\nRespond by calling the submit_plan tool.",
    }]
    usage = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(1, 3):  # one retry if it answers in plain text
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[PLAN_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        tool_call = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_call:
            plan = dict(tool_call.input)
            plan["confidence"] = max(0, min(100, int(plan["confidence"])))
            return plan, usage

        print(f"Attempt {attempt}: model replied without the tool; retrying.")
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": "Please submit that plan using the submit_plan tool."},
        ]

    sys.exit("Model did not call submit_plan after 2 attempts.")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def confidence_badge(score: int) -> str:
    if score >= 80:
        return f"🟢 {score}/100"
    if score >= 50:
        return f"🟡 {score}/100"
    return f"🔴 {score}/100"


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "_None_"


def format_comment(plan: dict, usage: dict) -> str:
    files = "\n".join(
        f"- `{f['path']}` — {f['change']}" for f in plan["files_to_change"]
    ) or "_None_"
    steps = "\n".join(f"{n}. {s}" for n, s in enumerate(plan["steps"], 1))
    return f"""## 🤖 Plan stage — confidence {confidence_badge(plan['confidence'])}

**Summary:** {plan['summary']}

**Why this confidence:** {plan['confidence_reason']}

### Files to change
{files}

### Steps
{steps}

### Acceptance criteria
{bullets(plan['acceptance_criteria'])}

### Risks
{bullets(plan['risks'])}

### Open questions
{bullets(plan['open_questions'])}

<sub>Model: {MODEL} · tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}</sub>
"""


def post_comment(issue_number: int, body: str) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]          # e.g. john/the-daily-run
    token = os.environ["GITHUB_TOKEN"]
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"body": body},
        timeout=30,
    )
    r.raise_for_status()


def save_run(issue: dict, plan: dict, usage: dict) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"issue-{issue.get('number', 'local')}-plan.json"
    path.write_text(json.dumps(
        {"stage": "plan", "model": MODEL, "issue": issue.get("number"),
         "plan": plan, "usage": usage},
        indent=2,
    ))
    return path


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plan stage")
    parser.add_argument("--issue-file", help="Local issue JSON (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the comment instead of posting it")
    args = parser.parse_args()

    issue = load_issue(args.issue_file)
    files = list_repo_files()
    context = build_repo_context(files)
    print(f"Issue #{issue.get('number')}: {issue.get('title')}")
    print(f"Sending {len(files)} files listed, ~{len(context):,} chars of context")

    plan, usage = request_plan(issue, context)
    comment = format_comment(plan, usage)
    saved = save_run(issue, plan, usage)
    print(f"Saved raw plan to {saved}")

    if args.dry_run:
        print("\n" + comment)
    else:
        post_comment(issue["number"], comment)
        print("Posted plan comment to the issue.")


if __name__ == "__main__":
    main()
