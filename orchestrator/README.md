# Mini Orchestrator — Rung 1: Plan Stage

Label a GitHub issue `ai-ready` and Claude posts a structured implementation
plan, with a confidence score, as a comment. Read-only: it never changes code.

## Files

| Path | What it does |
|---|---|
| `.github/workflows/ai-plan.yml` | Runs on the `ai-ready` label |
| `orchestrator/plan.py` | The plan stage (Python) |
| `orchestrator/requirements.txt` | `anthropic`, `requests` |
| `orchestrator/sample_issue.json` | Fake issue for local testing |

## Setup (about 10 minutes)

1. Copy `.github/` and `orchestrator/` into the root of your repo (e.g. The Daily Run). Commit and push.
2. Add `orchestrator_runs/` to `.gitignore`.
3. If The Daily Run already has an `ANTHROPIC_API_KEY` secret, you're set. Otherwise:
   **Settings → Secrets and variables → Actions → New repository secret** → `ANTHROPIC_API_KEY`.
4. Create a label named `ai-ready` (**Issues → Labels → New label**).
5. Open an issue (try the text in `sample_issue.json`), then add the `ai-ready` label.
6. Watch **Actions**; in a minute or so the plan comment appears on the issue.

`GITHUB_TOKEN` is provided automatically by Actions — no setup needed.

## Run it locally first (optional, recommended)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r orchestrator/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                    # Windows: $env:ANTHROPIC_API_KEY="..."
python orchestrator/plan.py --issue-file orchestrator/sample_issue.json --dry-run
```

`--dry-run` prints the comment instead of posting it. Run it from the repo root
so it can see your files.

## Cost guardrails

- `CONTEXT_CHAR_BUDGET` (60k chars, roughly 15k tokens) caps what's sent.
- `MAX_OUTPUT_TOKENS` caps the response.
- Each run prints its token usage in the comment footer.
- Change the model with an `ORCH_MODEL` env var.

## Things to try (this is the learning part)

1. Write a deliberately vague issue. Does confidence drop and do open questions appear?
2. Edit `SYSTEM_PROMPT` and compare plans for the same issue.
3. Open `orchestrator_runs/*.json` — this is what later stages and the dashboard will read.
4. Read `PLAN_TOOL`: forcing a tool call is how you get guaranteed-shape JSON out of a model.

## Next rungs

2. **Code stage** — follow the plan on a new branch and open a PR.
3. **Test + review stages** — run tests, critique the diff, retry on failure.
4. **Dashboard** — stage-by-stage status and confidence history.
