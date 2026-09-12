# THE AGENT

> Watch AI do a job, not just chat.

A local-only Engineers' Day prototype that demonstrates an agent loop:

```text
Task → Plan → Select safe tool → Execute → Observe → Recover/retry → Complete
```

## Run locally

Prerequisites: Python 3.11+, [Ollama](https://ollama.com), and the local model:

```powershell
ollama run qwen3:1.7b
```

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, normally `http://localhost:8501`.

## Judge demo script

1. **Student Performance Analyst** — identifies Ishaan and Rohan as students who may need attention.
2. **Environmental Research** — Qwen selects the local Knowledge Base and returns an EV-versus-petrol comparison.
3. **Error Recovery** — strict analysis detects invalid/missing scores, replans, safely excludes them, retries, and completes with average performance **83.33**.

### Custom task examples

Choose **Custom task** and try one of these:

- `Who are the top students in the local performance data?`
- `Which students have low attendance?`
- `Give me a summary of the student score distribution.`
- `Which students may need support based on their scores?`

Each request is classified locally by Qwen into an allowlisted analysis goal; the result card changes to match that goal.

Use the timeline on the Error Recovery demo to show the actual sequence:

```text
Planning → Tool selected → Executing → Error detected → Replanning → Retrying → Completed
```

## Architecture and safety

- **Local Qwen 1.7B via Ollama:** classifies the task into one allowlisted workflow only.
- **Python agent loop:** owns execution, observes each `ToolResult`, and applies the recovery policy.
- **Safe tool registry:** exposes only `data_analyzer`, `calculator`, and `knowledge_base`.
- **Local data:** CSV and JSON files in `data/`; no cloud service, API key, or external database.
- **No arbitrary execution:** the model cannot use the shell, access arbitrary paths, delete files, or run Python code.

If Qwen/Ollama is unavailable temporarily, a local keyword fallback keeps the three prepared demos reliable while preserving the same tool boundary.

## Test

```powershell
python -m unittest -v test_step2.py
```

The test suite checks safe tool restrictions, the actual invalid-data recovery/retry flow, all three demo data paths, and constrained model-plan validation.
