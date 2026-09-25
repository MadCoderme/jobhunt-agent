# Job Search Agent

An extensible, local-first job discovery workflow for technical roles. It collects listings from public job boards and optionally from already-authenticated browser tabs, filters and deduplicates them, scores promising roles with an LLM, and renders a browsable shortlist. High-scoring matches can also receive an ATS-oriented LaTeX resume variant.

The project is intentionally designed as a personal automation tool: credentials stay in a local `.env`, profile data stays in local files, and the shortlist database is accumulated on the machine running the agent.

## Features

- Normalizes listings from Arbeitnow, RemoteOK, Remotive, Jobicy, HN Who Is Hiring, and optional Adzuna sources.
- Captures listings from open Wellfound, Indeed, and Jobright tabs through Chrome DevTools Protocol.
- Applies inexpensive keyword, seniority, remote, salary, freshness, and deduplication filters before LLM calls.
- Scores matches with Gemini and falls back to Groq when configured.
- Persists results in SQLite and renders a cumulative HTML shortlist.
- Optionally selects and tailors one of two LaTeX resume variants for high-scoring jobs.
- Runs once or on a repeating schedule.

## Quick Start

Requirements: Python 3.10+ and API keys for at least one configured LLM provider. Resume tailoring additionally requires a LaTeX installation with `pdflatex`.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env
cp resume.example.txt resume.txt
python job_agent.py
```

For an installed command-line setup, use `pip install -e .` and then run `job-search` or `job-search-scheduler`.

The generated shortlist is written to `jobs_shortlist.html`. The agent stores its accumulated state in `jobs_seen.sqlite3`, so both files are local runtime output and are ignored by Git.

## Configuration

Start with [config.example.yaml](config.example.yaml). The main sections are:

- `preferences`: inexpensive matching filters and the natural-language profile used by scoring.
- `scoring`: score threshold, provider models, and per-run call cap.
- `freshness`: rules for dated and undated listings.
- `resume_tailoring`: optional high-score resume generation and base resume selection.
- `sources`: public API source switches.
- `browser_capture`: CDP connection and site-specific collection limits.

The example uses placeholder profile values. Replace them with your own preferences before running the agent.

## LLM Credentials

Copy [.env.example](.env.example) to `.env` and set one or both provider keys:

```dotenv
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
ADZUNA_APP_ID=optional_app_id
ADZUNA_APP_KEY=optional_app_key
```

Never commit `.env`, resume files, browser state, databases, or generated shortlists.

## Browser Capture

Browser capture reads tabs that you already opened and authenticated yourself. It does not log in, solve CAPTCHAs, or bypass access controls.

1. Quit Chrome completely.
2. Launch a separate Chrome profile with remote debugging enabled:

   ```bash
   google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-agent-profile"
   ```

3. Log in normally and open the target search pages.
4. Set `browser_capture.enabled: true` and run the agent.

Selectors are necessarily best-effort because job sites change their markup. See [browser_capture.py](src/job_search/browser_capture.py) for the selector tables and troubleshooting notes.

## Scheduled Runs

```bash
python run_scheduler.py --once
python run_scheduler.py --interval 30
python run_scheduler.py --config myconfig.yaml --once
```

The scheduler isolates failures to individual cycles and continues until interrupted.

## Project Layout

| Path | Purpose |
| --- | --- |
| `src/job_search/agent.py` | Main collection, filtering, scoring, persistence, and rendering pipeline. |
| `src/job_search/browser_capture.py` | Optional capture from already-open authenticated browser tabs. |
| `src/job_search/resume_tailor.py` | Optional LLM-assisted LaTeX resume tailoring and PDF compilation. |
| `src/job_search/scheduler.py` | Repeats the main pipeline on a fixed interval. |
| Root `*.py` launchers | Backward-compatible commands and imports for existing workflows. |
| `pyproject.toml` | Package metadata and optional console commands. |
| `config.example.yaml` | Sanitized configuration template. |
| `resumes/` | Base LaTeX resume variants used by tailoring. |
| `requirements.txt` | Python dependencies. |

## Development Checks

The project currently has no test suite. Before opening a change, run the lightweight checks below:

```bash
python -m py_compile src/job_search/*.py job_agent.py browser_capture.py resume_tailor.py run_scheduler.py
```

Do not run the full agent as a validation step unless you intend to make network requests and consume provider quota.

## Responsible Use

Respect each job board's terms, robots policies, rate limits, and authentication boundaries. Review every LLM-generated score and resume before applying. Resume tailoring must only reword or reorder facts already present in the selected base resume.