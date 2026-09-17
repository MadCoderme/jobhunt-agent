"""
Autopilot job-search agent.

Pipeline (LangGraph):
  fetch -> prefilter -> dedupe -> score (Gemini, fallback Groq) -> persist -> render

Free API sources: Arbeitnow, RemoteOK, Remotive, Jobicy, (optional) Adzuna.
Optional: reads job cards from your own already-open, logged-in browser tabs
for Wellfound / Indeed / Jobright via browser_capture.py (see that file for setup).

Usage:
  cp config.example.yaml config.yaml   # then edit it
  cp .env.example .env                 # then add your free Gemini/Groq keys
  echo "Paste your resume text here" > resume.txt
  pip install -r requirements.txt
  playwright install chromium          # only needed if you enable browser_capture
  python job_agent.py
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langgraph.graph import StateGraph, END

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_resume(path: str) -> str:
    if not os.path.exists(path):
        print(f"[config] WARNING: resume file '{path}' not found. Scoring will be generic.")
        return "No resume provided."
    with open(path, "r") as f:
        return f.read()


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

class AgentState(TypedDict):
    config: dict
    resume: str
    preferences: str
    raw_jobs: List[Dict]
    filtered_jobs: List[Dict]
    new_jobs: List[Dict]
    scored_jobs: List[Dict]


# --------------------------------------------------------------------------
# Source fetchers - each returns a list of normalized job dicts:
# {id, title, company, location, remote, url, description, source, posted_at}
# --------------------------------------------------------------------------

def fetch_arbeitnow() -> List[Dict]:
    try:
        r = requests.get("https://www.arbeitnow.com/api/job-board-api", timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        jobs = []
        for j in data:
            jobs.append({
                "id": f"arbeitnow:{j.get('slug')}",
                "title": j.get("title", ""),
                "company": j.get("company_name", "Unknown"),
                "location": j.get("location", ""),
                "remote": bool(j.get("remote")),
                "url": j.get("url", ""),
                "description": j.get("description", "")[:4000],
                "source": "arbeitnow",
                "posted_at": j.get("created_at"),
            })
        return jobs
    except Exception as e:
        print(f"[fetch_arbeitnow] failed: {e}")
        return []


def fetch_remoteok() -> List[Dict]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (job-agent personal use)"}
        r = requests.get("https://remoteok.com/api", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        jobs = []
        for j in data:
            if not isinstance(j, dict) or "id" not in j or "position" not in j:
                continue  # first item is a legal notice, not a job
            jobs.append({
                "id": f"remoteok:{j.get('id')}",
                "title": j.get("position", ""),
                "company": j.get("company", "Unknown"),
                "location": j.get("location", "Remote"),
                "remote": True,
                "url": j.get("url", ""),
                "description": j.get("description", "")[:4000],
                "source": "remoteok",
                "posted_at": j.get("date"),
            })
        return jobs
    except Exception as e:
        print(f"[fetch_remoteok] failed: {e}")
        return []


def fetch_remotive() -> List[Dict]:
    try:
        r = requests.get("https://remotive.com/api/remote-jobs", timeout=15)
        r.raise_for_status()
        data = r.json().get("jobs", [])
        jobs = []
        for j in data:
            jobs.append({
                "id": f"remotive:{j.get('id')}",
                "title": j.get("title", ""),
                "company": j.get("company_name", "Unknown"),
                "location": j.get("candidate_required_location", "Remote"),
                "remote": True,
                "url": j.get("url", ""),
                "description": j.get("description", "")[:4000],
                "source": "remotive",
                "posted_at": j.get("publication_date"),
            })
        return jobs
    except Exception as e:
        print(f"[fetch_remotive] failed: {e}")
        return []


def fetch_jobicy() -> List[Dict]:
    try:
        r = requests.get("https://jobicy.com/api/v2/remote-jobs?count=50", timeout=15)
        r.raise_for_status()
        data = r.json().get("jobs", [])
        jobs = []
        for j in data:
            jobs.append({
                "id": f"jobicy:{j.get('id')}",
                "title": j.get("jobTitle", ""),
                "company": j.get("companyName", "Unknown"),
                "location": j.get("jobGeo", "Remote"),
                "remote": True,
                "url": j.get("url", ""),
                "description": j.get("jobDescription", "")[:4000],
                "source": "jobicy",
                "posted_at": j.get("pubDate"),
            })
        return jobs
    except Exception as e:
        print(f"[fetch_jobicy] failed: {e}")
        return []


def fetch_adzuna(app_id: str, app_key: str, what: str = "software engineer", country: str = "us") -> List[Dict]:
    if not app_id or not app_key:
        return []
    try:
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": what,
            "results_per_page": 50,
            "content-type": "application/json",
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("results", [])
        jobs = []
        for j in data:
            jobs.append({
                "id": f"adzuna:{j.get('id')}",
                "title": j.get("title", ""),
                "company": (j.get("company") or {}).get("display_name", "Unknown"),
                "location": (j.get("location") or {}).get("display_name", ""),
                "remote": "remote" in (j.get("title", "") + j.get("description", "")).lower(),
                "url": j.get("redirect_url", ""),
                "description": j.get("description", "")[:4000],
                "source": "adzuna",
                "posted_at": j.get("created"),
            })
        return jobs
    except Exception as e:
        print(f"[fetch_adzuna] failed: {e}")
        return []


# --------------------------------------------------------------------------
# Node: fetch
# --------------------------------------------------------------------------

def fetch_node(state: AgentState) -> dict:
    cfg = state["config"]
    src_cfg = cfg.get("sources", {})
    jobs: List[Dict] = []

    if src_cfg.get("arbeitnow"):
        jobs += fetch_arbeitnow()
    if src_cfg.get("remoteok"):
        jobs += fetch_remoteok()
    if src_cfg.get("remotive"):
        jobs += fetch_remotive()
    if src_cfg.get("jobicy"):
        jobs += fetch_jobicy()
    if src_cfg.get("adzuna"):
        jobs += fetch_adzuna(ADZUNA_APP_ID, ADZUNA_APP_KEY)

    bc_cfg = cfg.get("browser_capture", {})
    if bc_cfg.get("enabled"):
        try:
            from browser_capture import capture_open_tabs
            jobs += capture_open_tabs(
                cdp_url=bc_cfg.get("cdp_url", "http://localhost:9222"),
                url_patterns=bc_cfg.get("site_url_patterns", []),
                scroll_passes=bc_cfg.get("scroll_passes", 6),
                scroll_pause_seconds=bc_cfg.get("scroll_pause_seconds", 1.2),
                visit_detail_pages=bc_cfg.get("visit_detail_pages", False),
                max_detail_pages_per_run=bc_cfg.get("max_detail_pages_per_run", 30),
                detail_page_delay_seconds=bc_cfg.get("detail_page_delay_seconds", 1.5),
                indeed_max_pages=bc_cfg.get("indeed_max_pages", 3),
            )
        except Exception as e:
            print(f"[fetch_node] browser_capture failed: {e}")

    print(f"[fetch] total raw jobs: {len(jobs)}")
    return {"raw_jobs": jobs}


# --------------------------------------------------------------------------
# Node: prefilter (cheap keyword rules, runs before any LLM call)
# --------------------------------------------------------------------------

def prefilter_node(state: AgentState) -> dict:
    cfg = state["config"]["preferences"]
    exclude_seniority = [s.lower() for s in cfg.get("exclude_seniority", [])]
    include_keywords = [s.lower() for s in cfg.get("include_keywords", [])]
    exclude_keywords = [s.lower() for s in cfg.get("exclude_keywords", [])]
    remote_only = cfg.get("remote_only", False)

    kept = []
    for j in state["raw_jobs"]:
        title = (j.get("title") or "").lower()
        desc = (j.get("description") or "").lower()
        blob = title + " " + desc

        structured_seniority = (j.get("seniority") or "").lower()
        if structured_seniority:
            if any(term in structured_seniority for term in exclude_seniority):
                continue
        elif any(term in title for term in exclude_seniority):
            continue
        
        if exclude_keywords and any(term in blob for term in exclude_keywords):
            continue
        if include_keywords and not any(term in blob for term in include_keywords):
            continue
        if remote_only and not j.get("remote", False) and "remote" not in blob:
            continue

        kept.append(j)

    print(f"[prefilter] {len(state['raw_jobs'])} -> {len(kept)} after keyword rules")
    return {"filtered_jobs": kept}


# --------------------------------------------------------------------------
# Node: dedupe against local SQLite "seen" store
# --------------------------------------------------------------------------

def _job_hash(job: Dict) -> str:
    key = job.get("id") or job.get("url") or (job.get("title", "") + job.get("company", ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            hash TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            url TEXT,
            source TEXT,
            score INTEGER,
            verdict TEXT,
            reasons TEXT,
            scored_by TEXT,
            first_seen TEXT
        )
    """)
    # Migration: fill in any columns missing from a DB created before this schema.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(seen_jobs)").fetchall()}
    for col, coltype in [("location", "TEXT"), ("source", "TEXT"), ("reasons", "TEXT"), ("scored_by", "TEXT")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE seen_jobs ADD COLUMN {col} {coltype}")
    conn.commit()
    return conn


def dedupe_node(state: AgentState) -> dict:
    db_path = state["config"].get("db_path", "jobs_seen.sqlite3")
    conn = _init_db(db_path)
    cur = conn.cursor()

    new_jobs = []
    for j in state["filtered_jobs"]:
        h = _job_hash(j)
        cur.execute("SELECT 1 FROM seen_jobs WHERE hash = ?", (h,))
        if cur.fetchone() is None:
            j["_hash"] = h
            new_jobs.append(j)

    conn.close()

    max_jobs = state["config"].get("scoring", {}).get("max_jobs_per_run", 40)
    new_jobs = new_jobs[:max_jobs]

    print(f"[dedupe] {len(new_jobs)} new jobs to score this run (capped at {max_jobs})")
    return {"new_jobs": new_jobs}


# --------------------------------------------------------------------------
# Node: score with Gemini, fallback to Groq
# --------------------------------------------------------------------------

SCORING_PROMPT = """You are a strict job-matching assistant. Compare the CANDIDATE RESUME and CANDIDATE PREFERENCES to the JOB POSTING and score fit from 0-100.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching this schema exactly:
{{"score": <integer 0-100>, "verdict": "<one short phrase>", "reasons": ["<reason1>", "<reason2>"]}}

CANDIDATE RESUME:
{resume}

CANDIDATE PREFERENCES:
{preferences}

JOB TITLE: {title}
COMPANY: {company}
JOB DESCRIPTION:
{description}
"""


def _parse_llm_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None


def score_with_gemini(prompt: str, model: str) -> Optional[dict]:
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    try:
        r = requests.post(url, json=body, timeout=30)
        if r.status_code == 429:
            print("[score_with_gemini] rate limited (429)")
            return None
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_json(text)
    except Exception as e:
        print(f"[score_with_gemini] failed: {e}")
        return None


def score_with_groq(prompt: str, model: str) -> Optional[dict]:
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        if r.status_code == 429:
            print("[score_with_groq] rate limited (429)")
            return None
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_llm_json(text)
    except Exception as e:
        print(f"[score_with_groq] failed: {e}")
        return None


def score_node(state: AgentState) -> dict:
    scoring_cfg = state["config"].get("scoring", {})
    gemini_model = scoring_cfg.get("gemini_model", "gemini-3.5-flash-lite")
    groq_model = scoring_cfg.get("groq_model", "llama-3.3-70b-versatile")
    resume = state["resume"]
    preferences = scoring_cfg.get("preferences", "No preferences provided.")

    scored = []
    for j in state["new_jobs"]:
        prompt = SCORING_PROMPT.format(
            resume=resume[:6000],
            preferences=preferences,
            title=j.get("title", ""),
            company=j.get("company", ""),
            description=(j.get("description") or "")[:3000],
        )

        result = score_with_gemini(prompt, gemini_model)
        used = "gemini"
        if result is None:
            result = score_with_groq(prompt, groq_model)
            used = "groq"

        if result is None:
            print(f"[score] both providers failed for '{j.get('title')}' - skipping this run")
            continue

        j["score"] = int(result.get("score", 0))
        j["verdict"] = result.get("verdict", "")
        j["reasons"] = result.get("reasons", [])
        j["scored_by"] = used
        scored.append(j)

        time.sleep(1.2)  # gentle pacing to stay well inside free-tier RPM limits

    print(f"[score] scored {len(scored)} jobs")
    return {"scored_jobs": scored}


# --------------------------------------------------------------------------
# Node: persist to SQLite (mark as seen, regardless of score)
# --------------------------------------------------------------------------

def persist_node(state: AgentState) -> dict:
    db_path = state["config"].get("db_path", "jobs_seen.sqlite3")
    conn = _init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
 
    for j in state["scored_jobs"]:
        conn.execute(
            "INSERT OR IGNORE INTO seen_jobs "
            "(hash, title, company, location, url, source, score, verdict, reasons, scored_by, first_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (j["_hash"], j.get("title"), j.get("company"), j.get("location"), j.get("url"),
             j.get("source"), j.get("score", 0), j.get("verdict", ""),
             json.dumps(j.get("reasons", [])), j.get("scored_by"), now),
        )
    conn.commit()
    conn.close()
    return {}


# --------------------------------------------------------------------------
# Node: render HTML shortlist
# --------------------------------------------------------------------------

 
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Job Shortlist - {{ generated_at }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background:#fafafa; color:#222; }
  h1 { font-size: 1.4rem; }
  .job { background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:16px 20px; margin-bottom:14px; position:relative; }
  .job.is-new { border-color:#2e8b57; box-shadow:0 0 0 1px #2e8b57; }
  .new-badge { position:absolute; top:14px; right:16px; background:#2e8b57; color:#fff; font-size:0.72rem; font-weight:bold; padding:2px 8px; border-radius:10px; letter-spacing:0.03em; }
  .job h2 { margin:0 0 4px 0; font-size:1.1rem; padding-right:70px; }
  .meta { color:#666; font-size:0.9rem; margin-bottom:8px; }
  .score { display:inline-block; font-weight:bold; padding:2px 10px; border-radius:12px; color:#fff; }
  .reasons { margin:8px 0 0 0; padding-left:18px; font-size:0.92rem; color:#333; }
  a.apply { display:inline-block; margin-top:8px; font-size:0.9rem; }
  .summary { color:#555; }
</style>
</head>
<body>
<h1>Job Shortlist &mdash; last updated {{ generated_at }}</h1>
<p class="summary">{{ jobs|length }} jobs at or above your score threshold, accumulated across all runs since this list was started. {{ new_count }} new since the last run.</p>
{% for j in jobs %}
<div class="job {{ 'is-new' if j.is_new else '' }}">
  {% if j.is_new %}<span class="new-badge">NEW</span>{% endif %}
  <h2>{{ j.title }} &mdash; {{ j.company }}</h2>
  <div class="meta">{{ j.location }} &middot; source: {{ j.source }} &middot; scored by {{ j.scored_by }} &middot; first seen {{ j.first_seen_display }}</div>
  <span class="score" style="background: {{ 'seagreen' if j.score >= 80 else ('darkorange' if j.score >= 65 else 'gray') }};">{{ j.score }}/100</span>
  &mdash; {{ j.verdict }}
  <ul class="reasons">
    {% for r in j.reasons %}<li>{{ r }}</li>{% endfor %}
  </ul>
  <a class="apply" href="{{ j.url }}" target="_blank">Open posting &rarr;</a>
</div>
{% endfor %}
</body>
</html>
"""
 
 
def render_node(state: AgentState) -> dict:
    threshold = state["config"].get("scoring", {}).get("score_threshold", 65)
    db_path = state["config"].get("db_path", "jobs_seen.sqlite3")
 
    new_hashes = {j["_hash"] for j in state["scored_jobs"]}
 
    conn = _init_db(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM seen_jobs WHERE score >= ? ORDER BY score DESC, first_seen DESC",
        (threshold,),
    ).fetchall()
    conn.close()
 
    jobs = []
    for row in rows:
        first_seen_raw = row["first_seen"] or ""
        try:
            first_seen_display = datetime.fromisoformat(first_seen_raw).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            first_seen_display = first_seen_raw
 
        try:
            reasons = json.loads(row["reasons"]) if row["reasons"] else []
        except Exception:
            reasons = []
 
        jobs.append({
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "url": row["url"],
            "source": row["source"],
            "score": row["score"],
            "verdict": row["verdict"],
            "reasons": reasons,
            "scored_by": row["scored_by"],
            "first_seen_display": first_seen_display,
            "is_new": row["hash"] in new_hashes,
        })
 
    # New-this-run jobs float to the top regardless of score, so you notice them.
    jobs.sort(key=lambda j: (not j["is_new"], -j["score"]))
 
    html = Template(HTML_TEMPLATE).render(
        jobs=jobs,
        new_count=len(new_hashes),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
 
    out_path = state["config"].get("output_html", "jobs_shortlist.html")
    with open(out_path, "w") as f:
        f.write(html)
 
    print(f"[render] {len(jobs)} total matches >= {threshold} ({len(new_hashes)} new this run) written to {out_path}")
    return {}


# --------------------------------------------------------------------------
# Build and run the graph
# --------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("prefilter", prefilter_node)
    workflow.add_node("dedupe", dedupe_node)
    workflow.add_node("score", score_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("render", render_node)

    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "prefilter")
    workflow.add_edge("prefilter", "dedupe")
    workflow.add_edge("dedupe", "score")
    workflow.add_edge("score", "persist")
    workflow.add_edge("persist", "render")
    workflow.add_edge("render", END)

    return workflow.compile()


def main():
    cfg = load_config("config.yaml")
    resume = load_resume(cfg.get("resume_path", "resume.txt"))

    app = build_graph()
    app.invoke({"config": cfg, "resume": resume})


if __name__ == "__main__":
    main()