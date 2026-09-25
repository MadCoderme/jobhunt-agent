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
    cp resume.example.txt resume.txt     # then replace it with your resume
  pip install -r requirements.txt
  playwright install chromium          # only needed if you enable browser_capture
  python job_agent.py
"""

import os
import re
import json
import time
import sqlite3
import hashlib
import email.utils
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import TypedDict, List, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langgraph.graph import StateGraph, END

from . import resume_tailor

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
    funded_companies: List[Dict]


# --------------------------------------------------------------------------
# Freshness helpers - resolve each job's actual posting date from whatever
# format its source gives us, so old listings can be filtered/cleaned up.
# --------------------------------------------------------------------------

def _parse_relative_date(text: Optional[str]) -> Optional[datetime]:
    """Parses phrases like 'Posted yesterday', '3 days ago', 'Reposted 36
    minutes ago' into an actual datetime. Returns None if nothing matches."""
    if not text:
        return None
    t = text.lower()
    now = datetime.now(timezone.utc)

    if "just posted" in t or re.search(r"\btoday\b", t):
        return now
    if "yesterday" in t:
        return now - timedelta(days=1)

    m = re.search(r"(\d+)\+?\s*(minute|hour|day|week|month)s?\s*ago", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {
            "minute": timedelta(minutes=n),
            "hour": timedelta(hours=n),
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=30 * n),
        }[unit]
        return now - delta

    return None


def _resolve_posted_at(job: Dict) -> Optional[datetime]:
    """Tries, in order: a relative-time tag captured off a browser card,
    a relative-time phrase inside the description/card text, then a
    structured posted_at field from an API source (unix timestamp, ISO 8601,
    or RFC 2822). Returns None if the posting date genuinely can't be
    determined - callers treat unknown age as 'keep' by default rather than
    silently dropping possibly-good matches."""
    parsed = _parse_relative_date(job.get("posted_tag"))
    if parsed:
        return parsed

    parsed = _parse_relative_date((job.get("description") or "")[:300])
    if parsed:
        return parsed

    raw = job.get("posted_at")
    if not raw:
        return None

    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(raw, str):
        raw_stripped = raw.strip()
        if raw_stripped.isdigit():
            try:
                return datetime.fromtimestamp(int(raw_stripped), tz=timezone.utc)
            except Exception:
                pass
        try:
            dt = datetime.fromisoformat(raw_stripped.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
        try:
            dt = email.utils.parsedate_to_datetime(raw_stripped)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
        except Exception:
            pass

    return None


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


def fetch_hn_who_is_hiring() -> List[Dict]:
    """Hacker News' monthly 'Ask HN: Who is hiring?' thread - real listings
    posted directly by hiring companies, via HN's official, free, no-key
    Algolia search API. Each top-level comment is one listing; the format
    varies but usually starts 'Company | Role | Location | ...'."""
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"tags": "story,author_whoishiring", "query": "Who is Hiring", "hitsPerPage": 5},
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json().get("hits", [])
        thread = next((h for h in hits if "who is hiring" in (h.get("title") or "").lower()), None)
        if not thread:
            print("[fetch_hn_who_is_hiring] could not find the current thread")
            return []

        story_id = thread["objectID"]
        r2 = requests.get(f"https://hn.algolia.com/api/v1/items/{story_id}", timeout=20)
        r2.raise_for_status()
        children = r2.json().get("children", []) or []

        jobs = []
        for c in children:
            text = c.get("text") or ""
            if not text:
                continue
            clean = re.sub(r"<[^>]+>", " ", text)
            for a, b in [("&amp;", "&"), ("&gt;", ">"), ("&lt;", "<"), ("&#x27;", "'"), ("&quot;", '"')]:
                clean = clean.replace(a, b)
            clean = clean.strip()
            first_line = clean.split("\n")[0][:160]
            company_guess = re.split(r"[|\u2013-]", first_line)[0].strip() or "Unknown"

            jobs.append({
                "id": f"hn_whoishiring:{c.get('id')}",
                "title": first_line or "See listing",
                "company": company_guess,
                "location": "See listing",
                "remote": "remote" in clean.lower(),
                "url": f"https://news.ycombinator.com/item?id={c.get('id')}",
                "description": clean[:4000],
                "source": "hn_whoishiring",
                "posted_at": c.get("created_at"),
            })
        print(f"[fetch_hn_who_is_hiring] parsed {len(jobs)} listings from thread {story_id}")
        return jobs
    except Exception as e:
        print(f"[fetch_hn_who_is_hiring] failed: {e}")
        return []


def fetch_recently_funded_companies(lookback_days: int = 14) -> List[Dict]:
    """Recently-funded-startup leads (YC and otherwise) from two free,
    official sources: HN's funding-announcement stories (Algolia search API)
    and TechCrunch's public venture RSS feed. This surfaces company NAMES as
    leads worth checking manually - it does not claim to know their actual
    open roles, since no free API reliably provides that across arbitrary
    companies."""
    leads: List[Dict] = []

    try:
        cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
        for query in ["raises seed", "raises Series", "Series A funding", "Series B funding", "Y Combinator"]:
            r = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={
                    "tags": "story",
                    "query": query,
                    "numericFilters": f"created_at_i>{cutoff_ts}",
                    "hitsPerPage": 15,
                },
                timeout=15,
            )
            r.raise_for_status()
            for h in r.json().get("hits", []):
                title = (h.get("title") or "").strip()
                if not title:
                    continue
                company = re.split(r"\braises\b|\braised\b", title, flags=re.IGNORECASE)[0].strip()
                leads.append({
                    "name": company[:120] or title[:120],
                    "blurb": title,
                    "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    "source": "hn",
                })
    except Exception as e:
        print(f"[fetch_recently_funded_companies] HN query failed: {e}")

    try:
        r = requests.get("https://techcrunch.com/category/venture/feed/", timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keywords = ["raises", "raised", "series a", "series b", "series c", "seed round", "funding"]
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not any(k in title.lower() for k in keywords):
                continue
            company = re.split(r"\braises\b|\braised\b", title, flags=re.IGNORECASE)[0].strip()
            leads.append({
                "name": company[:120] or title[:120],
                "blurb": title,
                "url": link,
                "source": "techcrunch",
            })
    except Exception as e:
        print(f"[fetch_recently_funded_companies] TechCrunch RSS failed: {e}")

    seen = set()
    deduped = []
    for lead in leads:
        key = lead["name"].lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(lead)

    print(f"[fetch_recently_funded_companies] found {len(deduped)} funding leads")
    return deduped


# --------------------------------------------------------------------------
# Node: fetch
# --------------------------------------------------------------------------

def fetch_node(state: AgentState) -> dict:
    cfg = state["config"]
    src_cfg = cfg.get("sources", {})
    db_path = cfg.get("db_path", "jobs_seen.sqlite3")
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
    if src_cfg.get("hn_who_is_hiring"):
        jobs += fetch_hn_who_is_hiring()

    bc_cfg = cfg.get("browser_capture", {})
    if bc_cfg.get("enabled"):
        try:
            from .browser_capture import capture_open_tabs
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

    # Recently-funded-startup leads: fetched, persisted so they accumulate
    # across cycles too, then used to tag any job below whose company matches.
    funded_cfg = cfg.get("funded_startups", {})
    funded_companies: List[Dict] = []
    if funded_cfg.get("enabled", True):
        lookback_days = funded_cfg.get("lookback_days", 14)
        try:
            fresh_leads = fetch_recently_funded_companies(lookback_days=lookback_days)
            _persist_funded_leads(db_path, fresh_leads)
        except Exception as e:
            print(f"[fetch_node] funded-startup lead fetch failed: {e}")
        funded_companies = _load_recent_funded_leads(db_path, lookback_days)

    if funded_companies:
        for j in jobs:
            company_lower = (j.get("company") or "").lower().strip()
            if not company_lower or company_lower == "unknown":
                continue
            for lead in funded_companies:
                name_lower = lead["name"].lower().strip()
                if name_lower and (name_lower in company_lower or company_lower in name_lower):
                    j["funded_recently"] = True
                    j["funding_note"] = lead.get("blurb", "")
                    break

    print(f"[fetch] total raw jobs: {len(jobs)} ({len(funded_companies)} funded-startup leads on file)")
    return {"raw_jobs": jobs, "funded_companies": funded_companies}


# --------------------------------------------------------------------------
# Node: prefilter (cheap keyword rules, runs before any LLM call)
# --------------------------------------------------------------------------

def prefilter_node(state: AgentState) -> dict:
    cfg = state["config"]["preferences"]
    freshness_cfg = state["config"].get("freshness", {})
    max_age_days = freshness_cfg.get("max_age_days", 7)
    drop_unknown_age = freshness_cfg.get("drop_unknown_age", False)

    exclude_seniority = [s.lower() for s in cfg.get("exclude_seniority", [])]
    include_keywords = [s.lower() for s in cfg.get("include_keywords", [])]
    exclude_keywords = [s.lower() for s in cfg.get("exclude_keywords", [])]
    remote_only = cfg.get("remote_only", False)

    now = datetime.now(timezone.utc)
    kept = []
    dropped_stale = 0
    for j in state["raw_jobs"]:
        title = (j.get("title") or "").lower()
        desc = (j.get("description") or "").lower()
        blob = title + " " + desc

        # Jobright cards expose an explicit seniority string (e.g. "Mid Level",
        # "Senior") - trust that over guessing from the title when present.
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

        posted_dt = _resolve_posted_at(j)
        if posted_dt:
            age_days = (now - posted_dt).total_seconds() / 86400
            if age_days > max_age_days:
                dropped_stale += 1
                continue
            j["_posted_at_resolved"] = posted_dt.isoformat()
        else:
            j["_posted_at_resolved"] = None
            if drop_unknown_age:
                dropped_stale += 1
                continue

        kept.append(j)

    print(f"[prefilter] {len(state['raw_jobs'])} -> {len(kept)} after keyword rules "
          f"({dropped_stale} dropped for being older than {max_age_days}d or unparseable age)")
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
    new_cols = [
        ("location", "TEXT"), ("source", "TEXT"), ("reasons", "TEXT"), ("scored_by", "TEXT"),
        ("posted_at", "TEXT"), ("resume_pdf_path", "TEXT"),
        ("funded_recently", "INTEGER"), ("funding_note", "TEXT"),
    ]
    for col, coltype in new_cols:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE seen_jobs ADD COLUMN {col} {coltype}")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS funded_leads (
            name TEXT PRIMARY KEY,
            blurb TEXT,
            url TEXT,
            source TEXT,
            found_at TEXT
        )
    """)
    conn.commit()
    return conn


def _persist_funded_leads(db_path: str, leads: List[Dict]) -> None:
    if not leads:
        return
    conn = _init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    for lead in leads:
        # Preserve the original found_at on repeat sightings of the same
        # company rather than resetting its "how recent" clock every cycle.
        conn.execute(
            "INSERT INTO funded_leads (name, blurb, url, source, found_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET blurb=excluded.blurb, url=excluded.url, source=excluded.source",
            (lead["name"], lead.get("blurb", ""), lead.get("url", ""), lead.get("source", ""), now),
        )
    conn.commit()
    conn.close()


def _load_recent_funded_leads(db_path: str, lookback_days: int) -> List[Dict]:
    conn = _init_db(db_path)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM funded_leads WHERE found_at >= ? ORDER BY found_at DESC", (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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

        time.sleep(5)  # gentle pacing to stay well inside free-tier RPM limits

    print(f"[score] scored {len(scored)} jobs")
    return {"scored_jobs": scored}


# --------------------------------------------------------------------------
# Node: tailor resume (only for high-potential jobs) - rewords/reorders
# TRUTHFUL content from your base LaTeX resume for ATS keyword matching.
# Never invents experience - see resume_tailor.py for the enforced rules.
# --------------------------------------------------------------------------

def tailor_node(state: AgentState) -> dict:
    cfg = state["config"].get("resume_tailoring", {})
    if not cfg.get("enabled", False):
        return {}

    min_score = cfg.get("min_score", 85)
    output_dir = cfg.get("output_dir", "tailored_resumes")
    os.makedirs(output_dir, exist_ok=True)

    resumes_cfg = cfg.get("resumes", {})
    ai_path = resumes_cfg.get("ai")
    fs_path = resumes_cfg.get("fullstack")
    if not ai_path and not fs_path:
        print("[tailor] no resume paths configured under resume_tailoring.resumes - skipping")
        return {}

    ai_keywords = cfg.get("ai_keywords", [])
    fs_keywords = cfg.get("fullstack_keywords", [])
    gemini_model = cfg.get("gemini_model", "gemini-3.8-flash")
    groq_model = cfg.get("groq_model", "llama-3.3-70b-versatile")

    resume_cache: Dict[str, Optional[str]] = {}

    def _load_variant(variant: str) -> Optional[str]:
        path = ai_path if variant == "ai" else fs_path
        if not path:
            return None
        if path not in resume_cache:
            if not os.path.exists(path):
                print(f"[tailor] resume file not found: {path}")
                resume_cache[path] = None
            else:
                with open(path, "r") as f:
                    resume_cache[path] = f.read()
        return resume_cache[path]

    candidates = [j for j in state["scored_jobs"] if j.get("score", 0) >= min_score]
    print(f"[tailor] {len(candidates)} job(s) at or above min_score={min_score} for resume tailoring")

    tailored_count = 0
    for j in candidates:
        variant = resume_tailor.classify_resume_variant(
            j.get("title", ""), j.get("description", ""), ai_keywords, fs_keywords
        )
        base_tex = _load_variant(variant)
        if not base_tex:
            continue

        tex = resume_tailor.tailor_resume(
            job_title=j.get("title", ""),
            company=j.get("company", ""),
            job_description=j.get("description", ""),
            base_resume_tex=base_tex,
            gemini_model=gemini_model,
            groq_model=groq_model,
            gemini_api_key=GEMINI_API_KEY,
            groq_api_key=GROQ_API_KEY,
        )
        if not tex:
            print(f"[tailor] tailoring failed for '{j.get('title')}' at {j.get('company')} - skipping")
            continue

        base_name = resume_tailor.safe_filename(
            f"{j.get('company', '')}_{j.get('title', '')}_{j.get('_hash', '')[:8]}"
        )
        tex_path = os.path.join(output_dir, f"{base_name}.tex")
        with open(tex_path, "w") as f:
            f.write(tex)

        pdf_path = resume_tailor.compile_latex_to_pdf(tex_path, output_dir)

        j["resume_variant"] = variant
        j["resume_tex_path"] = tex_path
        j["resume_pdf_path"] = pdf_path

        tailored_count += 1
        time.sleep(5)  # another LLM call - same gentle pacing as scoring

    print(f"[tailor] produced tailored resumes for {tailored_count} job(s)")
    return {}


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
            "(hash, title, company, location, url, source, score, verdict, reasons, scored_by, "
            "first_seen, posted_at, resume_pdf_path, funded_recently, funding_note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (j["_hash"], j.get("title"), j.get("company"), j.get("location"), j.get("url"),
             j.get("source"), j.get("score", 0), j.get("verdict", ""),
             json.dumps(j.get("reasons", [])), j.get("scored_by"), now,
             j.get("_posted_at_resolved"), j.get("resume_pdf_path"),
             1 if j.get("funded_recently") else 0, j.get("funding_note", "")),
        )
    conn.commit()
    conn.close()
    return {}


# --------------------------------------------------------------------------
# Node: cleanup - prunes stale entries from the accumulated list so the
# output doesn't grow forever and doesn't keep showing week-old postings.
# --------------------------------------------------------------------------

def cleanup_node(state: AgentState) -> dict:
    freshness_cfg = state["config"].get("freshness", {})
    max_age_days = freshness_cfg.get("max_age_days", 7)
    max_unknown_age_days = freshness_cfg.get("max_unknown_age_days", 30)
    db_path = state["config"].get("db_path", "jobs_seen.sqlite3")

    now = datetime.now(timezone.utc)
    posted_cutoff = (now - timedelta(days=max_age_days)).isoformat()
    unknown_cutoff = (now - timedelta(days=max_unknown_age_days)).isoformat()

    conn = _init_db(db_path)
    cur = conn.execute(
        "DELETE FROM seen_jobs WHERE "
        "(posted_at IS NOT NULL AND posted_at != '' AND posted_at < ?) "
        "OR ((posted_at IS NULL OR posted_at = '') AND first_seen < ?)",
        (posted_cutoff, unknown_cutoff),
    )
    deleted = cur.rowcount
    conn.execute("DELETE FROM funded_leads WHERE found_at < ?", (unknown_cutoff,))
    conn.commit()
    conn.close()

    if deleted:
        print(f"[cleanup] pruned {deleted} stale job(s) - older than {max_age_days}d, "
              f"or unknown-age and older than {max_unknown_age_days}d")
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
  h2.section-title { font-size: 1.1rem; margin-top: 2.5em; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  .job { background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:16px 20px; margin-bottom:14px; position:relative; }
  .job.is-new { border-color:#2e8b57; box-shadow:0 0 0 1px #2e8b57; }
  .new-badge { position:absolute; top:14px; right:16px; background:#2e8b57; color:#fff; font-size:0.72rem; font-weight:bold; padding:2px 8px; border-radius:10px; letter-spacing:0.03em; }
  .funded-badge { display:inline-block; background:#fff3cd; color:#8a6100; font-size:0.72rem; font-weight:bold; padding:2px 8px; border-radius:10px; margin-left:6px; vertical-align:middle; }
  .job h2 { margin:0 0 4px 0; font-size:1.1rem; padding-right:70px; }
  .meta { color:#666; font-size:0.9rem; margin-bottom:8px; }
  .score { display:inline-block; font-weight:bold; padding:2px 10px; border-radius:12px; color:#fff; }
  .reasons { margin:8px 0 0 0; padding-left:18px; font-size:0.92rem; color:#333; }
  a.apply, a.resume-link { display:inline-block; margin-top:8px; margin-right:14px; font-size:0.9rem; }
  .summary { color:#555; }
  .lead { background:#fff; border:1px solid #eee; border-radius:8px; padding:10px 16px; margin-bottom:8px; font-size:0.92rem; }
</style>
</head>
<body>
<h1>Job Shortlist &mdash; last updated {{ generated_at }}</h1>
<p class="summary">{{ jobs|length }} jobs at or above your score threshold, accumulated across all runs. {{ new_count }} new since the last run. Listings older than your configured freshness window are pruned automatically.</p>
{% for j in jobs %}
<div class="job {{ 'is-new' if j.is_new else '' }}">
  {% if j.is_new %}<span class="new-badge">NEW</span>{% endif %}
  <h2>{{ j.title }} &mdash; {{ j.company }}{% if j.funded_recently %}<span class="funded-badge">recently funded</span>{% endif %}</h2>
  <div class="meta">{{ j.location }} &middot; source: {{ j.source }} &middot; scored by {{ j.scored_by }} &middot; first seen {{ j.first_seen_display }}</div>
  <span class="score" style="background: {{ 'seagreen' if j.score >= 80 else ('darkorange' if j.score >= 65 else 'gray') }};">{{ j.score }}/100</span>
  &mdash; {{ j.verdict }}
  <ul class="reasons">
    {% for r in j.reasons %}<li>{{ r }}</li>{% endfor %}
  </ul>
  {% if j.funding_note %}<div class="meta">{{ j.funding_note }}</div>{% endif %}
  <div>
    <a class="apply" href="{{ j.url }}" target="_blank">Open posting &rarr;</a>
    {% if j.resume_pdf_path %}<a class="resume-link" href="{{ j.resume_pdf_path }}" target="_blank">Tailored resume (PDF) &rarr;</a>{% endif %}
  </div>
</div>
{% endfor %}

{% if funded_leads %}
<h2 class="section-title">Recently Funded Startups &amp; YC Companies</h2>
<p class="summary">Leads to check manually, not confirmed open roles - job listings at these companies aren't guaranteed to show up in the sources above.</p>
{% for lead in funded_leads %}
<div class="lead"><strong>{{ lead.name }}</strong> &mdash; {{ lead.blurb }} &middot; <a href="{{ lead.url }}" target="_blank">source ({{ lead.source }})</a></div>
{% endfor %}
{% endif %}
</body>
</html>
"""


def render_node(state: AgentState) -> dict:
    threshold = state["config"].get("scoring", {}).get("score_threshold", 70)
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
            "resume_pdf_path": row["resume_pdf_path"],
            "funded_recently": bool(row["funded_recently"]),
            "funding_note": row["funding_note"],
        })

    # New-this-run jobs float to the top regardless of score, so you notice them.
    jobs.sort(key=lambda j: (not j["is_new"], -j["score"]))

    funded_leads = state.get("funded_companies", [])

    html = Template(HTML_TEMPLATE).render(
        jobs=jobs,
        new_count=len(new_hashes),
        funded_leads=funded_leads,
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
    workflow.add_node("tailor", tailor_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("cleanup", cleanup_node)
    workflow.add_node("render", render_node)

    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "prefilter")
    workflow.add_edge("prefilter", "dedupe")
    workflow.add_edge("dedupe", "score")
    workflow.add_edge("score", "tailor")
    workflow.add_edge("tailor", "persist")
    workflow.add_edge("persist", "cleanup")
    workflow.add_edge("cleanup", "render")
    workflow.add_edge("render", END)

    return workflow.compile()


def main():
    cfg = load_config("config.yaml")
    resume = load_resume(cfg.get("resume_path", "resume.txt"))

    app = build_graph()
    app.invoke({"config": cfg, "resume": resume})


if __name__ == "__main__":
    main()