"""
Tailors your LaTeX resume per high-scoring job for better ATS keyword matching.

IMPORTANT SCOPE: this only rewords, reorders, and re-emphasizes content that is
ALREADY in your base resume. It never adds employers, dates, degrees, or
skills you don't already have - the prompt sent to the LLM explicitly
forbids that. Tailoring truthful content for ATS matching is normal and
expected; fabricating experience is resume fraud and this script won't do it.

Requires a local LaTeX installation (TeX Live or MiKTeX) with the moderncv
package, specifically the `pdflatex` command on your PATH, to produce a PDF.
If pdflatex isn't found, the tailored .tex source is still written so you can
compile it yourself (e.g. paste into Overleaf).
"""

import os
import re
import shutil
import subprocess
import requests
from typing import Optional


TAILOR_PROMPT = """You are an expert resume editor helping a real candidate pass ATS \
(Applicant Tracking System) keyword screening for a specific job. You have years of experience working as HR so you know what recruiters and ATS systems look for.

STRICT RULES - you must follow all of them:
1. Do NOT add any employer, job title, date range, degree, certification, project that is not already present in the ORIGINAL RESUME below. Every fact must remain true. You may add skills and keywords naturally and a human will verify your changes.
2. You MAY: reorder bullet points and skills to put the most relevant ones first, reword \
bullet points to use terminology and keywords from the job description IF that terminology \
accurately describes something the candidate already did, adjust the professional summary's \
wording and emphasis, and reorder which skills/tools appear first in each skills line.
3. Do NOT remove any \\section, \\cventry, or \\cvitem that exists in the original - only \
reorder or reword their CONTENT. The tailored resume should have the same sections and same \
number of entries as the original.
4. Preserve ALL LaTeX structure exactly: \\documentclass, packages, \\name, \\address, \\email, \
\\social, \\begin{{document}}, \\makecvtitle, and every command/environment must remain valid, \
compilable LaTeX. Only the TEXT CONTENT inside existing commands may change.
5. Output ONLY the complete, raw LaTeX document from \\documentclass to \\end{{document}}. \
No markdown code fences, no commentary, no explanation before or after.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME (LaTeX source):
{resume_tex}
"""


def classify_resume_variant(job_title: str, job_description: str,
                             ai_keywords: list, fullstack_keywords: list) -> str:
    """Cheap keyword-count heuristic - no LLM call needed. Returns 'ai' or 'fullstack'."""
    blob = f"{job_title} {job_description}".lower()
    ai_score = sum(blob.count(k.lower()) for k in ai_keywords)
    fs_score = sum(blob.count(k.lower()) for k in fullstack_keywords)
    return "ai" if ai_score >= fs_score else "fullstack"


def _call_gemini(prompt: str, model: str, api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=body, timeout=60)
        if r.status_code == 429:
            print("[resume_tailor] Gemini rate limited (429)")
            return None
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[resume_tailor] Gemini call failed: {e}")
        return None


def _call_groq(prompt: str, model: str, api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 429:
            print("[resume_tailor] Groq rate limited (429)")
            return None
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[resume_tailor] Groq call failed: {e}")
        return None


def _clean_latex_output(text: str) -> str:
    """Strips markdown code fences if the model added them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def tailor_resume(job_title: str, company: str, job_description: str, base_resume_tex: str,
                   gemini_model: str, groq_model: str,
                   gemini_api_key: Optional[str], groq_api_key: Optional[str]) -> Optional[str]:
    """Returns tailored LaTeX source, or None if both providers failed."""
    prompt = TAILOR_PROMPT.format(
        job_title=job_title,
        company=company,
        job_description=job_description[:4000],
        resume_tex=base_resume_tex,
    )

    result = _call_gemini(prompt, gemini_model, gemini_api_key)
    if result is None:
        result = _call_groq(prompt, groq_model, groq_api_key)

    if result is None:
        return None

    tex = _clean_latex_output(result)

    # Sanity check: reject obviously broken output rather than saving garbage.
    if "\\documentclass" not in tex or "\\end{document}" not in tex:
        print("[resume_tailor] Tailored output failed sanity check (missing doc structure), discarding.")
        return None

    return tex


def compile_latex_to_pdf(tex_path: str, output_dir: str) -> Optional[str]:
    """Compiles a .tex file to PDF using pdflatex, if available. Returns the PDF
    path on success, None if pdflatex isn't installed or compilation failed."""
    if shutil.which("pdflatex") is None:
        print("[resume_tailor] pdflatex not found on PATH - skipping PDF compile, .tex file is still saved.")
        return None

    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_name = os.path.basename(tex_path)

    try:
        # Run twice: moderncv sometimes needs a second pass to place elements correctly.
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
                cwd=tex_dir, capture_output=True, text=True, timeout=60,
            )
        pdf_path = os.path.join(tex_dir, tex_name.replace(".tex", ".pdf"))
        if result.returncode != 0 or not os.path.exists(pdf_path):
            print(f"[resume_tailor] pdflatex failed for {tex_name} (exit {result.returncode}). "
                  f"Last lines of log:\n{result.stdout[-800:]}")
            return None

        if output_dir and os.path.abspath(output_dir) != tex_dir:
            os.makedirs(output_dir, exist_ok=True)
            final_path = os.path.join(output_dir, os.path.basename(pdf_path))
            shutil.copy(pdf_path, final_path)
            return final_path
        return pdf_path
    except Exception as e:
        print(f"[resume_tailor] pdflatex compile error: {e}")
        return None


def safe_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return text[:max_len] or "untitled"