"""
Reads job listings out of browser tabs YOU already have open and logged into.

This does NOT log in, does NOT solve captchas, and does NOT bypass any
anti-bot protection - it connects to your existing, human-authenticated
Chrome session via the DevTools Protocol and reads/scrolls tabs that are
already open, the same way a browser extension would.

SETUP (do this once per machine):
  1. Fully quit Chrome.
  2. Launch it with remote debugging enabled, e.g. on macOS:
       /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \
         --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-agent-profile"
     On Windows (from cmd):
       "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ^
         --remote-debugging-port=9222 --user-data-dir="C:\\chrome-agent-profile"
     On Linux:
       google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-agent-profile"
  3. Log into Wellfound / Indeed / Jobright normally in that Chrome window and
     open your search-results pages (leave them open, as you said).
  4. Run the main agent - it will attach to this Chrome instance.

CSS SELECTORS WILL DRIFT. These sites redesign their DOM regularly and are
JS-heavy, so the selectors below are a best-effort starting point, not a
guarantee. If a site returns zero jobs, open DevTools (F12) on that tab,
inspect a job card, and update the selector list for that site below.
"""

import os
import json
import time
from typing import List, Dict

from playwright.sync_api import sync_playwright


# Persists each site's original search-results URL across separate script runs
# (each 30-minute cycle is a fresh Python process, so this can't just live in
# a variable) - lets us reset a tab back to page 1 / a fresh search on every
# cycle instead of drifting further through pagination each time.
_TAB_ORIGIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".browser_tab_origins.json")


def _load_tab_origins() -> Dict[str, str]:
    if os.path.exists(_TAB_ORIGIN_FILE):
        try:
            with open(_TAB_ORIGIN_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_tab_origins(origins: Dict[str, str]) -> None:
    try:
        with open(_TAB_ORIGIN_FILE, "w") as f:
            json.dump(origins, f)
    except Exception as e:
        print(f"[browser_capture] could not save tab origin cache: {e}")


def _reset_or_reload_tab(page, site: str) -> None:
    """On the very first cycle, remembers this tab's URL as the site's 'home'
    search page and reloads it fresh. On every later cycle, navigates back to
    that same home URL (undoing any pagination drift) and loads it fresh, so
    each 30-minute run starts from page 1 with current results rather than
    wherever the previous run's pagination left off."""
    origins = _load_tab_origins()
    home_url = origins.get(site)

    try:
        if home_url and page.url != home_url:
            page.goto(home_url, wait_until="domcontentloaded", timeout=20000)
        else:
            if not home_url:
                origins[site] = page.url
                _save_tab_origins(origins)
            page.reload(wait_until="domcontentloaded", timeout=20000)
        time.sleep(1.0)  # let client-side rendering settle after navigation
    except Exception as e:
        print(f"[browser_capture] {site}: reload/reset failed ({e}), continuing with tab as-is")


# Best-effort selectors per site. Update these if a site changes its markup -
# right-click a job card -> Inspect -> find the repeating container class.
SITE_SELECTORS = {
    "wellfound.com": {
        "card": 'a[class*="jobLink"]',
        "title": 'span[class*="_title__"]',
        "company": 'span[class*="__wf_unused_placeholder__"]',  # not queried directly - filled by _enrich_wellfound_stubs
        "link": 'a[class*="__wf_unused_placeholder__"]',        # forces fallback to the card element itself
    },
    "indeed.com": {
        "card": "div.job_seen_beacon, td.resultContent",
        "title": "h2.jobTitle span, a.jcs-JobTitle span",
        "company": "span.companyName",
        "link": "a.jcs-JobTitle, h2.jobTitle a",
    },
    "jobright.ai": {
        "card": 'div[data-tut="jobs-card-match-score"], div[class*="index_job-card__"]',
        "title": 'h2[class*="job-title"]',
        "company": 'div[class*="company-name"]',
        "link": 'a[href^="/jobs/info/"]',
    },
}

# Jobright cards expose extra structured fields beyond title/company/link -
# these are worth pulling since they're more reliable than keyword-guessing
# (e.g. an explicit "Mid Level" / "Senior" string instead of parsing the title).
JOBRIGHT_EXTRA_SELECTORS = {
    "location": 'span[class*="primary-location"]',
    "seniority": 'div:has(svg[aria-label="seniority"]) span',
    "remote_tag": 'div:has(svg[aria-label="remote"]) span',
    "experience": 'div:has(svg[aria-label="date"]) span',
    "match_score": 'span[class*="percent-value"]',
}

# Selectors for the FULL job description once you're on a job's own detail page.
# These are best-effort guesses and will need verification/adjustment via DevTools -
# see the note at the top of this file.
DETAIL_SELECTORS = {
    "wellfound.com": "[data-test='JobDetails'], div[class*='styles_description'], div[class*='JobDescription']",
    "indeed.com": "#jobDescriptionText",
    "jobright.ai": "div[class*='index_jobDetailContent__'], div[class*='JobDescription']",
}

# Indeed paginates with a "Next" link/button rather than infinite scroll.
PAGINATION_SELECTORS = {
    "indeed.com": "a[data-testid='pagination-page-next'], a[aria-label='Next Page']",
}


def _match_site(url: str):
    for domain in SITE_SELECTORS:
        if domain in url:
            return domain
    return None


def _resolve_href(href: str, base_url: str) -> str:
    if href.startswith("/"):
        from urllib.parse import urlparse
        base = urlparse(base_url)
        return f"{base.scheme}://{base.netloc}{href}"
    return href


def _collect_cards_from_page(page, site: str, selectors: dict) -> List[Dict]:
    """Extracts card-level stubs (title, company, url, snippet) from the
    currently-loaded listing page."""
    stubs = []
    try:
        cards = page.query_selector_all(selectors["card"])
    except Exception as e:
        print(f"[browser_capture] Selector error on {site}: {e}")
        return stubs

    for card in cards:
        try:
            title_el = card.query_selector(selectors["title"])
            company_el = card.query_selector(selectors["company"])
            link_el = card.query_selector(selectors["link"]) or card

            title = title_el.inner_text().strip() if title_el else None
            company = company_el.inner_text().strip() if company_el else None
            href = link_el.get_attribute("href") if link_el else None
            if not title or not href:
                continue
            href = _resolve_href(href, page.url)

            stubs.append({
                "id": f"{site}:{href}",
                "title": title,
                "company": company or "Unknown",
                "location": "Remote",
                "remote": True,
                "url": href,
                "description": card.inner_text()[:2000],  # placeholder, replaced if detail visit succeeds
                "source": site,
                "posted_at": None,
            })
        except Exception:
            continue

    if site == "jobright.ai":
        _enrich_jobright_stubs(cards, stubs)
    elif site == "wellfound.com":
        _enrich_wellfound_stubs(cards, stubs)

    return stubs


def _enrich_jobright_stubs(cards, stubs: List[Dict]) -> None:
    """Pulls Jobright's extra structured fields (seniority, match score, etc.)
    into the corresponding stub. Matches by index since cards/stubs are built
    in the same iteration order, skipping any card that failed earlier."""
    # Rebuild a lookup since some cards may have been skipped above (no title/href).
    stub_by_index = {i: s for i, s in enumerate(stubs)}
    valid_card_idx = 0
    for card in cards:
        try:
            title_el = card.query_selector(SITE_SELECTORS["jobright.ai"]["title"])
            link_el = card.query_selector(SITE_SELECTORS["jobright.ai"]["link"])
            if not title_el or not link_el:
                continue  # this card was skipped when building stubs too

            stub = stub_by_index.get(valid_card_idx)
            valid_card_idx += 1
            if not stub:
                continue

            for field, sel in JOBRIGHT_EXTRA_SELECTORS.items():
                el = card.query_selector(sel)
                if el:
                    text = el.inner_text().strip()
                    if field == "location" and text:
                        stub["location"] = text
                    elif field == "remote_tag":
                        stub["remote"] = "remote" in text.lower()
                    elif field in ("seniority", "experience", "match_score"):
                        stub[field] = text
        except Exception:
            continue


def _enrich_wellfound_stubs(cards, stubs: List[Dict]) -> None:
    """Wellfound nests each job link inside a company block
    (data-test="StartupResult") rather than repeating the company name on
    every card, so the company name has to be found by walking UP from the
    job link to its enclosing company container. Also pulls location,
    compensation, and the 'Posted X ago' tag, all of which sit alongside the
    title inside the same link element."""
    stub_by_index = {i: s for i, s in enumerate(stubs)}
    valid_idx = 0
    title_selector = SITE_SELECTORS["wellfound.com"]["title"]

    for card in cards:
        try:
            title_el = card.query_selector(title_selector)
            href = card.get_attribute("href")
            if not title_el or not href:
                continue  # this card was skipped when building stubs too

            stub = stub_by_index.get(valid_idx)
            valid_idx += 1
            if not stub:
                continue

            try:
                company_name = card.evaluate(
                    """el => {
                        const startup = el.closest('[data-test="StartupResult"]');
                        const h2 = startup ? startup.querySelector('h2') : null;
                        return h2 ? h2.innerText.trim() : '';
                    }"""
                )
            except Exception:
                company_name = ""
            if company_name:
                stub["company"] = company_name

            loc_els = card.query_selector_all('span[class*="_location__"]')
            loc_texts = [e.inner_text().strip() for e in loc_els if e.inner_text().strip()]
            if loc_texts:
                stub["location"] = ", ".join(loc_texts)
                stub["remote"] = any("remote" in t.lower() for t in loc_texts)

            comp_el = card.query_selector('span[class*="_compensation__"]')
            if comp_el:
                stub["compensation"] = comp_el.inner_text().strip()

            tags_el = card.query_selector('div[class*="_tags__"]')
            if tags_el:
                stub["posted_tag"] = tags_el.inner_text().strip()
        except Exception:
            continue


def _paginate_and_collect(page, site: str, selectors: dict,
                           scroll_passes: int, scroll_pause_seconds: float,
                           max_pages: int) -> List[Dict]:
    """Handles both infinite-scroll sites (Wellfound, Jobright) and
    page-by-page sites (Indeed). Returns deduped stubs across all pages
    visited."""
    all_stubs: Dict[str, Dict] = {}

    pagination_selector = PAGINATION_SELECTORS.get(site)

    for page_num in range(1, max_pages + 1):
        # Scroll to trigger any lazy-loaded content on the current page/view.
        for _ in range(scroll_passes):
            try:
                page.mouse.wheel(0, 2000)
                time.sleep(scroll_pause_seconds)
            except Exception:
                break

        for stub in _collect_cards_from_page(page, site, selectors):
            all_stubs[stub["url"]] = stub

        if not pagination_selector:
            break  # infinite-scroll site: scrolling already loaded everything it's going to

        if page_num >= max_pages:
            break

        try:
            next_link = page.query_selector(pagination_selector)
            if not next_link:
                print(f"[browser_capture] {site}: no further pagination link found, stopping at page {page_num}")
                break
            next_link.click()
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(1.5)  # human-like pause between pages
        except Exception as e:
            print(f"[browser_capture] {site}: pagination click failed ({e}), stopping at page {page_num}")
            break

    return list(all_stubs.values())


def _visit_detail_pages(context, stubs: List[Dict], remaining_budget: int,
                         delay_seconds: float) -> List[Dict]:
    """Opens each stub's URL in a new tab within the same (already logged-in)
    browser context, extracts the full job description, and merges it in.
    Respects a global remaining_budget of detail-page visits for this run."""
    enriched = []
    for stub in stubs:
        if remaining_budget <= 0:
            enriched.append(stub)  # keep the card-snippet version, just don't visit
            continue

        site = stub["source"]
        detail_selector = DETAIL_SELECTORS.get(site)
        if not detail_selector:
            enriched.append(stub)
            continue

        detail_page = None
        try:
            detail_page = context.new_page()
            detail_page.goto(stub["url"], timeout=20000, wait_until="domcontentloaded")
            detail_page.wait_for_timeout(1500)
            desc_el = detail_page.query_selector(detail_selector)
            if desc_el:
                full_text = desc_el.inner_text().strip()
                if full_text:
                    stub["description"] = full_text[:6000]
        except Exception as e:
            print(f"[browser_capture] detail visit failed for {stub['url']}: {e}")
        finally:
            if detail_page:
                try:
                    detail_page.close()
                except Exception:
                    pass

        remaining_budget -= 1
        enriched.append(stub)
        time.sleep(delay_seconds)

    return enriched


def capture_open_tabs(cdp_url: str, url_patterns: List[str],
                       scroll_passes: int = 6, scroll_pause_seconds: float = 1.2,
                       visit_detail_pages: bool = False,
                       max_detail_pages_per_run: int = 30,
                       detail_page_delay_seconds: float = 1.5,
                       indeed_max_pages: int = 3) -> List[Dict]:
    """
    Connects to an already-running Chrome via CDP, finds tabs matching
    url_patterns, pages through results (Indeed) or scrolls (Wellfound,
    Jobright) to gather job stubs, then optionally visits each job's own
    detail page (within the same logged-in context) to pull the full
    description instead of just the card snippet.

    Returns a list of normalized job dicts, same schema as the API sources:
    {id, title, company, location, remote, url, description, source, posted_at}
    """
    jobs: List[Dict] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_url)
        except Exception as e:
            print(f"[browser_capture] Could not connect to Chrome at {cdp_url}: {e}")
            print("[browser_capture] Make sure Chrome is running with --remote-debugging-port=9222")
            return jobs

        detail_budget = max_detail_pages_per_run

        for context in browser.contexts:
            for page in context.pages:
                url = page.url
                if not any(pat in url for pat in url_patterns):
                    continue

                site = _match_site(url)
                if not site:
                    continue

                _reset_or_reload_tab(page, site)

                selectors = SITE_SELECTORS[site]
                max_pages = indeed_max_pages if site == "indeed.com" else 1

                stubs = _paginate_and_collect(
                    page, site, selectors,
                    scroll_passes, scroll_pause_seconds, max_pages,
                )
                print(f"[browser_capture] {site}: collected {len(stubs)} listing stubs")

                if visit_detail_pages and stubs:
                    before = detail_budget
                    stubs = _visit_detail_pages(context, stubs, detail_budget, detail_page_delay_seconds)
                    visited = min(before, len(stubs))
                    detail_budget = max(0, detail_budget - visited)

                jobs.extend(stubs)

        browser.close()

    print(f"[browser_capture] Captured {len(jobs)} total jobs from open tabs.")
    return jobs