"""
Runs the job-search agent on a repeating schedule (default: every 30 minutes)
so it keeps working in the background while you do other things.

Each cycle re-runs the full pipeline (fetch -> prefilter -> dedupe -> score
-> persist -> render). Results ACCUMULATE across cycles in jobs_seen.sqlite3 -
jobs_shortlist.html always shows every match found so far, with the ones
found in the most recent cycle marked "NEW" and floated to the top.

Leave Chrome running in the background with your Wellfound / Indeed /
Jobright tabs open and logged in (see browser_capture.py for the launch
command) - each cycle reconnects to it, resets each tab back to its original
search page, and reloads it before reading anything, so you're always
scoring current listings rather than a stale page.

Usage:
  python run_scheduler.py                 # every 30 minutes, forever
  python run_scheduler.py --interval 15   # every 15 minutes instead
  python run_scheduler.py --once          # run a single cycle and exit
  python run_scheduler.py --config myconfig.yaml
"""

import argparse
import time
import traceback
from datetime import datetime

from .agent import load_config, load_resume, build_graph


def run_once(config_path: str) -> None:
    cfg = load_config(config_path)
    resume = load_resume(cfg.get("resume_path", "resume.txt"))
    app = build_graph()
    app.invoke({"config": cfg, "resume": resume})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the job agent repeatedly on a fixed interval.")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between cycles (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit, instead of looping")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    args = parser.parse_args()

    cycle = 0
    while True:
        cycle += 1
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n===== Cycle {cycle} started {started} =====")

        try:
            run_once(args.config)
            print(f"[scheduler] Cycle {cycle} completed successfully.")
        except Exception as e:
            # A single bad cycle (e.g. a source API hiccup or Chrome not running)
            # shouldn't kill the whole schedule - log it and try again next cycle.
            print(f"[scheduler] Cycle {cycle} failed: {e}")
            traceback.print_exc()

        if args.once:
            break

        print(f"[scheduler] Sleeping {args.interval} minute(s) until next cycle... (Ctrl+C to stop)")
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n[scheduler] Stopped by user.")
            break


if __name__ == "__main__":
    main()
