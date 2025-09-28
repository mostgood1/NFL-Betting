from __future__ import annotations

"""
Trigger the app's admin daily update endpoint.

Usage:
    python scripts/trigger_daily_update.py \
        --base-url https://<your-render-app> \
        --key <ADMIN_KEY> \
        [--push 1] [--auth-mode header|query|bearer]

Environment variable fallbacks:
  ADMIN_BASE_URL or BASE_URL  -> base URL (e.g., https://nfl-betting.onrender.com)
  ADMIN_KEY                   -> admin key for the endpoint

Notes:
    - Default: sends key via X-Admin-Key header (safer for logs)
    - Query mode: calls GET /api/admin/daily-update?push=1&key=...
  - Use --push 0 to run without pushing a commit from the server.
  - Pair with Render Cron Job or any scheduler.
"""

import argparse
import os
import sys
from urllib.parse import urlencode

import requests


def _env(name: str, *alts: str) -> str | None:
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Trigger admin daily update endpoint")
    p.add_argument("--base-url", dest="base_url", default=None, help="Base URL, e.g. https://<app>.onrender.com")
    p.add_argument("--key", dest="key", default=None, help="Admin key (ADMIN_KEY)")
    p.add_argument("--push", dest="push", default="1", help="1 to push commit after run; 0 to skip push")
    p.add_argument("--auth-mode", dest="auth_mode", choices=["header","query","bearer"], default="header",
                   help="How to send the admin key (default: header)")
    args = p.parse_args()

    base_url = args.base_url or _env("ADMIN_BASE_URL", "BASE_URL")
    key = args.key or _env("ADMIN_KEY")
    push_flag = str(args.push or "1").strip()
    auth_mode = args.auth_mode

    if not base_url:
        print("Missing --base-url (or ADMIN_BASE_URL/BASE_URL in env)", file=sys.stderr)
        return 2
    if not key:
        print("Missing --key (or ADMIN_KEY in env)", file=sys.stderr)
        return 2

    if base_url.endswith("/"):
        base_url = base_url[:-1]

    headers = {}
    if auth_mode == "header":
        headers["X-Admin-Key"] = key
        qs = urlencode({"push": push_flag})
    elif auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {key}"
        qs = urlencode({"push": push_flag})
    else:  # query
        qs = urlencode({"push": push_flag, "key": key})

    url = f"{base_url}/api/admin/daily-update?{qs}"
    if auth_mode == "query":
        print(f"Triggering: {url}")
    else:
        print(f"Triggering: {url} (auth: {auth_mode} header)")

    try:
        r = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    print(f"Status: {r.status_code}")
    try:
        print(r.json())
    except Exception:
        print(r.text[:500])
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
