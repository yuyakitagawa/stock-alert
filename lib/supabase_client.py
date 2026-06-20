"""Supabase REST API client for stock-alert pipeline."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

_BATCH_SIZE = 500
_TIMEOUT = 30


def _headers(prefer: str = "resolution=merge-duplicates") -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def upsert(table: str, rows: list[dict], on_conflict: str = "") -> None:
    if not rows or not is_configured():
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i: i + _BATCH_SIZE]
        resp = requests.post(url, headers=_headers(), json=batch, timeout=_TIMEOUT)
        if not resp.ok:
            print(f"[supabase] {table} upsert failed ({len(batch)} rows): "
                  f"{resp.status_code} {resp.text[:300]}")


def insert_ignore(table: str, rows: list[dict], on_conflict: str = "") -> None:
    if not rows or not is_configured():
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    headers = _headers(prefer="resolution=ignore-duplicates")
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i: i + _BATCH_SIZE]
        resp = requests.post(url, headers=headers, json=batch, timeout=_TIMEOUT)
        if not resp.ok:
            print(f"[supabase] {table} insert_ignore failed ({len(batch)} rows): "
                  f"{resp.status_code} {resp.text[:300]}")


def select(table: str, query: str = "", limit: int = 0) -> list[dict]:
    if not is_configured():
        return []
    base = f"{SUPABASE_URL}/rest/v1/{table}"
    page_size = 1000
    offset = 0
    out: list[dict] = []
    while True:
        parts = [query] if query else []
        parts.append(f"limit={page_size}&offset={offset}")
        url = f"{base}?{'&'.join(parts)}"
        resp = requests.get(url, headers=_headers(), timeout=_TIMEOUT)
        if not resp.ok:
            print(f"[supabase] {table} select failed: {resp.status_code} {resp.text[:200]}")
            break
        rows = resp.json()
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
        if limit and len(out) >= limit:
            out = out[:limit]
            break
    return out


def select_one(table: str, query: str = "") -> dict | None:
    if not is_configured():
        return None
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}&limit=1"
    headers = _headers()
    headers["Accept"] = "application/json"
    resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
    if not resp.ok:
        return None
    rows = resp.json()
    return rows[0] if rows else None


def delete(table: str, query: str) -> None:
    if not is_configured():
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    requests.delete(url, headers=_headers(), timeout=_TIMEOUT)


def rpc(fn_name: str, params: dict) -> list | dict | None:
    if not is_configured():
        return None
    url = f"{SUPABASE_URL}/rest/v1/rpc/{fn_name}"
    resp = requests.post(url, headers=_headers(), json=params, timeout=_TIMEOUT)
    if not resp.ok:
        print(f"[supabase] rpc/{fn_name} failed: {resp.status_code} {resp.text[:200]}")
        return None
    return resp.json()
