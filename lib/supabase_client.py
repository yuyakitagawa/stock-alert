"""Supabase REST API client for stock-alert pipeline."""
import math
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# GitHub Secrets等に混入した前後の空白でホスト名解決が壊れるため必ずstripする
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

# 本番プロジェクトのURLの控え。テストが SUPABASE_URL を差し替えても変わらないので、
# 「いま書こうとしている先が本番か」の判定に使う（_block_production_write）。
_ENV_SUPABASE_URL = SUPABASE_URL

_BATCH_SIZE = 500
_TIMEOUT = 30
_MAX_RETRIES = 3
_RETRY_BACKOFF_SEC = 2


def _request(method: str, url: str, **kwargs) -> requests.Response:
    """requests呼び出しの共通ラッパー。タイムアウト等の一時的なネットワーク失敗は
    指数バックオフ(2s/4s/8s)で最大_MAX_RETRIES回リトライする。単発のタイムアウトで
    数時間かかるバックテスト/日次パイプライン全体が落ちるのを防ぐため。"""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.RequestException as e:
            if attempt == _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF_SEC * (2 ** attempt)
            print(f"[supabase] {method} {url[:80]} failed ({e}), "
                  f"retry {attempt + 1}/{_MAX_RETRIES} in {wait}s")
            time.sleep(wait)


def _sanitize(rows: list[dict]) -> list[dict]:
    """inf/-inf/NaN を None に置換する。1件でも非有限値が混じると
    JSON シリアライズ時に 'Out of range float values' で全 upsert が落ちるため。"""
    out = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _headers(prefer: str = "resolution=merge-duplicates") -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def running_under_test() -> bool:
    """テストランナーの中かどうか。`pytest tests/` と `python3 tests/test_x.py` の両方を見る。"""
    if "pytest" in sys.modules:
        return True
    return os.path.basename(sys.argv[0] or "").startswith("test_")


def _block_production_write(table: str, op: str) -> bool:
    """テスト実行中に本番プロジェクトへ書こうとしていたら True（呼び出しは握りつぶす）。

    なぜ必要か: tests/test_api_usage.py が atexit の flush() で本番 api_usage へ
    合成行を書いていた（2026-08-29、job=local / task="x" / cache_write 1,000,000 / $1.35）。
    その1行だけで当日合計が日次予算 $1.2 を超え、翌営業日の記事生成が全便
    `check_daily_cap()` に打ち切られる状態になっていた。テスト側で書き込みを毎回
    モックする規律に頼ると必ず漏れる（atexit のように後片付けの外で走る経路もある）ため、
    全ての書き込みが通る唯一の出口であるここで止める。

    URLを差し替えているテスト（test_supabase_client.py の example.test）は本番では
    ないので通す。読み取りは止めない（本番データを読むテストは事故ではない）。
    """
    if not (running_under_test() and _ENV_SUPABASE_URL and SUPABASE_URL == _ENV_SUPABASE_URL):
        return False
    print(f"[supabase] ⚠ テスト実行中のため本番への {op} {table} を中止しました")
    return True


def _dedup_batch(batch: list[dict], on_conflict: str) -> list[dict]:
    """バッチ内の重複キーを除去（後勝ち）。同一バッチに同じキーが2回あるとPostgreSQLがエラーになる。"""
    if not on_conflict:
        return batch
    keys = [k.strip() for k in on_conflict.split(",")]
    seen: dict[tuple, int] = {}
    for i, row in enumerate(batch):
        key = tuple(str(row.get(k, "")) for k in keys)
        seen[key] = i
    return [batch[i] for i in sorted(seen.values())]


# 書き込みに失敗したテーブルと行数（プロセス内で累積）。呼び出し側が write_failures() で
# 「保存できたつもりで進んでいないか」を確認できるようにする。
_write_failures: dict[str, int] = {}
# LINEはテーブルごとに1プロセス1回だけ。毎時のジョブで同じ障害を何十通も送らないため。
_notified_tables: set[str] = set()


def _record_write_failure(table: str, rows: int, detail: str) -> None:
    """DB書き込みの失敗を記録し、初回だけLINEへ流す。

    ワークフローの各ステップは continue-on-error で走っているため、書き込みが全滅しても
    ジョブは緑のまま `if: failure()` の通知も鳴らない。実際に2026-08-26〜27、
    edinet_large_holdings の upsert が毎便400で全滅したまま2日間気づけなかった。
    見張りを「ジョブの成否」に依存させず、失敗したその場から直接鳴らす。"""
    _write_failures[table] = _write_failures.get(table, 0) + rows
    if table in _notified_tables:
        return
    _notified_tables.add(table)
    try:
        from lib import notify
        notify.error("DB書き込み", f"{table} への保存に失敗しています（{rows}件）", detail=detail)
    except Exception as e:            # 通知の失敗で本処理を止めない
        print(f"[supabase] 通知に失敗: {e}")


def write_failures() -> dict[str, int]:
    """このプロセスで書き込みに失敗したテーブルと行数。正常なら空dict。"""
    return dict(_write_failures)


def _group_by_keys(batch: list[dict]) -> list[list[dict]]:
    """キー構成が同じ行どうしにまとめる。PostgRESTは1リクエスト内の全オブジェクトの
    キーが一致していないと PGRST102 "All object keys must match" で400を返し、
    バッチ丸ごと落ちる。「値があるときだけ送る」列（issuer_name, short_term_transfers等）が
    混ざると必ず踏むため、送る直前に構成別に分割する。
    （実例: 2026-08-26〜27、edinet_large_holdings の全70件が保存されずブログ記事が0件に）"""
    groups: dict[tuple, list[dict]] = {}
    for row in batch:
        groups.setdefault(tuple(sorted(row.keys())), []).append(row)
    return list(groups.values())


def upsert(table: str, rows: list[dict], on_conflict: str = "") -> bool:
    """全バッチが書けたら True、1バッチでも落ちたら False を返す。
    呼び出し側が戻り値を無視すれば従来どおりの「失敗してもログだけ」の挙動になるが、
    保存できたかどうかがジョブの成否そのものである処理（x_metrics等）は必ず見ること。
    見ていなかったせいで、NOT NULL違反で18行が毎日落ちてもジョブは success のままだった
    （2026-08-24〜25）。"""
    if not rows or not is_configured():
        return False
    if _block_production_write(table, "upsert"):
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    rows = _sanitize(rows)
    ok_all = True
    for i in range(0, len(rows), _BATCH_SIZE):
        deduped = _dedup_batch(rows[i: i + _BATCH_SIZE], on_conflict)
        for batch in _group_by_keys(deduped):
            try:
                resp = _request("POST", url, headers=_headers(), json=batch, timeout=_TIMEOUT)
            except Exception as e:
                print(f"[supabase] {table} upsert exception ({len(batch)} rows): {e}")
                _record_write_failure(table, len(batch), str(e))
                ok_all = False
                continue
            if not resp.ok:
                print(f"[supabase] {table} upsert failed ({len(batch)} rows): "
                      f"{resp.status_code} {resp.text[:500]}")
                _record_write_failure(table, len(batch),
                                      f"HTTP {resp.status_code} {resp.text[:200]}")
                ok_all = False
            else:
                print(f"[supabase] {table} upsert OK ({len(batch)} rows)")
    return ok_all


def insert_ignore(table: str, rows: list[dict], on_conflict: str = "") -> None:
    if not rows or not is_configured():
        return
    if _block_production_write(table, "insert_ignore"):
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
    headers = _headers(prefer="resolution=ignore-duplicates")
    rows = _sanitize(rows)
    for i in range(0, len(rows), _BATCH_SIZE):
        for batch in _group_by_keys(rows[i: i + _BATCH_SIZE]):
            try:
                resp = _request("POST", url, headers=headers, json=batch, timeout=_TIMEOUT)
            except Exception as e:
                print(f"[supabase] {table} insert_ignore exception ({len(batch)} rows): {e}")
                _record_write_failure(table, len(batch), str(e))
                continue
            if not resp.ok:
                print(f"[supabase] {table} insert_ignore failed ({len(batch)} rows): "
                      f"{resp.status_code} {resp.text[:300]}")
                _record_write_failure(table, len(batch),
                                      f"HTTP {resp.status_code} {resp.text[:200]}")


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
        resp = _request("GET", url, headers=_headers(), timeout=_TIMEOUT)
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
    resp = _request("GET", url, headers=headers, timeout=_TIMEOUT)
    if not resp.ok:
        return None
    rows = resp.json()
    return rows[0] if rows else None


def delete(table: str, query: str) -> None:
    if not is_configured():
        return
    if _block_production_write(table, "delete"):
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    _request("DELETE", url, headers=_headers(), timeout=_TIMEOUT)


def update(table: str, query: str, patch: dict) -> bool:
    """PATCH（部分更新）。query に一致する行の patch に含まれる列だけを書き換える。

    upsert() は使えない。PostgRESTのupsertはPOSTの本文に無い列をNULLで埋めるため、
    1列だけ更新するつもりで他の列を全部消す（実測: doc_idとarticle_published_atだけの
    upsertが issuer_code のNOT NULL制約で400になった）。
    """
    if not is_configured():
        return False
    if _block_production_write(table, "update"):
        return False
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    resp = _request("PATCH", url, headers=_headers(prefer="return=minimal"),
                    json=patch, timeout=_TIMEOUT)
    if not resp.ok:
        print(f"[supabase] {table} update failed: {resp.status_code} {resp.text[:200]}")
        _record_write_failure(table, 1, f"HTTP {resp.status_code}")
        return False
    return True


def rpc(fn_name: str, params: dict) -> list | dict | None:
    if not is_configured():
        return None
    url = f"{SUPABASE_URL}/rest/v1/rpc/{fn_name}"
    resp = _request("POST", url, headers=_headers(), json=params, timeout=_TIMEOUT)
    if not resp.ok:
        print(f"[supabase] rpc/{fn_name} failed: {resp.status_code} {resp.text[:200]}")
        return None
    return resp.json()
