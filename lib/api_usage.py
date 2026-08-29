"""
lib/api_usage.py
Anthropic APIの利用量（トークン・Web検索回数・推定コスト）をSupabaseに記録する。

なぜ必要か: 2026-08-23に月次上限へ到達したとき、「何が・どれだけ使ったか」を示す
記録がどこにも無かった。バックフィルのログを後からgrepして犯人を推定するしかなく、
用途別の内訳（会社説明バックフィルが上限を消した）を出すのに丸一日かかっている。
呼び出しごとにusageを残しておけば、上限に近づく前に内訳を出して止められる。

使い方:
    from lib import api_usage

    resp = client.messages.create(model=CLAUDE_MODEL, ...)
    api_usage.record(resp, task="blog_body")

集計は (UTC日付, ジョブ, タスク, モデル) 単位でプロセス内に貯め、プロセス終了時
(atexit)にまとめて書く。1回のAPI呼び出しごとにHTTPを足さないので既存の処理時間に
影響しない。記録は計測であって目的ではないため、集計も書き込みも失敗しても
例外を呼び出し側に伝播させない。

月次上限はUTC月初に戻る（"You will regain access on 2026-09-01 at 00:00 UTC."）ため、
日付はJSTではなくUTCで持つ。
"""
import atexit
import os
from datetime import datetime, timezone

# 1Mトークンあたりの単価(USD)。キャッシュ書き込みは入力の1.25倍、読み出しは0.1倍。
# 出典: Anthropic公開価格（2026-06時点）。モデルを増やしたらここに追記する。
_PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
}
_CACHE_WRITE_RATE = 1.25
_CACHE_READ_RATE = 0.10

# Web検索は $10 / 1,000検索。検索結果本文は入力トークンとしても別途課金される。
_WEB_SEARCH_USD = 0.01

_TABLE = "api_usage"

# key: (usage_date, job, task, model) -> 集計値
_buffer: dict[tuple, dict] = {}
_atexit_registered = False


def _model_key(model: str) -> str:
    """"claude-haiku-4-5-20251001" のような日付サフィックスを落として単価表を引く。"""
    name = str(model or "unknown")
    if name in _PRICES:
        return name
    for known in _PRICES:
        if name.startswith(known):
            return known
    return name


def _int(v) -> int:
    """usageの値を安全にintへ。Mockやnullが来ても0にして呼び出し側を止めない。"""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def estimate_cost(model: str, *, input_tokens: int = 0, output_tokens: int = 0,
                  cache_write_tokens: int = 0, cache_read_tokens: int = 0,
                  web_search_requests: int = 0) -> float:
    """推定コスト(USD)。単価表に無いモデルはトークン分を0として検索料だけ数える。"""
    price = _PRICES.get(_model_key(model))
    cost = _WEB_SEARCH_USD * web_search_requests
    if price:
        cost += (input_tokens * price["input"]
                 + output_tokens * price["output"]
                 + cache_write_tokens * price["input"] * _CACHE_WRITE_RATE
                 + cache_read_tokens * price["input"] * _CACHE_READ_RATE) / 1_000_000
    return cost


def _job_name() -> str:
    """どのジョブの消費かを識別する名前。GitHub Actions以外はlocal。"""
    return os.getenv("GITHUB_WORKFLOW") or "local"


def record(resp, *, task: str, model: str = "") -> None:
    """messages.create() のレスポンスからusageを1件積む。

    modelは省略時 resp.model を使う。resp が usage を持たない（テストのMock等）
    場合は何も積まない。
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        counts = {
            "input_tokens": _int(getattr(usage, "input_tokens", 0)),
            "output_tokens": _int(getattr(usage, "output_tokens", 0)),
            "cache_write_tokens": _int(getattr(usage, "cache_creation_input_tokens", 0)),
            "cache_read_tokens": _int(getattr(usage, "cache_read_input_tokens", 0)),
            "web_search_requests": _int(
                getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0)),
        }
        if not any(counts.values()):
            return
        model_name = str(model or getattr(resp, "model", "") or "unknown")
        key = (datetime.now(timezone.utc).date().isoformat(), _job_name(), task, model_name)
        row = _buffer.setdefault(key, dict(calls=0, cost_usd=0.0, **dict.fromkeys(counts, 0)))
        row["calls"] += 1
        for k, v in counts.items():
            row[k] += v
        row["cost_usd"] += estimate_cost(model_name, **counts)
        _register_atexit()
    except Exception as e:                      # 計測の失敗で本処理を止めない
        print(f"[api_usage] ⚠ 記録に失敗: {e}")


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(flush)
        _atexit_registered = True


def pending() -> list[dict]:
    """未送信の集計行。テストと、フラッシュ前に内訳を見たいとき用。"""
    return [
        {"usage_date": d, "job": j, "task": t, "model": m, **vals}
        for (d, j, t, m), vals in sorted(_buffer.items())
    ]


def flush() -> bool:
    """貯めた集計をSupabaseへ追記する。書けたらTrue。バッファは必ず空にする。

    追記専用（同じキーの行が複数あってもレポート側でSUMする）。上書きにすると、
    毎時ジョブとバックフィルが同時に走ったときに片方の消費が消える。
    """
    rows = pending()
    _buffer.clear()
    if not rows:
        return True
    for r in rows:
        r["cost_usd"] = round(r["cost_usd"], 6)
    try:
        from lib import supabase_client as sb
        return sb.upsert(_TABLE, rows)
    except Exception as e:
        print(f"[api_usage] ⚠ 保存に失敗: {e}")
        return False


def reset() -> None:
    """バッファを捨てる（テスト用）。"""
    _buffer.clear()
