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

書き込みのついでに当月の累計を上限（MONTHLY_BUDGET_USD）と突き合わせ、50/80/100%を
超えたらLINEへ1回だけ流す。上限に「到達してから」止める api_budget.py の手前で気づくため。
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

# Anthropic側で設定している月次利用上限(USD)。超過の手前で鳴らすためだけに使う。
# 上限を変えたら環境変数 ANTHROPIC_MONTHLY_BUDGET_USD で上書きする（0なら監視しない）。
DEFAULT_MONTHLY_BUDGET_USD = 15.0
# 当月の推定コストがこの割合(%)を超えたら通知する。100%はもう止まっている可能性がある。
_ALERT_LEVELS = (50, 80, 100)

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
        ok = sb.upsert(_TABLE, rows)
    except Exception as e:
        print(f"[api_usage] ⚠ 保存に失敗: {e}")
        return False
    check_budget()
    return ok


def reset() -> None:
    """バッファを捨てる（テスト用）。"""
    _buffer.clear()


def monthly_budget_usd() -> float:
    """監視に使う月次上限(USD)。0以下なら残枠監視をしない。"""
    try:
        return float(os.getenv("ANTHROPIC_MONTHLY_BUDGET_USD") or DEFAULT_MONTHLY_BUDGET_USD)
    except ValueError:
        return DEFAULT_MONTHLY_BUDGET_USD


def alert_level(spent_usd: float, budget_usd: float) -> int:
    """超えている警告水準(%)を返す。どれも超えていなければ0。"""
    if budget_usd <= 0:
        return 0
    pct = spent_usd / budget_usd * 100
    crossed = [lv for lv in _ALERT_LEVELS if pct >= lv]
    return max(crossed) if crossed else 0


def month_usage(month: str = "") -> tuple[float, dict]:
    """当月(UTC)の推定コスト合計と、タスク別の内訳を返す。"""
    from lib import supabase_client as sb

    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    rows = sb.select(_TABLE, f"usage_date=gte.{month}-01&select=usage_date,task,cost_usd")
    by_task: dict[str, float] = {}
    for r in rows:
        if not str(r.get("usage_date", "")).startswith(month):
            continue
        by_task[r["task"]] = by_task.get(r["task"], 0.0) + float(r.get("cost_usd") or 0)
    return sum(by_task.values()), by_task


def check_budget() -> int:
    """当月の消費が上限の何割かを超えていたらLINEへ1回だけ通知し、その水準を返す。

    通知は (月, 水準) 単位で重複排除する（notify.once）。毎時のジョブが同じ警告を
    何十通も送ると、本当に見てほしい1通が埋もれる。失敗しても本処理は止めない。
    """
    budget = monthly_budget_usd()
    if budget <= 0:
        return 0
    try:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        spent, by_task = month_usage(month)
        level = alert_level(spent, budget)
        if not level:
            return 0
        top = sorted(by_task.items(), key=lambda kv: -kv[1])[:3]
        detail = " / ".join(f"{t} ${c:.2f}" for t, c in top)
        text = (f"{'🚨' if level >= 100 else '⚠️'} Anthropic API残枠\n"
                f"{month}の推定コストが上限の{level}%に達しました"
                f"（${spent:.2f} / ${budget:.2f}）\n\n内訳上位: {detail}")
        from lib import notify
        notify.once(f"api_budget_{month}_{level}", text)
        return level
    except Exception as e:                      # 監視の失敗で本処理を止めない
        print(f"[api_usage] ⚠ 残枠チェックに失敗: {e}")
        return 0
