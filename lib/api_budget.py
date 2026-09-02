"""
lib/api_budget.py
Anthropic APIの月次利用上限に達したことを検知し、以降の呼び出しを打ち切る。

上限に達すると Anthropic は HTTP 400 (invalid_request_error) に
"You have reached your specified API usage limits." を載せて返す。これは
リトライで直らないため、1件目で気づいて残りをスキップするのが正しい。

放置すると何が起きるか: 2026-08-24 の EDINET Blog Hourly では、上限到達後も
記事候補ごとに事業内容・投資家プロフィール・本文の3系統を叩き続け、1回の実行で
十数回失敗した末に記事が無言で欠落していた。課金はされないが障害の発見が遅れる。

使い方:
    from lib import api_budget

    if api_budget.reached():
        return ""            # 呼ぶ前に打ち切る
    try:
        resp = client.messages.create(...)
    except Exception as e:
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
        else:
            print(f"    ⚠ 失敗: {e}")
"""

import re

# 上限超過は復旧まで毎便同じ理由で失敗するため、通知は dedupe_key で1日1通に抑える。
DEDUPE_KEY = "anthropic_usage_limit"
DAILY_DEDUPE_KEY = "anthropic_daily_cap"

SKIP_MESSAGE = (
    "    ⚠ Anthropic APIの利用上限に到達したため、以降のClaude呼び出しをスキップします"
)
DAILY_SKIP_MESSAGE = (
    "    ⚠ 本日のAPI予算に達したため、以降のClaude呼び出しをスキップします"
)

# 400のメッセージ本文に現れる目印。文言の揺れに備えて複数持つ。
_MARKERS = (
    "reached your specified api usage limits",
    "api usage limits",
    "usage limit",
    "credit balance is too low",
)

_reached = False
_notified = False
_daily_notified = False
# 日次予算で止まったか（月次上限と区別する。日次は設計どおりの停止、月次は障害）。
_daily_cap = False


def is_usage_limit_error(exc: BaseException) -> bool:
    """例外がAnthropicの利用上限エラーなら True。"""
    text = str(exc).lower()
    return any(m in text for m in _MARKERS)


def note(exc: BaseException) -> bool:
    """例外を記録する。利用上限エラーだった場合はフラグを立てて True を返す。

    以降 reached() が True を返すようになり、同一プロセス内の後続の呼び出しを
    呼ぶ前にスキップできる。初回の検知時だけLINEへ通知する（ワークフローは
    continue-on-error で緑のまま終わるため、ここで言わないと誰も気づかない）。
    """
    global _reached
    if not is_usage_limit_error(exc):
        return False
    _reached = True
    _notify_once(exc)
    return True


def regain_access_at(exc: BaseException) -> str:
    """エラー本文から復旧予定日時（"You will regain access on 2026-09-01 at 00:00 UTC."）を抜く。
    取れなければ空文字。通知に「いつ戻るか」を書くために使う。"""
    m = re.search(
        r"regain access on\s+(\d{4}-\d{2}-\d{2})(?:\s+at\s+(\d{2}:\d{2})\s*(\w+)?)?",
        str(exc), re.IGNORECASE,
    )
    if not m:
        return ""
    date_s, time_s, tz_s = m.group(1), m.group(2) or "", (m.group(3) or "").upper()
    return f"{date_s}{' ' + time_s if time_s else ''}{' ' + tz_s if tz_s else ''}"


def _build_message(exc: BaseException) -> str:
    """何が起きて、何をすれば直り、放置すると何が止まるかまで書く。

    「失敗しました」だけだと結局ログを見に行くことになる。特にこの障害は
    **クレジットを追加しても直らない**（残高不足ではなく設定した使用上限に当たっている）。
    2026-08-24は実際にチャージで復旧を試みて空振りし、上限引き上げに気づくまで時間を要した。
    """
    when = regain_access_at(exc)
    lines = [
        "上限に到達したため、ブログ記事・動画・日次レビューの生成を停止します。",
        "",
        "対処: Anthropic Consoleで月間の使用上限（usage limit）を引き上げてください。",
        "※クレジットの追加では解除されません（残高不足ではなく上限設定に当たっています）",
        "https://console.anthropic.com/settings/limits",
    ]
    if when:
        lines += ["", f"未対応の場合、自動復旧は {when} です。それまで記事は出ません。"]
    return "\n".join(lines)


def _notify_once(exc: BaseException) -> None:
    """1回だけLINE通知する。通知の失敗で本処理は止めない。

    同一プロセス内は _notified で、プロセスを跨ぐ連投は notify.push_once の
    dedupe_key で抑える。毎時のワークフローが上限超過で失敗し続けると、抑制が無ければ
    9:00〜21:00の13便ぶん同じ通知が届く。
    """
    global _notified
    if _notified:
        return
    _notified = True
    try:
        from lib import notify

        notify.error(
            "Anthropic API 利用上限",
            _build_message(exc),
            detail=str(exc),
            dedupe_key=DEDUPE_KEY,
        )
    except Exception as e:
        print(f"[api_budget] ⚠ LINE通知に失敗: {e}")


def stop_for_daily_cap(spent_usd: float, budget_usd: float) -> None:
    """当日の推定コストが日次予算に達したので、以降のClaude呼び出しを打ち切る。

    月次上限（Anthropic側の設定）に当たってから止まるのでは遅い。2026-08-23の停止は
    バックフィルが1日で月の予算を焼いたのが原因で、月次の50/80%通知が出た時には
    もう手遅れだった。1日ぶんの上限で先に止めれば、被害は翌日UTC 0時までに限定される。
    """
    global _reached, _daily_cap
    if _reached:
        return
    _reached = True
    _daily_cap = True
    _notify_daily_once(spent_usd, budget_usd)


def _notify_daily_once(spent_usd: float, budget_usd: float) -> None:
    """日次予算での打ち切りを1回だけLINEへ流す。通知の失敗で本処理は止めない。"""
    global _daily_notified
    if _daily_notified:
        return
    _daily_notified = True
    try:
        from lib import notify

        notify.error(
            "Anthropic API 日次予算",
            "\n".join([
                f"本日の推定コストが日次予算に達したため、以降の生成を止めました"
                f"（${spent_usd:.2f} / ${budget_usd:.2f}）。",
                "",
                "記事・動画はUTC 0時（JST 9時）で自動的に再開します。",
                "想定外に増えている場合はバックフィルの走らせすぎを疑ってください",
                "（内訳: python3 tools/api_usage_report.py --days 3 --by task）。",
                "予算を変えるには環境変数 ANTHROPIC_DAILY_BUDGET_USD。",
            ]),
            dedupe_key=DAILY_DEDUPE_KEY,
        )
    except Exception as e:
        print(f"[api_budget] ⚠ LINE通知に失敗: {e}")


def reached() -> bool:
    """このプロセスで既に利用上限（月次上限 or 日次予算）に達しているなら True。"""
    return _reached


def daily_cap_reached() -> bool:
    """止まった理由が日次予算（stop_for_daily_cap）なら True。月次上限（note）なら False。

    記事生成の台帳（lib/publish_ledger）がこの区別で終了コードを変える。日次予算は
    「今日はここまで」という設計どおりの停止なのでワークフローを赤くしない。月次上限は
    復旧まで記事が1本も出ない障害なので赤くする。"""
    return _daily_cap


def reset() -> None:
    """フラグを戻す（テスト用）。"""
    global _reached, _notified, _daily_notified, _daily_cap
    _reached = False
    _notified = False
    _daily_notified = False
    _daily_cap = False
