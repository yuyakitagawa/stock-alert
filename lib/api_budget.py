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

SKIP_MESSAGE = (
    "    ⚠ Anthropic APIの利用上限に到達したため、以降のClaude呼び出しをスキップします"
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


def _notify_once(exc: BaseException) -> None:
    """同一プロセスで1回だけLINE通知する。通知の失敗で本処理は止めない。"""
    global _notified
    if _notified:
        return
    _notified = True
    try:
        from lib import notify

        notify.error(
            "Anthropic API 利用上限",
            "上限に到達したため、ブログ記事・動画・日次レビューの生成を停止します。",
            detail=str(exc),
        )
    except Exception as e:
        print(f"[api_budget] ⚠ LINE通知に失敗: {e}")


def reached() -> bool:
    """このプロセスで既に利用上限に到達しているなら True。"""
    return _reached


def reset() -> None:
    """フラグを戻す（テスト用）。"""
    global _reached, _notified
    _reached = False
    _notified = False
