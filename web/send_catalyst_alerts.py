"""
カタリスト速報アラート（Step 5c）

毎日のパイプライン実行後:
1. 全ユーザーのウォッチ銘柄を取得
2. 当日の ext_tdnet_disclosures から開示を検索
3. カタリスト（上方修正・増配・自社株買い等）があればClaudeで要約してLINEプッシュ
4. カタリスト以外の決算も要約してプッシュ（決算リリースのみ）

必要な環境変数: LINE_CHANNEL_ACCESS_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY
"""
import json
import os
import sys
from datetime import date, timedelta

import anthropic
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

CATALYST_KEYWORDS = ["上方修正", "増配", "自社株買", "株式分割", "特別配当", "業績予想の修正", "復配", "株主優待"]
EARNINGS_KEYWORDS = ["決算短信", "四半期報告", "業績", "通期", "第1四半期", "第2四半期", "第3四半期"]


def sb_get(path: str) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def push_line(user_id: str, text: str) -> None:
    text = text[:5000]
    resp = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        },
        json={"to": user_id, "messages": [{"type": "text", "text": text}]},
        timeout=30,
    )
    if not resp.ok:
        print(f"LINE push failed for {user_id}: {resp.status_code} {resp.text}")


def get_all_watchlists() -> dict[str, list[dict]]:
    rows = sb_get(
        "dp_watchlist?select=line_user_id,code,name&order=line_user_id,created_at"
    )
    by_user: dict[str, list[dict]] = {}
    for r in rows:
        uid = r["line_user_id"]
        by_user.setdefault(uid, []).append(r)
    return by_user


def get_disclosures(codes: list[str], since: str) -> list[dict]:
    codes_param = ",".join(codes)
    return sb_get(
        f"ext_tdnet_disclosures?code=in.({codes_param})"
        f"&disclosed_at=gte.{since}"
        f"&select=code,disclosed_at,title,category,content"
        f"&order=disclosed_at.desc&limit=50"
    )


def get_stock_data(codes: list[str], today: str) -> dict[str, dict]:
    codes_param = ",".join(codes)
    rows = sb_get(
        f"gen_rankings?code=in.({codes_param})"
        f"&date=eq.{today}"
        f"&select=code,name,close,drop_prob,per,pbr"
    )
    return {r["code"]: r for r in rows}


def summarize_with_claude(disclosures: list[dict], stock_info: dict) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    items = []
    for d in disclosures:
        is_catalyst = any(kw in d["title"] for kw in CATALYST_KEYWORDS)
        mark = "🔥" if is_catalyst else "📄"
        code = d["code"]
        info = stock_info.get(code, {})
        name = info.get("name", code)
        close = info.get("close")
        dp = info.get("drop_prob")
        price_str = f"{close:,.0f}円 下落確率{dp}%" if close and dp is not None else ""
        content_snippet = (d.get("content") or "")[:500]
        items.append(
            f"{mark} {name}({code}) {price_str}\n"
            f"  [{d['disclosed_at'][:10]}] {d['title']}\n"
            f"  {content_snippet}"
        )

    prompt = (
        "以下はウォッチリスト銘柄の適時開示です。投資家向けに重要ポイントを簡潔にまとめてください。\n"
        "- 🔥はカタリスト（上方修正・増配・自社株買い等）\n"
        "- 各銘柄の開示内容と株価情報を踏まえて、買い増し・売却・様子見の観点でコメントする\n"
        "- 全体を400文字以内にまとめる\n\n"
        + "\n\n".join(items)
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def main() -> None:
    if not all([LINE_CHANNEL_ACCESS_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY]):
        print("必要な環境変数が不足しています。スキップします。")
        return

    today = date.today().isoformat()
    since = (date.today() - timedelta(days=1)).isoformat()

    watchlists = get_all_watchlists()
    if not watchlists:
        print("ウォッチリストが空です。")
        return

    print(f"対象ユーザー数: {len(watchlists)}")

    # 全ユニークコードを一括取得
    all_codes = list({code for items in watchlists.values() for item in items for code in [item["code"]]})
    disclosures = get_disclosures(all_codes, since)
    print(f"開示件数 (直近1日): {len(disclosures)}")

    if not disclosures:
        print("開示なし。終了します。")
        return

    stock_data = get_stock_data(all_codes, today)

    # コード → 開示リスト
    by_code: dict[str, list[dict]] = {}
    for d in disclosures:
        by_code.setdefault(d["code"], []).append(d)

    # ユーザーごとに通知
    for user_id, watch_items in watchlists.items():
        user_codes = {item["code"] for item in watch_items}
        user_disclosures = [d for d in disclosures if d["code"] in user_codes]

        if not user_disclosures:
            continue

        # カタリストのみ or 決算短信を含む場合に通知
        catalyst_disclosures = [d for d in user_disclosures if any(kw in d["title"] for kw in CATALYST_KEYWORDS)]
        earnings_disclosures = [d for d in user_disclosures if any(kw in d["title"] for kw in EARNINGS_KEYWORDS)]

        notify_disclosures = catalyst_disclosures or earnings_disclosures
        if not notify_disclosures:
            # カタリスト・決算でない開示のみ → スキップ
            continue

        print(f"  → {user_id}: {len(notify_disclosures)}件の開示")

        # AI要約
        try:
            summary = summarize_with_claude(notify_disclosures, stock_data)
        except Exception as e:
            print(f"Claude要約エラー: {e}")
            # フォールバック: テキスト形式
            lines = ["📰 ウォッチ銘柄の開示情報\n"]
            for d in notify_disclosures:
                code = d["code"]
                info = stock_data.get(code, {})
                name = info.get("name", code)
                is_catalyst = any(kw in d["title"] for kw in CATALYST_KEYWORDS)
                mark = "🔥" if is_catalyst else "📄"
                lines.append(f"{mark} {name}({code})\n  {d['disclosed_at'][:10]} {d['title']}")
            summary = "\n".join(lines)

        catalyst_count = len(catalyst_disclosures)
        header = "🔥 ウォッチ銘柄にカタリスト発生！" if catalyst_count > 0 else "📰 ウォッチ銘柄の決算情報"
        message = f"{header}\n\n{summary}\n\n※本情報は参考情報であり、投資判断は自己責任でお願いします"
        push_line(user_id, message)
        print(f"  → 送信完了: {user_id}")


if __name__ == "__main__":
    main()
