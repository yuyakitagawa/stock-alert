"""
カタリスト速報アラート（Step 5c）

毎日のパイプライン実行後:
1. 全ユーザーのウォッチ銘柄を取得
2. ext_tdnet_disclosures からカタリスト・決算開示を検索してLINEプッシュ
3. edinet_large_holdings から大量保有報告を検索してLINEプッシュ

必要な環境変数: LINE_CHANNEL_ACCESS_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY
"""
import os
import sys
from datetime import date, timedelta

import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from web._helpers import push_line, sb_get

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

CATALYST_KEYWORDS = ["上方修正", "増配", "自社株買", "株式分割", "特別配当", "業績予想の修正", "復配", "株主優待"]
EARNINGS_KEYWORDS = ["決算短信", "四半期報告", "業績", "通期", "第1四半期", "第2四半期", "第3四半期"]


def get_all_watchlists() -> dict[str, list[dict]]:
    rows = sb_get("dp_watchlist?select=line_user_id,code,name&order=line_user_id,created_at")
    by_user: dict[str, list[dict]] = {}
    for r in rows:
        by_user.setdefault(r["line_user_id"], []).append(r)
    return by_user


def get_disclosures(codes: list[str], since: str) -> list[dict]:
    codes_param = ",".join(codes)
    return sb_get(
        f"ext_tdnet_disclosures?code=in.({codes_param})"
        f"&disclosed_at=gte.{since}"
        f"&select=code,disclosed_at,title,category,content"
        f"&order=disclosed_at.desc&limit=50"
    )


def get_large_holdings(codes: list[str], since: str) -> list[dict]:
    codes_param = ",".join(codes)
    return sb_get(
        f"edinet_large_holdings?issuer_code=in.({codes_param})"
        f"&disc_date=gte.{since}"
        f"&select=issuer_code,issuer_name,filer_name,holding_ratio,doc_description,disc_date"
        f"&order=disc_date.desc&limit=30"
    )


def get_stock_data(codes: list[str], today: str) -> dict[str, dict]:
    codes_param = ",".join(codes)
    rows = sb_get(
        f"gen_rankings?code=in.({codes_param})"
        f"&date=eq.{today}"
        f"&select=code,name,close,drop_prob,per,pbr"
    )
    return {r["code"]: r for r in rows}


def _price_str(info: dict) -> str:
    close = info.get("close")
    dp = info.get("drop_prob")
    return f"{close:,.0f}円 下落確率{dp}%" if close and dp is not None else ""


def summarize_disclosures(disclosures: list[dict], stock_info: dict) -> str:
    items = []
    for d in disclosures:
        is_catalyst = any(kw in d["title"] for kw in CATALYST_KEYWORDS)
        info = stock_info.get(d["code"], {})
        items.append(
            f"{'🔥' if is_catalyst else '📄'} {info.get('name', d['code'])}({d['code']}) {_price_str(info)}\n"
            f"  [{d['disclosed_at'][:10]}] {d['title']}\n"
            f"  {(d.get('content') or '')[:500]}"
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": (
            "以下はウォッチリスト銘柄の適時開示です。投資家向けに重要ポイントを簡潔にまとめてください。\n"
            "- 🔥はカタリスト（上方修正・増配・自社株買い等）\n"
            "- 各銘柄の開示内容と株価情報を踏まえて、買い増し・売却・様子見の観点でコメントする\n"
            "- 全体を400文字以内にまとめる\n\n"
            + "\n\n".join(items)
        )}],
    )
    return resp.content[0].text.strip()


def summarize_large_holdings(holdings: list[dict], stock_info: dict) -> str:
    items = []
    for h in holdings:
        code = h["issuer_code"]
        info = stock_info.get(code, {})
        ratio = h.get("holding_ratio")
        items.append(
            f"🏦 {info.get('name') or h.get('issuer_name') or code}({code}) {_price_str(info)}\n"
            f"  [{h['disc_date'][:10]}] {h.get('doc_description', '')}\n"
            f"  申告者: {h.get('filer_name', '不明')} 保有比率: {f'{ratio:.2f}%' if ratio is not None else '不明'}"
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": (
            "以下はウォッチリスト銘柄のEDINET大量保有報告書です。投資家向けに重要ポイントを簡潔にまとめてください。\n"
            "- 新規取得・増加は買い圧力、減少・処分は売り圧力のシグナルとなりうる\n"
            "- 機関投資家や大株主の動向として、買い増し・売却・様子見の観点でコメントする\n"
            "- 全体を300文字以内にまとめる\n\n"
            + "\n\n".join(items)
        )}],
    )
    return resp.content[0].text.strip()


def _fallback_disclosure_text(disclosures: list[dict], stock_data: dict) -> str:
    lines = ["📰 ウォッチ銘柄の開示情報\n"]
    for d in disclosures:
        info = stock_data.get(d["code"], {})
        mark = "🔥" if any(kw in d["title"] for kw in CATALYST_KEYWORDS) else "📄"
        lines.append(f"{mark} {info.get('name', d['code'])}({d['code']})\n  {d['disclosed_at'][:10]} {d['title']}")
    return "\n".join(lines)


def _fallback_holdings_text(holdings: list[dict], stock_data: dict) -> str:
    lines = ["🏦 ウォッチ銘柄の大量保有報告\n"]
    for h in holdings:
        code = h["issuer_code"]
        info = stock_data.get(code, {})
        name = info.get("name") or h.get("issuer_name") or code
        ratio = h.get("holding_ratio")
        ratio_str = f"{ratio:.2f}%" if ratio is not None else "不明"
        lines.append(f"🏦 {name}({code})\n  {h['disc_date'][:10]} {h.get('filer_name','不明')} 保有比率{ratio_str}")
    return "\n".join(lines)


DISCLAIMER = "\n\n※本情報は参考情報であり、投資判断は自己責任でお願いします"


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

    all_codes = list({item["code"] for items in watchlists.values() for item in items})
    disclosures = get_disclosures(all_codes, since)
    large_holdings = get_large_holdings(all_codes, since)
    stock_data = get_stock_data(all_codes, today)
    print(f"TDnet開示: {len(disclosures)}件 / 大量保有報告: {len(large_holdings)}件")

    for user_id, watch_items in watchlists.items():
        user_codes = {item["code"] for item in watch_items}

        # TDnet適時開示
        user_disc = [d for d in disclosures if d["code"] in user_codes]
        catalyst = [d for d in user_disc if any(kw in d["title"] for kw in CATALYST_KEYWORDS)]
        earnings = [d for d in user_disc if any(kw in d["title"] for kw in EARNINGS_KEYWORDS)]
        notify = catalyst or earnings

        if notify:
            print(f"  {user_id[:8]}...: TDnet {len(notify)}件")
            try:
                summary = summarize_disclosures(notify, stock_data)
            except Exception as e:
                print(f"  Claude要約エラー: {e}")
                summary = _fallback_disclosure_text(notify, stock_data)
            header = "🔥 ウォッチ銘柄にカタリスト発生！" if catalyst else "📰 ウォッチ銘柄の決算情報"
            push_line(user_id, f"{header}\n\n{summary}{DISCLAIMER}")

        # EDINET大量保有
        user_holdings = [h for h in large_holdings if h["issuer_code"] in user_codes]
        if user_holdings:
            print(f"  {user_id[:8]}...: 大量保有 {len(user_holdings)}件")
            try:
                holding_summary = summarize_large_holdings(user_holdings, stock_data)
            except Exception as e:
                print(f"  Claude要約エラー: {e}")
                holding_summary = _fallback_holdings_text(user_holdings, stock_data)
            push_line(user_id, f"🏦 ウォッチ銘柄に大量保有報告\n\n{holding_summary}{DISCLAIMER}")


if __name__ == "__main__":
    main()
