"""
マーケットタイミング＆ウォッチリストアラート（Step 5b）

毎日のパイプライン終了後に実行:
1. 全銘柄の平均 drop_prob → N225 投資/キャッシュ判定
2. ユーザー別ウォッチ銘柄の dp 閾値アラート
3. 直近のEDINET大量保有・変更報告書（自己申告のみ除外。買い/売りは方向性を表示して両方通知。
   ウォッチ銘柄→法人/ファンド→保有比率の大きさの順に優先し、個人名の提出者は後回し）

通知先: LINE Messaging API (Push Message) × ユーザーごと
データ源: Supabase (gen_rankings, dp_watchlist, edinet_large_holdings)
必要な環境変数: LINE_CHANNEL_ACCESS_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY

依存: rank_stocks.py → export_to_web.py 実行後
"""
import os
import sys
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

MARKET_DP_CASH_THRESHOLD = 15.0
LARGE_HOLDINGS_DAYS = 3
LARGE_HOLDINGS_LIMIT = 5


def get_today_rankings(today_str: str) -> list[dict]:
    return sb.select("gen_rankings", f"date=eq.{today_str}&select=code,name,close,drop_prob")


def get_all_watchlists() -> dict[str, list[dict]]:
    rows = sb.select("dp_watchlist",
                      "select=line_user_id,code,name,dp_threshold,dp_sell_threshold&order=line_user_id,created_at")
    by_user: dict[str, list[dict]] = {}
    for r in rows:
        uid = r["line_user_id"]
        by_user.setdefault(uid, []).append(r)
    return by_user


def calc_market_dp(rankings: list[dict]) -> float | None:
    dps = [r["drop_prob"] for r in rankings if r.get("drop_prob") is not None]
    if not dps:
        return None
    return sum(dps) / len(dps)


def build_market_section(today_str: str, avg_dp: float | None) -> str:
    if avg_dp is None:
        return ""
    if avg_dp >= MARKET_DP_CASH_THRESHOLD:
        signal = "🔴 キャッシュ推奨"
    else:
        signal = "🟢 投資継続OK"
    return f"📊 N225シグナル ({today_str})\n平均下落確率: {avg_dp:.1f}% → {signal}"


def get_market_compare() -> dict | None:
    return sb.select_one("gen_market_compare", "order=date.desc")


def build_market_compare_section(compare: dict | None) -> str:
    if compare is None:
        return ""
    label = compare.get("label", "")
    reasons = compare.get("reasons") or []
    lines = [f"🌐 日経 vs S&P500: {label}"]
    if reasons:
        lines.append("・".join(reasons))
    return "\n".join(lines)


def get_recent_large_holdings(days: int = LARGE_HOLDINGS_DAYS) -> list[dict]:
    """直近days日のEDINET大量保有・変更報告書を取得し、自己申告（提出者≒対象企業）だけ除外して返す。
    譲渡/売却（sell）は「大口の動向」として買いと同様に見たいので除外しない。"""
    from lib.db import get_edinet_large_holdings_recent
    from tools.scan_large_holdings import is_noise_match, load_name_map

    name_map = load_name_map()
    rows = get_edinet_large_holdings_recent(days=days)
    out = []
    for r in rows:
        code = r.get("issuer_code")
        name = name_map.get(code, "")
        reason = is_noise_match(r.get("filer_name", ""), name, r.get("doc_description") or "")
        if reason == "self_filing":
            continue
        out.append({**r, "name": name})
    return out


def build_large_holdings_section(
    holdings: list[dict],
    watch_codes: set[str] | None = None,
    limit: int = LARGE_HOLDINGS_LIMIT,
) -> str:
    """大口保有動向セクションを整形する。全件出すと多すぎるため、
    ウォッチ銘柄を最優先・次に個人名より法人/ファンドを優先・その中で保有比率が
    大きい（動きが大きい）順に並べてlimit件に絞る。
    残りはLINEチャットで「大量保有」等と聞けば個別に答えられる（check_catalystツール）。"""
    from tools.scan_large_holdings import is_sell_disclosure, is_individual_filer

    if not holdings:
        return ""
    watch_codes = watch_codes or set()

    def sort_key(h):
        is_watch = str(h.get("issuer_code")) in watch_codes
        is_individual = is_individual_filer(h.get("filer_name", ""))
        ratio = h.get("holding_ratio") or 0
        return (not is_watch, is_individual, -abs(ratio))

    ordered = sorted(holdings, key=sort_key)

    lines = ["🏦 大口保有動向（直近・ウォッチ銘柄/大きい動き優先）:"]
    for h in ordered[:limit]:
        code = str(h.get("issuer_code", ""))
        label = f"{h['name']}({code})" if h.get("name") else code
        mark = "⭐" if code in watch_codes else ""
        doc_type = "大量保有" if h.get("doc_type_code") == "350" else "変更"
        ratio = h.get("holding_ratio")
        ratio_str = f"{ratio:.1f}%" if ratio is not None else "-"
        filer = h.get("filer_name", "")
        disc = h.get("disc_date", "")
        direction = "📉売り" if is_sell_disclosure(h.get("doc_description") or "") else "📈買い"
        lines.append(f"  {mark}{label}: {filer}が{ratio_str}保有 {direction} [{doc_type}] ({disc})")
    if len(ordered) > limit:
        lines.append(f"  ...他{len(ordered) - limit}件（LINEで「大量保有」と聞けば確認できます）")
    return "\n".join(lines)


def build_watchlist_section(
    watchlist: list[dict],
    ranking_map: dict[str, dict],
) -> str:
    lines = ["📋 ウォッチリスト状況:"]

    for w in watchlist:
        r = ranking_map.get(w["code"])
        if r is None:
            lines.append(f"  {w['name']}({w['code']}): データなし")
            continue
        dp = r.get("drop_prob")
        if dp is None:
            lines.append(f"  {w['name']}({w['code']}): データなし")
            continue
        buy_th = w.get("dp_threshold", 8.0)
        sell_th = w.get("dp_sell_threshold", 20.0)
        close = r.get("close", 0)

        if dp < buy_th:
            mark = "🔔買い時！"
        elif dp >= sell_th:
            mark = "⚠️売り検討"
        else:
            mark = ""

        lines.append(
            f"  {w['name']}({w['code']}) {close:,.0f}円"
            f" 下落確率{dp:.1f}% {mark}"
        )

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


def send_line_push(user_id: str, message: str) -> bool:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[market_timing] LINE_CHANNEL_ACCESS_TOKEN 未設定。送信スキップ。")
        return False
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": message}],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.ok:
            print(f"[market_timing] LINE送信完了 → {user_id[:8]}...")
            return True
        else:
            print(f"[market_timing] LINE送信失敗 ({user_id[:8]}...): {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[market_timing] LINE送信エラー ({user_id[:8]}...): {e}")
        return False


def main() -> None:
    today_str = date.today().isoformat()
    print(f"[market_timing] {today_str} マーケットタイミング判定開始")

    rankings = get_today_rankings(today_str)
    if not rankings:
        print("[market_timing] 当日ランキングなし。終了。")
        return

    avg_dp = calc_market_dp(rankings)
    if avg_dp is not None:
        status = "キャッシュ推奨" if avg_dp >= MARKET_DP_CASH_THRESHOLD else "投資継続"
        print(f"[market_timing] 市場平均dp: {avg_dp:.1f}% → {status}")

    market_msg = build_market_section(today_str, avg_dp)
    compare_msg = build_market_compare_section(get_market_compare())
    try:
        recent_holdings = get_recent_large_holdings()
    except Exception as e:
        print(f"[market_timing] 大口保有動向の取得に失敗（スキップ）: {e}")
        recent_holdings = []
    ranking_map = {r["code"]: r for r in rankings}

    # ユーザー別ウォッチリストを取得して個別通知
    user_watchlists = get_all_watchlists()

    if not user_watchlists:
        # ウォッチリスト登録ユーザーがいなくても、LINE_USER_IDがあれば市場シグナルだけ送信
        fallback_uid = os.getenv("LINE_USER_ID", "")
        holdings_msg = build_large_holdings_section(recent_holdings)
        fallback_parts = [p for p in (market_msg, compare_msg, holdings_msg) if p]
        if fallback_uid and fallback_parts:
            print(f"[market_timing] ウォッチリスト未登録。フォールバック送信。")
            send_line_push(fallback_uid, "\n\n".join(fallback_parts))
    else:
        for user_id, watchlist in user_watchlists.items():
            watch_msg = build_watchlist_section(watchlist, ranking_map)

            watch_codes = {str(w["code"]) for w in watchlist}
            holdings_msg = build_large_holdings_section(recent_holdings, watch_codes=watch_codes)

            parts = [p for p in (market_msg, compare_msg, holdings_msg, watch_msg) if p]

            if parts:
                message = "\n\n".join(parts)
                print(f"\n--- {user_id[:8]}... ---\n{message}\n")
                send_line_push(user_id, message)
            else:
                print(f"[market_timing] {user_id[:8]}...: 通知すべき内容なし。")

            # ウォッチリストのdp状況をログ
            for w in watchlist:
                r = ranking_map.get(w["code"])
                if r and r.get("drop_prob") is not None:
                    dp = r["drop_prob"]
                    th = w.get("dp_threshold", 8.0)
                    if dp < th:
                        print(f"[market_timing] 🔔 {w['name']}({w['code']}): dp={dp:.1f} < {th} → アラート!")
                    else:
                        print(f"[market_timing] {w['name']}({w['code']}): dp={dp:.1f} (閾値{th}未到達)")


if __name__ == "__main__":
    main()
