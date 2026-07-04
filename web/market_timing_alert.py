"""
マーケットタイミング＆ウォッチリストアラート（Step 5b）

毎日のパイプライン終了後に実行:
1. 全銘柄の平均 drop_prob → N225 投資/キャッシュ判定
2. ユーザー別ウォッチ銘柄の dp 閾値アラート

通知先: LINE Messaging API (Push Message) × ユーザーごと
データ源: Supabase (gen_rankings, dp_watchlist)
必要な環境変数: LINE_CHANNEL_ACCESS_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY

依存: rank_stocks.py → export_to_web.py 実行後
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from web._helpers import push_line, sb_get

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

MARKET_DP_CASH_THRESHOLD = 15.0


def get_today_rankings(today_str: str) -> list[dict]:
    return sb_get(
        f"gen_rankings?date=eq.{today_str}"
        f"&select=code,name,close,drop_prob"
        f"&order=code.asc"
    )


def get_all_watchlists() -> dict[str, list[dict]]:
    rows = sb_get("dp_watchlist?select=line_user_id,code,name,dp_threshold,dp_sell_threshold&order=line_user_id,created_at")
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


def build_watchlist_section(
    watchlist: list[dict],
    ranking_map: dict[str, dict],
) -> tuple[str, list[dict]]:
    alerts: list[dict] = []
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
            alerts.append({
                "code": w["code"], "name": w["name"],
                "dp": dp, "threshold": buy_th, "close": close,
                "signal": "buy",
            })
        elif dp >= sell_th:
            mark = "⚠️売り検討"
            alerts.append({
                "code": w["code"], "name": w["name"],
                "dp": dp, "threshold": sell_th, "close": close,
                "signal": "sell",
            })
        else:
            mark = ""

        lines.append(
            f"  {w['name']}({w['code']}) {close:,.0f}円"
            f" 下落確率{dp:.1f}% {mark}"
        )

    if len(lines) <= 1:
        return "", alerts

    return "\n".join(lines), alerts


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
    ranking_map = {r["code"]: r for r in rankings}

    # ユーザー別ウォッチリストを取得して個別通知
    user_watchlists = get_all_watchlists()
    all_alerts: list[dict] = []

    if not user_watchlists:
        # ウォッチリスト登録ユーザーがいなくても、LINE_USER_IDがあれば市場シグナルだけ送信
        fallback_uid = os.getenv("LINE_USER_ID", "")
        if fallback_uid and market_msg:
            print(f"[market_timing] ウォッチリスト未登録。フォールバック送信。")
            push_line(fallback_uid, market_msg)
    else:
        for user_id, watchlist in user_watchlists.items():
            watch_msg, alerts = build_watchlist_section(watchlist, ranking_map)
            all_alerts.extend(alerts)

            parts = []
            if market_msg:
                parts.append(market_msg)
            if watch_msg:
                parts.append(watch_msg)

            if not parts and market_msg:
                parts.append(market_msg)

            if parts:
                message = "\n\n".join(parts)
                print(f"\n--- {user_id[:8]}... ---\n{message}\n")
                push_line(user_id, message)
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

    # 結果をJSONに保存（Webアプリ連携用）
    output = {
        "date": today_str,
        "avg_dp": round(avg_dp, 1) if avg_dp is not None else None,
        "n225_signal": "cash" if avg_dp and avg_dp >= MARKET_DP_CASH_THRESHOLD else "invest",
        "watchlist_alerts": all_alerts,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "market_timing.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[market_timing] 結果保存: {out_path}")


if __name__ == "__main__":
    main()
