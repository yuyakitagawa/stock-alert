"""
マーケットタイミング＆ウォッチリストアラート（Step 5b）

毎日のパイプライン終了後に実行:
1. 全銘柄の平均 drop_prob → N225 投資/キャッシュ判定
2. ウォッチ銘柄の dp 閾値アラート（SBI HD 等）

依存: rank_stocks.py → export_to_web.py 実行後
"""
import json
import os
import smtplib
import sys
from datetime import date
from email.mime.text import MIMEText

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
ALERT_TO = os.getenv("ALERT_TO", "")

WATCHLIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "dp_watchlist.json",
)

DEFAULT_WATCHLIST = [
    {"code": "8473", "name": "SBI HD", "dp_threshold": 8.0},
]

MARKET_DP_CASH_THRESHOLD = 15.0


def load_watchlist() -> list[dict]:
    if os.path.exists(WATCHLIST_PATH):
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_WATCHLIST, f, ensure_ascii=False, indent=2)
    return DEFAULT_WATCHLIST


def get_today_rankings(today_str: str) -> list[dict]:
    """ローカルDB から当日ランキングを取得"""
    from lib.db import init_db
    import sqlite3
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stock_alert.db",
    )
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT code, name, close, drop_prob FROM daily_ranking WHERE date = ?",
        (today_str,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def calc_market_dp(rankings: list[dict]) -> float | None:
    dps = [r["drop_prob"] for r in rankings if r.get("drop_prob") is not None]
    if not dps:
        return None
    return sum(dps) / len(dps)


def build_alert_email(
    today_str: str,
    avg_dp: float | None,
    watchlist_alerts: list[dict],
) -> tuple[str, str] | None:
    """件名と本文を生成。通知すべき内容がなければ None。"""
    lines = []
    subject_parts = []

    # --- N225 タイミングシグナル ---
    if avg_dp is not None:
        if avg_dp >= MARKET_DP_CASH_THRESHOLD:
            signal = "🔴 キャッシュ推奨（avg_dp ≥ 15）"
            subject_parts.append("🔴N225→キャッシュ")
        else:
            signal = "🟢 N225 投資継続OK"
            subject_parts.append("🟢N225→投資継続")
        lines.append("━━ N225 タイミングシグナル ━━")
        lines.append(f"  全銘柄 平均dp: {avg_dp:.1f}%")
        lines.append(f"  判定: {signal}")
        lines.append(f"  ルール: avg_dp < {MARKET_DP_CASH_THRESHOLD} → N225 ETF 保有")
        lines.append(f"          avg_dp ≥ {MARKET_DP_CASH_THRESHOLD} → キャッシュへ退避")
        lines.append("")

    # --- ウォッチ銘柄アラート ---
    if watchlist_alerts:
        lines.append("━━ ウォッチ銘柄アラート ━━")
        for a in watchlist_alerts:
            lines.append(
                f"  🔔 {a['name']}({a['code']}): dp = {a['dp']:.1f}% "
                f"→ 閾値 {a['threshold']} を下回りました！"
            )
            lines.append(f"     株価: {a['close']:,.0f}円")
            subject_parts.append(f"🔔{a['name']} dp<{a['threshold']}")
        lines.append("")

    if not subject_parts:
        return None

    subject = f"[stock-alert] {today_str} {' / '.join(subject_parts)}"
    body = "\n".join(lines) + "\n※ このメールは stock-alert パイプラインが自動送信しています。\n"
    return subject, body


def send_email(subject: str, body: str) -> bool:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("[market_timing] GMAIL未設定。メール送信スキップ。")
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ALERT_TO
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, ALERT_TO, msg.as_string())
        print(f"[market_timing] メール送信完了: {subject}")
        return True
    except Exception as e:
        print(f"[market_timing] メール送信失敗: {e}")
        return False


def main() -> None:
    today_str = date.today().isoformat()
    print(f"[market_timing] {today_str} マーケットタイミング判定開始")

    rankings = get_today_rankings(today_str)
    if not rankings:
        print("[market_timing] 当日ランキングなし。終了。")
        return

    # 1. 市場平均dp
    avg_dp = calc_market_dp(rankings)
    if avg_dp is not None:
        status = "キャッシュ推奨" if avg_dp >= MARKET_DP_CASH_THRESHOLD else "投資継続"
        print(f"[market_timing] 市場平均dp: {avg_dp:.1f}% → {status}")

    # 2. ウォッチ銘柄チェック
    watchlist = load_watchlist()
    ranking_map = {r["code"]: r for r in rankings}
    watchlist_alerts = []
    for w in watchlist:
        r = ranking_map.get(w["code"])
        if r is None:
            print(f"[market_timing] {w['name']}({w['code']}): ランキングに未掲載")
            continue
        dp = r.get("drop_prob")
        if dp is None:
            continue
        threshold = w.get("dp_threshold", 8.0)
        if dp < threshold:
            watchlist_alerts.append({
                "code": w["code"],
                "name": w["name"],
                "dp": dp,
                "threshold": threshold,
                "close": r.get("close", 0),
            })
            print(f"[market_timing] 🔔 {w['name']}({w['code']}): dp={dp:.1f} < {threshold} → アラート!")
        else:
            print(f"[market_timing] {w['name']}({w['code']}): dp={dp:.1f} (閾値{threshold}未到達)")

    # 3. メール送信
    result = build_alert_email(today_str, avg_dp, watchlist_alerts)
    if result:
        subject, body = result
        print(f"\n{body}")
        send_email(subject, body)
    else:
        print("[market_timing] 通知すべき内容なし。")

    # 4. 結果をJSONに保存（Webアプリ連携用）
    output = {
        "date": today_str,
        "avg_dp": round(avg_dp, 1) if avg_dp is not None else None,
        "n225_signal": "cash" if avg_dp and avg_dp >= MARKET_DP_CASH_THRESHOLD else "invest",
        "watchlist_alerts": watchlist_alerts,
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
