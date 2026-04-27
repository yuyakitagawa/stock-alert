import requests
import numpy as np
import os
import glob
import smtplib
import joblib
import pandas as pd
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NET_SELL_THRESHOLD = -5   # ネットスコア（上昇-下落）がこれ未満で売り検討

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _get_dividend_yield(code):
    """Yahoo Finance quoteSummaryから配当利回り(%)を取得"""
    ticker = f"{code}.T"
    url = (
        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        f"?modules=summaryDetail"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        result = data.get("quoteSummary", {}).get("result", [])
        if not result:
            return None
        sd = result[0].get("summaryDetail", {})
        # dividendYield → trailingAnnualDividendYield の順で試す
        for key in ("dividendYield", "trailingAnnualDividendYield"):
            dy = sd.get(key, {})
            raw = dy.get("raw") if isinstance(dy, dict) else None
            if raw:
                return round(raw * 100, 2)
        return None
    except Exception:
        return None


def load_held_stocks():
    """Google SheetsまたはCSVからチェック銘柄を読み込む"""
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/stock-alert"))
        from sheets_helper import load_watch_list
        return load_watch_list()
    except Exception as e:
        print(f"[WARN] sheets_helper失敗、CSVで代替: {e}")
        csv_path = os.path.expanduser("~/stock-alert/watch_list.csv")
        if not os.path.exists(csv_path):
            print("ERROR: watch_list.csvが見つかりません")
            return {}
        df = pd.read_csv(csv_path, dtype=str)
        return dict(zip(df["コード"].str.strip(), df["銘柄名"].str.strip()))


def get_prices(code, days=400):
    ticker = f"{code}.T"
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = (
            result[0].get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose", [])
        )
        if not timestamps or not closes:
            return None
        df = pd.DataFrame(
            {"Close": closes},
            index=pd.to_datetime(timestamps, unit="s", utc=True)
        )
        return df.dropna()
    except Exception:
        return None


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


SEQ_DAYS = 60  # 生リターン系列の日数

def extract_features(p):
    if len(p) < 91:
        return None
    current = p[-1]
    if current == 0:
        return None
    ret5  = (current - p[-6])  / p[-6]  if len(p) >= 6  else 0
    ret20 = (current - p[-21]) / p[-21] if len(p) >= 21 else 0
    ret60 = (current - p[-61]) / p[-61] if len(p) >= 61 else 0
    ret90 = (current - p[-91]) / p[-91]
    ma5  = p[-5:].mean()
    ma25 = p[-25:].mean() if len(p) >= 25 else p.mean()
    ma75 = p[-75:].mean() if len(p) >= 75 else p.mean()
    ma5_25  = ma5  / ma25 - 1 if ma25 > 0 else 0
    ma25_75 = ma25 / ma75 - 1 if ma75 > 0 else 0
    rsi = calc_rsi(p, period=14)
    dr20 = np.diff(p[-21:]) / p[-21:-1] if len(p) >= 21 else np.array([0])
    vol20 = dr20.std() * np.sqrt(252) * 100
    dr60 = np.diff(p[-61:]) / p[-61:-1] if len(p) >= 61 else np.array([0])
    vol60 = dr60.std() * np.sqrt(252) * 100
    week52 = p[-252:] if len(p) >= 252 else p
    hi, lo = week52.max(), week52.min()
    pos52 = (current - lo) / (hi - lo) if hi > lo else 0.5
    # 過去60日の日次リターン系列
    if len(p) >= SEQ_DAYS + 1:
        seq = np.diff(p[-(SEQ_DAYS + 1):]) / p[-(SEQ_DAYS + 1):-1]
        seq = np.clip(seq, -0.2, 0.2).tolist()
    else:
        seq = [0.0] * SEQ_DAYS
    return [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52] + seq




def load_top_ranking(n=10):
    """最新のランキングCSVから上位N銘柄を読み込む"""
    files = glob.glob(os.path.expanduser("~/stock-alert/ranking_*.csv"))
    if not files:
        return None
    df = pd.read_csv(max(files, key=os.path.getmtime))
    return df.head(n)


def get_judgment(net):
    if net >= 15:
        return "🟢強気買い", "#1a7a1a"
    elif net >= 5:
        return "🔵やや強気", "#1a4a8a"
    elif net >= -5:
        return "🟡中立", "#7a6a00"
    elif net >= -15:
        return "🟠やや弱気", "#b05000"
    else:
        return "🔴売り検討", "#c0392b"


def build_earnings_section(results, earnings_dict):
    """決算サマリーセクションのHTML生成"""
    if not earnings_dict:
        return ""
    rows_html = ""
    for r in results:
        summary = earnings_dict.get(r["code"])
        if not summary:
            continue
        net = r.get("net", r["prob"])
        net_color = "#27ae60" if net >= 5 else ("#c0392b" if net < -5 else "#888")
        rows_html += (
            f"<tr>"
            f"<td><b>{r['name']}</b></td>"
            f"<td style='text-align:center'>{r['code']}</td>"
            f"<td style='color:{net_color};text-align:center'>{net:+.1f}%</td>"
            f"<td style='color:#444;font-size:13px'>{summary}</td>"
            f"</tr>"
        )
    if not rows_html:
        return ""
    return f"""
    <h2>📋 決算サマリー（AI要約）</h2>
    <p style='color:#666;font-size:13px'>株探の直近業績をClaude AIが要約しました。</p>
    <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
    <tr style='background:#e8f4e8'><th>銘柄名</th><th>コード</th><th>ネット</th><th>決算サマリー</th></tr>
    {rows_html}
    </table>
    """


def build_html(results, today, earnings_dict=None):
    sell = [r for r in results if r["signal"] == "sell"]
    hold = [r for r in results if r["signal"] == "hold"]

    def make_row(r, highlight=False):
        drop_str = f"{r['drop_prob']:.1f}%" if r.get("drop_prob") is not None else "-"
        drop_color = "#c0392b" if r.get("drop_prob") is not None and r["drop_prob"] >= 40 else "#555"
        net = r.get("net", r["prob"])
        judgment, j_color = get_judgment(net)
        vol = r.get("vol")
        vol_label = r.get("vol_label", "")
        recommend = r.get("recommend", "")
        vol_color = "#c0392b" if vol and vol >= 60 else ("#b05000" if vol and vol >= 40 else "#555")
        bg = "background:#fff3cd;" if highlight else ""
        return (
            f"<tr style='{bg}'>"
            f"<td>{r['name']}</td>"
            f"<td style='text-align:center'>{r['code']}</td>"
            f"<td style='text-align:right'>¥{r['close']:,.0f}</td>"
            f"<td style='text-align:center;color:#2980b9'>{r['prob']:.1f}%</td>"
            f"<td style='text-align:center;color:{drop_color}'>{drop_str}</td>"
            f"<td style='text-align:center;font-weight:bold'>{net:+.1f}%</td>"
            f"<td style='text-align:center;color:{j_color}'><b>{judgment}</b></td>"
            f"<td style='text-align:center;color:{vol_color}'>{f'{vol:.1f}% {vol_label}' if vol else '-'}</td>"
            f"<td style='text-align:center'><b>{recommend}</b></td>"
            f"</tr>"
        )

    rows_sell = "".join(make_row(r, highlight=True) for r in sell)
    rows_hold = "".join(make_row(r) for r in hold)

    table_header = (
        "<tr style='background:#eee'>"
        "<th>銘柄名</th><th>コード</th><th>直近株価</th>"
        "<th>上昇確率</th><th>下落確率</th><th>ネット</th><th>判定</th><th>ボラ</th><th>推奨</th>"
        "</tr>"
    )

    sell_section = ""
    if sell:
        sell_section = f"""
        <h2 style='color:#c0392b'>⚠️ 要注意シグナル（{len(sell)}銘柄）</h2>
        <p>以下のチェック銘柄はネットスコア（上昇確率-下落確率）がマイナスです。ニュースや決算を確認してください。</p>
        <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
        <tr style='background:#f8d7da'><th>銘柄名</th><th>コード</th><th>直近株価</th><th>上昇確率</th><th>下落確率</th><th>ネット</th><th>判定</th><th>ボラ</th><th>推奨</th></tr>
        {rows_sell}
        </table>
        """
    else:
        sell_section = "<h2 style='color:#27ae60'>✅ 要注意シグナルなし</h2><p>今週は全銘柄がポジティブ/中立判定です。</p>"

    x_section = ""
    earnings_section = build_earnings_section(results, earnings_dict or {})

    # 注目株ランキングセクション
    ranking = load_top_ranking(10)
    ranking_rows = ""
    if ranking is not None:
        for _, row in ranking.iterrows():
            net = row.get("ネット(%)", row["上昇確率(%)"])
            drop = row.get("下落確率(%)", "-")
            vol = row.get("ボラ(%)", "-")
            vol_label = row.get("ボラ水準", "")
            recommend = row.get("推奨", "")
            judgment = row.get("判定", "")
            drop_color = "#c0392b" if isinstance(drop, float) and drop >= 40 else "#555"
            net_color = "#27ae60" if net >= 5 else ("#c0392b" if net < -5 else "#888")
            ranking_rows += (
                f"<tr>"
                f"<td style='text-align:center'>{int(row['順位'])}</td>"
                f"<td>{row['銘柄名']}</td>"
                f"<td style='text-align:center'>{int(row['銘柄コード'])}</td>"
                f"<td style='text-align:right'>¥{int(row['直近株価(円)']):,}</td>"
                f"<td style='text-align:center;color:#2980b9'>{row['上昇確率(%)']:.1f}%</td>"
                f"<td style='text-align:center;color:{drop_color}'>{f'{drop:.1f}%' if isinstance(drop, float) else drop}</td>"
                f"<td style='text-align:center;color:{net_color}'><b>{net:+.1f}%</b></td>"
                f"<td style='text-align:center'>{judgment}</td>"
                f"<td style='text-align:center'>{f'{vol:.1f}% {vol_label}' if isinstance(vol, float) else vol}</td>"
                f"<td style='text-align:center'><b>{recommend}</b></td>"
                f"</tr>"
            )
    ranking_section = f"""
    <h2>📈 今週の注目株 Top10</h2>
    <p style='color:#666;font-size:13px'>ネット = 上昇確率 - 下落確率 ／ ボラ = 20日年率換算ボラティリティ</p>
    <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
    <tr style='background:#def'><th>順位</th><th>銘柄名</th><th>コード</th><th>株価</th><th>上昇確率</th><th>下落確率</th><th>ネット</th><th>判定</th><th>ボラ</th><th>推奨</th></tr>
    {ranking_rows}
    </table>
    """ if ranking_rows else ""

    return f"""
    <html><body style='font-family:sans-serif;max-width:640px;margin:0 auto;padding:20px'>
    <h1>📊 週次レポート</h1>
    <p style='color:#666'>{today}</p>
    {x_section}
    <hr>
    {ranking_section}
    <hr>
    {sell_section}
    {earnings_section}
    <h2>📋 全チェック銘柄サマリー</h2>
    <p style='color:#666;font-size:13px'>ネット = 上昇確率 - 下落確率 ／ ボラ = 20日年率換算ボラティリティ（🟢低&lt;20% 🟡中&lt;40% 🟠高&lt;60% 🔴超高）</p>
    <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
    {table_header}
    {rows_sell}{rows_hold}
    </table>
    <hr>
    <p style='color:#999;font-size:12px'>このメールは過去データに基づく参考情報です。投資判断はご自身の責任で行ってください。</p>
    </body></html>
    """


def send_email(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())


def main():
    today = datetime.now().strftime("%Y年%m月%d日")
    print("=" * 50)
    print(f"チェック銘柄アラート  {today}")
    print("=" * 50)

    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("ERROR: .envにGMAIL_ADDRESSとGMAIL_APP_PASSWORDを設定してください")
        return

    held_stocks = load_held_stocks()
    if not held_stocks:
        return
    print(f"チェック銘柄: {len(held_stocks)}銘柄")

    rise_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    drop_path = os.path.expanduser("~/stock-alert/rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pklが見つかりません。先にrf_predict.pyを実行してください")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None

    results = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        feat = extract_features(prices["Close"].values)
        if feat is None:
            continue
        rise_prob = float(rise_model.predict_proba([feat])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat])[0][1]) * 100 if drop_model else None
        close = float(prices["Close"].iloc[-1])
        net = rise_prob - drop_prob if drop_prob is not None else rise_prob
        signal = "sell" if net < NET_SELL_THRESHOLD else "hold"
        judgment, _ = get_judgment(net)
        # ボラティリティ（feat[7] = vol20, 年率換算%）
        vol = round(feat[7], 1)
        if vol < 20:
            vol_label = "🟢低"
        elif vol < 40:
            vol_label = "🟡中"
        elif vol < 60:
            vol_label = "🟠高"
        else:
            vol_label = "🔴超高"
        # 総合推奨
        if net >= 5 and vol < 40:
            recommend = "✅ 推奨"
        elif net >= 5 and vol >= 40:
            recommend = "⚡ 高リスク"
        elif net < -5 and vol >= 40:
            recommend = "🚫 避ける"
        else:
            recommend = "⏳ 様子見"
        print(f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}% 下落{drop_prob:5.1f}% ネット{net:+.1f}% ボラ{vol:.1f}%{vol_label} {recommend}" if drop_prob else f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}%")
        results.append({"code": code, "name": name, "prob": rise_prob, "drop_prob": drop_prob, "net": net, "close": close, "signal": signal, "vol": vol, "vol_label": vol_label, "recommend": recommend})

    # 決算サマリー取得
    earnings_dict = {}
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/stock-alert"))
        from earnings_summary import get_earnings_summary
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            print("\n決算サマリー取得中...")
            for r in results:
                res = get_earnings_summary(r["code"], r["name"])
                if res and res.get("summary"):
                    earnings_dict[r["code"]] = res["summary"]
                    print(f"  ✅ {r['name']}: 取得")
                else:
                    print(f"  ⚠️  {r['name']}: データなし")
        else:
            print("\n[INFO] ANTHROPIC_API_KEY未設定のため決算サマリーをスキップ")
    except Exception as e:
        print(f"\n[WARN] 決算サマリーエラー: {e}")

    sell_count = sum(1 for r in results if r["signal"] == "sell")
    subject = f"【週次レポート】{today} 注目株Top10 / 要注意{sell_count}銘柄"
    html = build_html(results, today, earnings_dict)

    print(f"\nGmail送信中 → {GMAIL_ADDRESS}")
    try:
        send_email(subject, html)
        print("送信完了 ✅")
    except Exception as e:
        print(f"送信失敗: {e}")


if __name__ == "__main__":
    main()
