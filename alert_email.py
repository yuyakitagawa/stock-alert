import requests
import numpy as np
import os
import smtplib
import joblib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SELL_SIGNAL_PROB = 0.4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def load_held_stocks():
    """環境変数から保有銘柄を読み込む"""
    raw = os.getenv("HELD_STOCKS", "")
    if not raw:
        print("ERROR: .envにHELD_STOCKSが設定されていません")
        return {}
    stocks = {}
    for item in raw.split(","):
        item = item.strip()
        if ":" in item:
            code, name = item.split(":", 1)
            stocks[code.strip()] = name.strip()
    return stocks


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
        import pandas as pd
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
    return [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52]


def build_html(results, today):
    sell = [r for r in results if r["signal"] == "sell"]
    hold = [r for r in results if r["signal"] == "hold"]

    rows_sell = ""
    for r in sell:
        rows_sell += (
            f"<tr style='background:#fff3cd'>"
            f"<td>⚠️ {r['name']}</td>"
            f"<td style='text-align:center'>{r['code']}</td>"
            f"<td style='text-align:right'>¥{r['close']:,.0f}</td>"
            f"<td style='text-align:center;color:#c0392b'><b>{r['prob']:.1f}%</b></td>"
            f"</tr>"
        )

    rows_hold = ""
    for r in hold:
        rows_hold += (
            f"<tr>"
            f"<td>✅ {r['name']}</td>"
            f"<td style='text-align:center'>{r['code']}</td>"
            f"<td style='text-align:right'>¥{r['close']:,.0f}</td>"
            f"<td style='text-align:center;color:#27ae60'>{r['prob']:.1f}%</td>"
            f"</tr>"
        )

    sell_section = ""
    if sell:
        sell_section = f"""
        <h2 style='color:#c0392b'>⚠️ 売り検討シグナル（{len(sell)}銘柄）</h2>
        <p>以下のチェック銘柄は3ヶ月後に+15%以上上昇する確率が40%未満です。ニュースや決算を確認してください。</p>
        <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
        <tr style='background:#f8d7da'><th>銘柄名</th><th>コード</th><th>直近株価</th><th>上昇確率</th></tr>
        {rows_sell}
        </table>
        """
    else:
        sell_section = "<h2 style='color:#27ae60'>✅ 売りシグナルなし</h2><p>今週は全銘柄が保持判定です。</p>"

    return f"""
    <html><body style='font-family:sans-serif;max-width:600px;margin:0 auto;padding:20px'>
    <h1>📊 チェック銘柄アラート</h1>
    <p style='color:#666'>{today}</p>
    {sell_section}
    <h2>📋 全チェック銘柄サマリー</h2>
    <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;width:100%'>
    <tr style='background:#eee'><th>銘柄名</th><th>コード</th><th>直近株価</th><th>上昇確率</th></tr>
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

    model_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    if not os.path.exists(model_path):
        print("ERROR: rf_model.pklが見つかりません。先にrf_predict.pyを実行してください")
        return
    model = joblib.load(model_path)

    results = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        feat = extract_features(prices["Close"].values)
        if feat is None:
            continue
        prob = float(model.predict_proba([feat])[0][1]) * 100
        close = float(prices["Close"].iloc[-1])
        signal = "sell" if prob < SELL_SIGNAL_PROB * 100 else "hold"
        mark = "⚠️  売り検討" if signal == "sell" else "✅ 保持"
        print(f"  {mark}  {name}({code}): 上昇確率 {prob:.1f}%")
        results.append({"code": code, "name": name, "prob": prob, "close": close, "signal": signal})

    sell_count = sum(1 for r in results if r["signal"] == "sell")
    subject = f"【チェック銘柄アラート】{today} 売りシグナル{sell_count}銘柄"
    html = build_html(results, today)

    print(f"\nGmail送信中 → {GMAIL_ADDRESS}")
    try:
        send_email(subject, html)
        print("送信完了 ✅")
    except Exception as e:
        print(f"送信失敗: {e}")


if __name__ == "__main__":
    main()