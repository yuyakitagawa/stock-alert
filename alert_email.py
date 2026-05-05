import numpy as np
from utils import get_prices, get_nikkei_returns, calc_rsi, extract_features, add_cs_rank_features, HEADERS, SEQ_DAYS
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
NET_SELL_THRESHOLD    = -5    # ネットスコア（上昇-下落）がこれ未満で売り検討
BEAR_MARKET_THRESHOLD = -5.0  # 日経20日リターンがこれ以下で下落相場と判定




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



def build_html(results, today, is_bear=False, nk5=None, nk20=None, nk60=None):
    CSS = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         max-width:640px;margin:0 auto;padding:16px;color:#222;background:#f5f5f5}
    .card{background:#fff;border-radius:10px;padding:16px;margin-bottom:16px;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
    h2{margin:0 0 12px;font-size:16px}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th{background:#f0f0f0;padding:8px 6px;text-align:center;font-weight:600;
       border-bottom:2px solid #ddd;white-space:nowrap}
    td{padding:8px 6px;border-bottom:1px solid #eee;vertical-align:middle}
    .net-pos{color:#0a7a0a;font-weight:700}
    .net-neg{color:#c0392b;font-weight:700}
    .net-neu{color:#888;font-weight:700}
    .rel-pos{color:#0a7a0a}
    .rel-neg{color:#c0392b}
    .badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:12px;font-weight:600}
    .badge-sell{background:#fde8e8;color:#c0392b}
    .badge-buy{background:#e8f5e9;color:#1a7a1a}
    .badge-neu{background:#f5f5f5;color:#888}
    """

    # ネットスコア → 色クラス
    def net_cls(n):
        return "net-pos" if n >= 5 else ("net-neg" if n < -5 else "net-neu")

    # 日経比 → 色クラス
    def rel_cls(r):
        if r is None: return ""
        return "rel-pos" if r >= 0 else "rel-neg"

    def rel_str(r):
        return f"{r:+.1f}%" if r is not None else "-"

    # ---- セクション1: 売り検討 ----
    sells = sorted([r for r in results if r["signal"] == "sell"], key=lambda x: x["net"])
    sell_rows = ""
    for r in sells:
        j, _ = get_judgment(r["net"])
        sell_rows += f"""
        <tr>
          <td><b>{r['name']}</b><br><span style='color:#888;font-size:12px'>{r['code']} ¥{r['close']:,.0f}</span></td>
          <td class='{net_cls(r["net"])}' style='text-align:center'>{r['net']:+.1f}%</td>
          <td class='{rel_cls(r.get("rel20"))}' style='text-align:center'>{rel_str(r.get("rel20"))}</td>
          <td style='text-align:center;color:#888;font-size:13px'>{r.get("vol","-"):.1f}%{r.get("vol_label","")}</td>
        </tr>"""
    sell_section = f"""
    <div class='card' style='border-left:4px solid #c0392b'>
      <h2>🔴 売り検討 ({len(sells)}銘柄)</h2>
      <p style='color:#666;font-size:13px;margin:0 0 10px'>ネットスコアがマイナス。ニュース・決算を確認してください。</p>
      <table><tr style='background:#fde8e8'><th>銘柄</th><th>ネット</th><th>日経比20d</th><th>ボラ</th></tr>{sell_rows}</table>
    </div>""" if sells else f"""
    <div class='card' style='border-left:4px solid #27ae60'>
      <h2>✅ 売り検討なし</h2>
      <p style='color:#666;margin:0'>全チェック銘柄がポジティブ/中立判定です。</p>
    </div>"""

    # ---- セクション2: スクリーナー上位（新規候補） ----
    held_codes = {str(r["code"]) for r in results}
    ranking = load_top_ranking(15)
    new_rows = ""
    new_count = 0
    if ranking is not None:
        for _, row in ranking.iterrows():
            if str(int(row["銘柄コード"])) in held_codes:
                continue
            if new_count >= 10:
                break
            net = row.get("ネット(%)", row["上昇確率(%)"])
            vol = row.get("ボラ(%)", 0)
            vol_label = row.get("ボラ水準", "")
            rel20_r = row.get("日経比20日(%)", None)
            rel20_val = float(rel20_r) if isinstance(rel20_r, (int, float)) and not isinstance(rel20_r, bool) else None
            per = row.get("PER"); pbr = row.get("PBR")
            fund_str = ""
            if per and str(per) not in ("nan", "None", "-"):
                fund_str += f"PER{float(per):.0f}"
            if pbr and str(pbr) not in ("nan", "None", "-"):
                fund_str += f" PBR{float(pbr):.1f}"
            new_rows += f"""
            <tr>
              <td><b>{row['銘柄名']}</b><br>
                  <span style='color:#888;font-size:12px'>{int(row['銘柄コード'])} ¥{int(row['直近株価(円)']):,}
                  {"  <span style='color:#aaa'>" + fund_str + "</span>" if fund_str else ""}</span></td>
              <td class='{net_cls(net)}' style='text-align:center'>{net:+.1f}%</td>
              <td class='{rel_cls(rel20_val)}' style='text-align:center'>{rel_str(rel20_val)}</td>
              <td style='text-align:center;color:#888;font-size:13px'>{vol:.1f}%{vol_label}</td>
            </tr>"""
            new_count += 1
    ranking_section = f"""
    <div class='card' style='border-left:4px solid #2980b9'>
      <h2>📈 新規候補 Top{new_count}（スクリーナー上位・未保有）</h2>
      <p style='color:#666;font-size:13px;margin:0 0 10px'>ネット = 上昇確率 - 下落確率 ／ 日経比20d = 過去20日の日経225比超過リターン</p>
      <table><tr style='background:#e8f0fe'><th>銘柄</th><th>ネット</th><th>日経比20d</th><th>ボラ</th></tr>{new_rows}</table>
    </div>""" if new_rows else ""

    # ---- セクション3: 全チェック銘柄一覧 ----
    sorted_results = sorted(results, key=lambda x: x["net"], reverse=True)
    all_rows = ""
    for r in sorted_results:
        j, _ = get_judgment(r["net"])
        badge_cls = "badge-sell" if r["signal"] == "sell" else ("badge-buy" if r["net"] >= 10 else "badge-neu")
        all_rows += f"""
        <tr>
          <td><b>{r['name']}</b><br>
              <span style='color:#888;font-size:12px'>{r['code']} ¥{r['close']:,.0f}</span></td>
          <td class='{net_cls(r["net"])}' style='text-align:center'>{r['net']:+.1f}%</td>
          <td style='text-align:center;font-size:13px'>{j}</td>
          <td class='{rel_cls(r.get("rel20"))}' style='text-align:center;font-size:13px'>{rel_str(r.get("rel20"))}</td>
          <td style='text-align:center;color:#888;font-size:12px'>{r.get("vol",0):.0f}%{r.get("vol_label","")}</td>
        </tr>"""

    buy_cnt  = sum(1 for r in results if r.get("recommend","").startswith(("✅","🔵","⚡")))
    neu_cnt  = len(results) - len(sells) - buy_cnt

    # ---- ヘッダー ----
    nk_str = ""
    if nk5 is not None:
        nk_str = f"日経225: 5日{nk5:+.1f}% / 20日{nk20:+.1f}% / 60日{nk60:+.1f}%"
    bear_banner = f"""
    <div style='background:#fff3cd;border:2px solid #f0ad4e;border-radius:8px;padding:12px;margin-bottom:12px'>
      <b>⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）</b><br>
      <span style='font-size:13px'>モデルスコアの信頼性が低下しています。買いシグナルは慎重に判断してください。</span>
    </div>""" if is_bear else ""

    return f"""<html><head>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <style>{CSS}</style></head>
    <body>
    <!-- ヘッダー -->
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;border-radius:10px;padding:18px;margin-bottom:16px'>
      <div style='font-size:20px;font-weight:700;margin-bottom:4px'>📊 チェック銘柄アラート</div>
      <div style='font-size:13px;color:#aaa'>{today} ／ {nk_str}</div>
    </div>
    {bear_banner}
    <!-- サマリーバー -->
    <div style='display:flex;gap:8px;margin-bottom:16px'>
      <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
        <div style='font-size:26px;font-weight:700;color:#c0392b'>{len(sells)}</div>
        <div style='font-size:12px;color:#888'>売り検討</div>
      </div>
      <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
        <div style='font-size:26px;font-weight:700;color:#0a7a0a'>{buy_cnt}</div>
        <div style='font-size:12px;color:#888'>買い可能性</div>
      </div>
      <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
        <div style='font-size:26px;font-weight:700;color:#888'>{neu_cnt}</div>
        <div style='font-size:12px;color:#888'>様子見</div>
      </div>
    </div>
    {sell_section}
    {ranking_section}
    <!-- 全銘柄一覧 -->
    <div class='card'>
      <h2>📋 チェック銘柄一覧（{len(results)}銘柄 / ネット順）</h2>
      <p style='color:#666;font-size:12px;margin:0 0 10px'>ネット=上昇確率-下落確率 ／ 日経比20d=過去20日超過リターン</p>
      <table>
        <tr><th>銘柄</th><th>ネット</th><th>判定</th><th>日経比20d</th><th>ボラ</th></tr>
        {all_rows}
      </table>
    </div>
    <p style='color:#aaa;font-size:11px;text-align:center;margin-top:8px'>
      このメールは過去データに基づく参考情報です。投資判断はご自身の責任で行ってください。
    </p>
    </body></html>"""


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

    print("日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD
    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")

    # フェーズ1: 全銘柄の特徴量を収集
    raw_data = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
        feat = extract_features(prices["Close"].values, prices["Volume"].tolist() if "Volume" in prices.columns else None, nk_rets)
        if feat is None:
            continue
        if feat[12] > 0.15 or feat[10] < -0.15:
            continue
        raw_data.append((code, name, prices, feat))

    # フェーズ2: クロスセクショナルランク特徴量を付加（34次元化）
    if not raw_data:
        print("有効銘柄なし"); return
    feats_matrix = np.array([d[3] for d in raw_data], dtype=float)
    feats_aug = add_cs_rank_features(feats_matrix)

    results = []
    for idx, (code, name, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100 if drop_model else None
        close = float(prices["Close"].iloc[-1])
        net = rise_prob - drop_prob if drop_prob is not None else rise_prob
        signal = "sell" if net < NET_SELL_THRESHOLD else "hold"
        judgment, _ = get_judgment(net)
        # ボラティリティ（feat[7] = vol20, 年率換算%）
        vol = round(feat_aug[7], 1)
        if vol < 20:
            vol_label = "🟢低"
        elif vol < 40:
            vol_label = "🟡中"
        elif vol < 60:
            vol_label = "🟠高"
        else:
            vol_label = "🔴超高"
        # 総合推奨
        if net >= 10 and vol < 40:
            recommend = "✅ 買い可能性あり"
        elif net >= 5 and vol < 40:
            recommend = "🔵 買い可能性あり"
        elif net >= 5 and vol >= 40:
            recommend = "⚡ 買い可能性あり（荒れ注意）"
        elif net < -10 and vol < 40:
            recommend = "🔴 売り可能性あり"
        elif net < -5 and vol < 40:
            recommend = "⚠️ 売り可能性あり"
        elif net < -5 and vol >= 40:
            recommend = "🌀 売り様子見"
        else:
            recommend = "⏳ 様子見"
        # 日経比リターン（相対強度）
        p = prices["Close"].values
        s5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) >= 6  else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        rel5  = round(s5  - nk5,  1) if nk5  is not None else None
        rel20 = round(s20 - nk20, 1) if nk20 is not None else None
        print(f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}% 下落{drop_prob:5.1f}% ネット{net:+.1f}% 日経比20d{rel20:+.1f}% ボラ{vol:.1f}%{vol_label}" if drop_prob and rel20 is not None else f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}%")
        results.append({"code": code, "name": name, "prob": rise_prob, "drop_prob": drop_prob, "net": net, "close": close, "signal": signal, "vol": vol, "vol_label": vol_label, "recommend": recommend, "rel5": rel5, "rel20": rel20})

    sell_count  = sum(1 for r in results if r["signal"] == "sell")
    buy_count   = sum(1 for r in results if r.get("recommend","").startswith(("✅","🔵","⚡")))
    bear_prefix = "⚠️下落相場 " if is_bear else ""
    nk_str = f"日経{nk20:+.1f}%" if nk20 is not None else ""
    subject = f"{bear_prefix}[{today[5:]}] 売り{sell_count} / 買い候補{buy_count} / {nk_str} {len(results)}銘柄"
    html = build_html(results, today, is_bear=is_bear, nk5=nk5, nk20=nk20, nk60=nk60)

    print(f"\nGmail送信中 → {GMAIL_ADDRESS}")
    try:
        send_email(subject, html)
        print("送信完了 ✅")
    except Exception as e:
        print(f"送信失敗: {e}")


if __name__ == "__main__":
    main()
