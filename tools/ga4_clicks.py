"""
tools/ga4_clicks.py

GA4のクリックログ（`GaClickTracker`が全ボタン・リンクに仕込んでいる`click`イベント）を
Data APIで取り出し、PDCAのCheckに使える形にして表示する（手動実行専用）。

なぜサーバーログではなくGA4か:
  `blog_crawler_log`の"Browser"判定は、直近30日206,678PVのうち86.7%が1IPで100PV超の
  機械アクセスで、残りも1IP15.3PVと人の値ではない（`tools/traffic_report.py`）。
  実測でも滞在1PVの訪問者15,092人に対し2〜10PVがわずか76人という二極分布で、
  「どのCTAが押されたか」をこのログから読むことはできない。クリックはGA4にしか無い。

出すもの（すべて前期間との差分付き。差分が無いとPDCAのCheckにならない）:
  1. ページ別のクリック数とクリック率（そのページを見た人のうち何かを押した人の割合）
  2. CTA別（ボタン文言）のクリック数 … `label`をカスタムディメンションに登録済みの場合のみ
  3. 流入元別（utm_source=line / youtube / x を識別）のセッションとエンゲージメント率

必要な設定（どちらも1回だけ。未設定なら実行時に手順を表示する）:
  - GCPプロジェクトで Google Analytics Data API を有効化
  - GA4プロパティの「プロパティのアクセス管理」に、サービスアカウントのメールアドレスを
    「閲覧者」で追加
  - .env に GA4_PROPERTY_ID=（GA4管理画面 > プロパティ設定 のプロパティID、数字9桁）

実行:
  python3 tools/ga4_clicks.py                 # 直近28日 vs その前の28日
  python3 tools/ga4_clicks.py --days 7        # 期間を変える
  python3 tools/ga4_clicks.py --limit 30      # 表示行数
"""
import argparse
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

API_ROOT = "https://analyticsdata.googleapis.com/v1beta"
SCOPE = "https://www.googleapis.com/auth/analytics.readonly"
DEFAULT_DAYS = 28
DEFAULT_LIMIT = 20
# GaClickTrackerがsendGAEventで送っているイベント名。ここを変えるならコンポーネント側も変える。
CLICK_EVENT = "click"


def credentials_path() -> str:
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcp_key.json")


def access_token() -> "str | None":
    """サービスアカウントのアクセストークン。鍵が無ければNone。"""
    path = credentials_path()
    if not os.path.exists(path):
        return None
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(path, scopes=[SCOPE])
    creds.refresh(Request())
    return creds.token


def explain_error(status: int, payload: dict, property_id: str) -> str:
    """APIの失敗を「次に何をすれば直るか」に翻訳する。GA4のエラーはメッセージが長い割に
    必要な操作（APIの有効化 / プロパティへのユーザー追加）が読み取りにくい。"""
    err = (payload.get("error") or {})
    reason = ""
    for d in err.get("details") or []:
        reason = d.get("reason") or reason
    email = ""
    try:
        with open(credentials_path()) as f:
            email = json.load(f).get("client_email", "")
    except Exception:
        pass

    if reason == "SERVICE_DISABLED":
        return ("Google Analytics Data API がGCPプロジェクトで無効です。\n"
                "  → https://console.cloud.google.com/apis/library/analyticsdata.googleapis.com "
                "で有効化してください（数分で反映）")
    if status == 403:
        return (f"プロパティ {property_id} へのアクセス権がありません。\n"
                f"  → GA4管理画面 > プロパティのアクセス管理 で `{email}` を「閲覧者」として追加してください")
    if status == 404:
        return (f"プロパティ {property_id} が見つかりません。\n"
                "  → GA4管理画面 > プロパティ設定 の「プロパティID」（数字）を GA4_PROPERTY_ID に設定してください")
    return f"HTTP {status}: {err.get('message') or json.dumps(payload)[:300]}"


def run_report(token: str, property_id: str, body: dict) -> tuple:
    """(rows, error_message)。rowsは[{dim値のtuple: [metric値...]}]ではなく生のGA4行。"""
    resp = requests.post(f"{API_ROOT}/properties/{property_id}:runReport",
                         headers={"Authorization": f"Bearer {token}"}, json=body, timeout=60)
    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": {"message": resp.text[:300]}}
        return [], explain_error(resp.status_code, payload, property_id)
    return (resp.json().get("rows") or []), ""


def parse_rows(rows: list) -> dict:
    """GA4の行を {ディメンション値(タプル): [指標(float)...]} に均す。
    ディメンションが1つのときはタプルではなく文字列をキーにする（呼び出し側の可読性のため）。"""
    out = {}
    for r in rows:
        dims = tuple(d.get("value", "") for d in r.get("dimensionValues") or [])
        vals = [float(m.get("value") or 0) for m in r.get("metricValues") or []]
        out[dims[0] if len(dims) == 1 else dims] = vals
    return out


def delta(now: float, before: float) -> str:
    """前期間との差。0→正の増加は「新規」と出す（%にすると∞になり読めない）。"""
    if before == 0:
        return "  (新規)" if now else ""
    return f"  ({(now - before) / before * 100:+.0f}%)"


def _period_body(dimensions: list, metrics: list, start: date, end: date,
                 limit: int, dimension_filter: "dict | None" = None) -> dict:
    body = {
        "dateRanges": [{"startDate": start.isoformat(), "endDate": end.isoformat()}],
        "dimensions": [{"name": d} for d in dimensions],
        "metrics": [{"name": m} for m in metrics],
        "limit": limit,
    }
    if dimension_filter:
        body["dimensionFilter"] = dimension_filter
    return body


def _click_filter() -> dict:
    return {"filter": {"fieldName": "eventName",
                       "stringFilter": {"matchType": "EXACT", "value": CLICK_EVENT}}}


def report(days: int, limit: int) -> int:
    property_id = os.getenv("GA4_PROPERTY_ID", "").strip()
    if not property_id:
        print("[ga4] GA4_PROPERTY_ID が未設定です（GA4管理画面 > プロパティ設定 の数字ID）")
        return 1
    try:
        token = access_token()
    except Exception as e:
        print(f"[ga4] サービスアカウント鍵の読み込みに失敗: {e}")
        return 1
    if not token:
        print(f"[ga4] サービスアカウント鍵が見つかりません: {credentials_path()}")
        return 1

    end = date.today() - timedelta(days=1)          # GA4は当日ぶんが確定しないので前日まで
    start = end - timedelta(days=days - 1)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    print(f"GA4 クリックログ: {start}〜{end}（比較: {prev_start}〜{prev_end}）\n")

    # 1) ページ別クリック数 と クリック率
    #    率の分母はPVではなく「そのページを見た人数」にする。1人が同じページで複数回
    #    押すのは普通なので、クリック数÷PVだと率が100%を超えて読めなくなる（実測で
    #    /ranking/sells が216.7%）。「見た人のうち何%が何かを押したか」なら必ず100%以下。
    fail = 0
    cur, err = run_report(token, property_id, _period_body(
        ["pagePath"], ["eventCount", "totalUsers"], start, end, limit, _click_filter()))
    if err:
        print(f"[ga4] 取得に失敗しました。\n  {err}")
        return 1
    prev, _ = run_report(token, property_id, _period_body(
        ["pagePath"], ["eventCount"], prev_start, prev_end, 500, _click_filter()))
    views, _ = run_report(token, property_id, _period_body(
        ["pagePath"], ["screenPageViews", "totalUsers"], start, end, 500))
    cur_p, prev_p, view_p = parse_rows(cur), parse_rows(prev), parse_rows(views)

    print(f"■ ページ別クリック（{CLICK_EVENT}イベント）")
    if not cur_p:
        print("  クリックが記録されていません（GaClickTrackerの配信 or GA4の計測を確認）")
        fail = 1
    for path, vals in sorted(cur_p.items(), key=lambda kv: -kv[1][0])[:limit]:
        clicks = vals[0]
        clicked_users = vals[1] if len(vals) > 1 else 0
        pv, page_users = (list(view_p.get(path) or [0, 0]) + [0, 0])[:2]
        rate = f"{clicked_users / page_users * 100:5.1f}%" if page_users else "   -  "
        print(f"  {clicks:7,.0f}回  押した人{rate}  (PV {pv:6,.0f} / 閲覧者 {page_users:5,.0f}人)  {path}"
              + delta(clicks, (prev_p.get(path) or [0])[0]))

    # 2) CTA別（ボタン文言）。labelはイベントパラメータなので、GA4でカスタムディメンションに
    #    登録していないと参照できない。未登録は設定漏れなので、その場で手順を出す。
    cur_l, err_l = run_report(token, property_id, _period_body(
        ["customEvent:label"], ["eventCount"], start, end, limit, _click_filter()))
    print(f"\n■ CTA別クリック（ボタン文言）")
    if err_l:
        print("  labelがカスタムディメンション未登録のため取得できません。")
        print("  → GA4管理画面 > データの表示 > カスタム定義 > カスタムディメンションを作成")
        print("     ディメンション名 label / 範囲 イベント / イベントパラメータ label")
        print("     （登録した日以降のデータのみ集計されます）")
    else:
        prev_l, _ = run_report(token, property_id, _period_body(
            ["customEvent:label"], ["eventCount"], prev_start, prev_end, 500, _click_filter()))
        prev_lp = parse_rows(prev_l)
        for label, vals in sorted(parse_rows(cur_l).items(), key=lambda kv: -kv[1][0])[:limit]:
            print(f"  {vals[0]:7,.0f}回  {label[:60]}" + delta(vals[0], (prev_lp.get(label) or [0])[0]))

    # 3) 流入元別。LINE/YouTube/XのUTMがどれだけ人を連れてきているかはここでしか分からない。
    cur_s, err_s = run_report(token, property_id, _period_body(
        ["sessionSource", "sessionMedium"], ["sessions", "engagedSessions"], start, end, limit))
    print(f"\n■ 流入元別セッション")
    if err_s:
        print(f"  取得に失敗: {err_s}")
    else:
        prev_s, _ = run_report(token, property_id, _period_body(
            ["sessionSource", "sessionMedium"], ["sessions", "engagedSessions"],
            prev_start, prev_end, 500))
        prev_sp = parse_rows(prev_s)
        for key, vals in sorted(parse_rows(cur_s).items(), key=lambda kv: -kv[1][0])[:limit]:
            sessions, engaged = vals[0], (vals[1] if len(vals) > 1 else 0)
            rate = f"{engaged / sessions * 100:5.1f}%" if sessions else "   -  "
            print(f"  {sessions:7,.0f}件  エンゲージ{rate}  {key[0]} / {key[1]}"
                  + delta(sessions, (prev_sp.get(key) or [0])[0]))
    return fail


def main():
    p = argparse.ArgumentParser(description="GA4のクリックログを取得して前期間と比較する")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    a = p.parse_args()
    sys.exit(report(a.days, a.limit))


if __name__ == "__main__":
    main()
