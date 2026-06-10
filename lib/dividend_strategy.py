"""
配当落ち後戻し買い戦略 — 共通ヘルパー

シミュレーション結果に基づくルール:
  - 株主優待あり（_fund_cache.json の yutai_month を使用）
  - 配当利回り >= 1.5%（_div_cache.json）
  - 権利落ち日から5〜14営業日後がエントリー窓
  - 回避月: 落ち月が6・8・9月（6月は5月落ち、8月は7月落ちに対応）
  - 5月落ちは中利回り(1.5-3%)のみ推奨
"""
import os, sys, json, sqlite3, calendar
import pandas as pd
from datetime import date, timedelta

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH       = os.path.join(BASE_DIR, "stock_alert.db")
DIV_CACHE     = os.path.join(BASE_DIR, "tools", "_div_cache.json")
# スプシから取得した優待月のローカルキャッシュ（daily refresh）
YUTAI_CACHE   = os.path.join(BASE_DIR, "tools", "_yutai_months.json")

# 避けるべき落ち月（シミュレーション結果より）
AVOID_EX_MONTHS = {6, 8, 9}
# エントリー窓: 権利落ち後 N〜M 営業日（近似: カレンダー日 7〜20日後）
ENTRY_MIN_DAYS = 7
ENTRY_MAX_DAYS = 20
# 利回り閾値
MIN_YIELD = 3.0


def _load_biz_days() -> list:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT DISTINCT date FROM price_cache WHERE code='7203' ORDER BY date", conn
    )
    conn.close()
    return pd.to_datetime(df["date"]).dt.date.tolist()


def _calc_ex_date(record_month: int, year: int, biz_days: list) -> date | None:
    """権利確定月の最終営業日の前営業日（≒配当落ち日）"""
    last_day = calendar.monthrange(year, record_month)[1]
    month_end = date(year, record_month, last_day)
    before = [d for d in biz_days if d <= month_end]
    if len(before) < 2:
        return None
    return before[-2]  # 月末最終営業日の1つ前


def _load_yutai_months() -> dict[str, list[int]]:
    """優待確定月マップを返す。ローカルキャッシュ優先、なければスプシから取得して保存。"""
    if os.path.exists(YUTAI_CACHE):
        mtime = os.path.getmtime(YUTAI_CACHE)
        if (date.today().toordinal() - date.fromtimestamp(mtime).toordinal()) == 0:
            with open(YUTAI_CACHE) as f:
                return json.load(f)

    # スプシから取得
    try:
        sys.path.insert(0, BASE_DIR)
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            os.path.join(BASE_DIR, "gcp_key.json"),
            scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(os.environ["SPREADSHEET_ID"])
        ws = sh.worksheet("📋 銘柄ファンダメンタル")
        rows = ws.get_all_values()
        header = rows[0]
        col_code  = header.index("コード")
        col_yutai = header.index("優待確定月")
        result: dict[str, list[int]] = {}
        for row in rows[1:]:
            code = str(row[col_code]).strip()
            yutai_str = row[col_yutai].strip()
            if not yutai_str or yutai_str in ("なし", "—", "-", ""):
                continue
            months = []
            for part in yutai_str.split(","):
                part = part.strip().replace("月", "")
                if part.isdigit():
                    months.append(int(part))
            if months:
                result[code] = months
        with open(YUTAI_CACHE, "w") as f:
            json.dump(result, f)
        return result
    except Exception:
        return {}


def get_candidates(today: date) -> list[dict]:
    """
    今日がエントリー窓内の戻し買い候補銘柄を返す。
    戻り値: [{"code", "yutai_month", "ex_date", "days_since_ex",
               "div_yield", "name", "close", "net", "drop_prob"}]
    """
    if not os.path.exists(DIV_CACHE):
        return []
    yutai_months = _load_yutai_months()   # code -> [month, ...]
    if not yutai_months:
        return []
    with open(DIV_CACHE) as f:
        div_data: dict[str, dict] = json.load(f)

    biz_days = _load_biz_days()
    today_str = today.isoformat()

    # daily_ranking から最新日の銘柄データを一括取得（当日データがない場合は最新日にフォールバック）
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    latest = conn.execute(
        "SELECT MAX(date) FROM daily_ranking WHERE date <= ?", (today_str,)
    ).fetchone()[0]
    if not latest:
        conn.close()
        return []
    ranking = {
        str(r["code"]): r
        for r in conn.execute(
            "SELECT code, name, close, net, drop_prob FROM daily_ranking WHERE date=?",
            (latest,)
        ).fetchall()
    }
    conn.close()

    candidates = []
    for code, months_list in yutai_months.items():
        # 複数月ある場合は今日に最も近い落ち日の月を使う
        yutai_month = months_list[0] if months_list else None
        if not yutai_month:
            continue

        # 配当利回り取得（過去の平均）
        divs = div_data.get(str(code), {})
        if not divs:
            continue
        recent_divs = sorted(divs.items(), reverse=True)[:4]
        if not recent_divs:
            continue
        avg_div = sum(v for _, v in recent_divs) / len(recent_divs)

        rec = ranking.get(str(code))
        if not rec:
            continue
        close = rec["close"]
        if not close or close <= 0:
            continue

        div_yield = avg_div / close * 100
        if div_yield < MIN_YIELD:
            continue

        # 今年・昨年の権利落ち日を調べ、エントリー窓に入っているか確認
        for yr in [today.year, today.year - 1]:
            ex_date = _calc_ex_date(yutai_month, yr, biz_days)
            if not ex_date:
                continue
            days_since = (today - ex_date).days
            if not (ENTRY_MIN_DAYS <= days_since <= ENTRY_MAX_DAYS):
                continue

            # 落ち月フィルタ
            ex_month = ex_date.month
            if ex_month in AVOID_EX_MONTHS:
                break

            candidates.append({
                "code":          str(code),
                "yutai_month":   yutai_month,
                "ex_date":       ex_date.isoformat(),
                "ex_month":      ex_month,
                "days_since_ex": days_since,
                "div_yield":     round(div_yield, 2),
                "name":          rec["name"],
                "close":         rec["close"],
                "net":           rec["net"],
                "drop_prob":     rec["drop_prob"],
            })
            break  # 1銘柄1エントリー

    # 利回り降順でソート
    candidates.sort(key=lambda x: x["div_yield"], reverse=True)
    return candidates
