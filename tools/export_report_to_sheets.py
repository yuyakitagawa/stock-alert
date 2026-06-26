"""
export_report_to_sheets.py
バックテスト成績・運用シミュレーション・PDCA履歴・特徴量一覧をGoogleスプレッドシートに書き出す。
"""
import sys, os, re, csv, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import gspread
from google.oauth2.service_account import Credentials

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
GCP_KEY_PATH   = os.path.join(BASE_DIR, "gcp_key.json")
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# ────────────────────────────────────────────────────────────
# Sheets ユーティリティ
# ────────────────────────────────────────────────────────────

def get_or_create_sheet(spreadsheet, title, rows=500, cols=30):
    try:
        ws = spreadsheet.worksheet(title)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
        return ws
    # 既存シートのサイズが不足する場合は拡張
    if ws.row_count < rows or ws.col_count < cols:
        ws.resize(rows=max(ws.row_count, rows), cols=max(ws.col_count, cols))
    return ws


def write_sheet(ws, data: list[list], header_rows=1):
    """data = [[row1], [row2], ...] をバッチで書き込み（大量行対応）。"""
    if not data:
        return
    import time as _t
    BATCH = 1000
    for i in range(0, len(data), BATCH):
        chunk = data[i:i+BATCH]
        start_row = i + 1
        ws.update(range_name=f"A{start_row}", values=chunk)
        if i + BATCH < len(data):
            _t.sleep(1.2)  # write quota 対策
    # ヘッダー行を太字・背景色
    if header_rows:
        ws.format(f"A1:{chr(64+len(data[0]))}1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.23, "green": 0.37, "blue": 0.60},
            "horizontalAlignment": "CENTER",
        })
        # ヘッダーテキストを白に
        ws.format(f"A1:{chr(64+len(data[0]))}1", {
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })


# ────────────────────────────────────────────────────────────
# 1. バックテスト成績
# ────────────────────────────────────────────────────────────

def build_backtest_data():
    from lib.db import load_market_index_data
    _nk_df = load_market_index_data("N225", days=2200)
    if _nk_df is not None and len(_nk_df) > 0:
        nk_series = _nk_df["Close"].squeeze()
    else:
        nk_series = pd.Series(dtype=float)

    def get_nk(d):
        dt = pd.Timestamp(d)
        m = nk_series.index >= dt
        if m.any(): return float(nk_series[m].iloc[0])
        return float(nk_series[nk_series.index <= dt].iloc[-1])

    files = sorted(glob.glob(f"{BASE_DIR}/simulations/backtests/rolling21d_*.csv"))
    PERIOD_LABELS = {
        "2022-06-01_2022-12-31": "利上げ局面 2022H2",
        "2023-04-01_2023-10-01": "強気相場 2023",
        "2024-01-01_2024-06-30": "好調期 2024Q1-Q2",
        "2024-07-01_2024-10-01": "円キャリー崩壊 2024",
        "2025-05-14_2025-08-14": "直近 2025",
        "2026-01-01_2026-06-01": "2026年初",
    }

    rows = [["期間ラベル", "期間コード", "ラウンド", "エントリー日", "エグジット日",
             "モデルリターン(%)", "日経リターン(%)", "アルファ(%)", "日経20日(%)",
             "判定", "銘柄数"]]

    all_rounds = []
    for fp in files:
        m = re.search(r"rolling21d_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$", fp)
        if not m: continue
        ps, pe = m.group(1), m.group(2)
        period_code = f"{ps}_{pe}"
        label = PERIOD_LABELS.get(period_code, period_code)

        records = list(csv.DictReader(open(fp, encoding='utf-8-sig')))
        selected = [r for r in records if r.get('selected','0').strip()=='1']
        rounds_map = {}
        for r in selected:
            rn = int(r['ラウンド'])
            rounds_map.setdefault(rn, []).append(float(r['return']) if r.get('return') else 0)

        for rn in sorted(rounds_map.keys()):
            rets = rounds_map[rn]
            e = [r for r in selected if int(r['ラウンド']) == rn][0]
            nk0 = get_nk(e['entry']); nk1 = get_nk(e['exit'])
            nk_ret = (nk1 - nk0) / nk0 * 100 if nk0 and nk1 else 0

            past = nk_series[nk_series.index < pd.Timestamp(e['entry'])].tail(20)
            nk_20d = float((nk0 - float(past.iloc[0])) / float(past.iloc[0]) * 100) if len(past)>=15 else 0
            model_ret = sum(rets)/len(rets)
            alpha = model_ret - nk_ret
            switch = "日経に切替" if nk_20d > 3.0 else "モデル"

            rows.append([label, period_code, rn, e['entry'], e['exit'],
                         round(model_ret,2), round(nk_ret,2), round(alpha,2), round(nk_20d,2),
                         switch, len(rets)])
            all_rounds.append({'model_ret': model_ret, 'nk_ret': nk_ret, 'nk_20d': nk_20d})

    # 合計行
    avg_m = sum(r['model_ret'] for r in all_rounds)/len(all_rounds)
    avg_n = sum(r['nk_ret']    for r in all_rounds)/len(all_rounds)
    rows.append(["★ 全体平均", "", len(all_rounds), "", "",
                 round(avg_m,2), round(avg_n,2), round(avg_m-avg_n,2), "", "", ""])

    return rows, all_rounds


# ────────────────────────────────────────────────────────────
# 2. 運用シミュレーション
# ────────────────────────────────────────────────────────────

def build_simulation_data(all_rounds):
    THRESH = 3.0
    # CAGR計算
    files = sorted(glob.glob(f"{BASE_DIR}/simulations/backtests/rolling21d_*.csv"))
    entries = []
    for fp in files:
        records = list(csv.DictReader(open(fp, encoding='utf-8-sig')))
        selected = [r for r in records if r.get('selected','0').strip()=='1']
        if selected:
            entries.append(selected[0]['entry'])

    if len(entries) >= 2:
        start_dt = datetime.strptime(min(entries), '%Y-%m-%d')
        end_dt   = datetime.strptime(all_rounds[-1].get('entry', max(entries)), '%Y-%m-%d') if 'entry' in all_rounds[-1] else datetime.now()
        years = 3.2
    else:
        years = 3.2

    cap_m = cap_n = cap_s = 300.0
    for r in all_rounds:
        mr = r['model_ret']/100; nr = r['nk_ret']/100
        cap_m *= (1+mr); cap_n *= (1+nr)
        cap_s *= (1+(nr if r['nk_20d'] > THRESH else mr))

    cagr_m = (cap_m/300)**(1/years) - 1
    cagr_n = (cap_n/300)**(1/years) - 1
    cagr_s = (cap_s/300)**(1/years) - 1

    scenarios = [
        ("モデル単体",      cagr_m),
        ("日経225(実績)",   cagr_n),
        ("切替戦略(nk20d>3%)", cagr_s),
        ("目標 42%/年",    0.42),
        ("日経長期平均8%",  0.08),
    ]

    # ── シート1: CAGR & 複利推移 ──
    header1 = ["戦略", "年率CAGR(%)", "1年後", "3年後", "5年後", "7年後", "10年後", "1億到達年数"]
    rows1 = [header1]
    for name, cagr in scenarios:
        mr = (1+cagr)**(1/12)-1
        row = [name, round(cagr*100,1)]
        for y in [1,3,5,7,10]:
            v = 300*(1+cagr)**y
            row.append(f"{v/10000:.2f}億" if v>=10000 else f"{v:.0f}万")
        # 1億到達
        cap = 300.0
        for mo in range(1, 601):
            cap = cap*(1+mr)
            if cap >= 10000:
                row.append(f"{mo//12}年{mo%12}ヶ月")
                break
        else:
            row.append("50年超")
        rows1.append(row)

    # ── シート2: 毎月10万積立 ──
    header2 = ["戦略", "年率CAGR(%)", "1年後", "3年後", "5年後", "7年後", "10年後", "1億到達年数"]
    rows2 = [header2]
    for name, cagr in scenarios:
        mr = (1+cagr)**(1/12)-1
        row = [name, round(cagr*100,1)]
        for y in [1,3,5,7,10]:
            months = y*12
            v = 300*(1+mr)**months
            if mr > 0:
                v += 10*((1+mr)**months-1)/mr
            row.append(f"{v/10000:.2f}億" if v>=10000 else f"{v:.0f}万")
        # 1億到達（積立あり）
        cap = 300.0
        for mo in range(1, 601):
            cap = cap*(1+mr)+10
            if cap >= 10000:
                row.append(f"{mo//12}年{mo%12}ヶ月")
                break
        else:
            row.append("50年超")
        rows2.append(row)

    return rows1, rows2



# ────────────────────────────────────────────────────────────
# 4. 特徴量一覧（44次元）
# ────────────────────────────────────────────────────────────

def build_feature_data():
    rows = [["#", "特徴量名", "グループ", "説明", "正規化・単位"]]
    features = [
        # テクニカル10
        (1,  "ret5",           "テクニカル",     "5日リターン",                    "fraction"),
        (2,  "ret20",          "テクニカル",     "20日リターン",                   "fraction"),
        (3,  "ret60",          "テクニカル",     "60日リターン",                   "fraction"),
        (4,  "ret90",          "テクニカル",     "90日リターン",                   "fraction"),
        (5,  "ma5_25",         "テクニカル",     "MA5/MA25乖離率",                 "fraction"),
        (6,  "ma25_75",        "テクニカル",     "MA25/MA75乖離率",                "fraction"),
        (7,  "rsi",            "テクニカル",     "RSI(14日)",                      "0〜100"),
        (8,  "vol20",          "テクニカル",     "20日ヒストリカルボラ",            "% 年率換算"),
        (9,  "vol60",          "テクニカル",     "60日ヒストリカルボラ",            "% 年率換算"),
        (10, "pos52",          "テクニカル",     "52週高値安値レンジ内位置",        "0〜1"),
        # トレンド反転5
        (11, "drawdown60",     "トレンド反転",   "60日高値からのドローダウン",      "fraction (負)"),
        (12, "from_hi52",      "トレンド反転",   "52週高値からの距離",             "fraction (負)"),
        (13, "down_streak",    "トレンド反転",   "連続下落日数",                   "0〜1 (20日で1.0)"),
        (14, "momentum_accel", "トレンド反転",   "モメンタム加速度(ret5-ret20/4)", "fraction"),
        (15, "ma_cross_dir",   "トレンド反転",   "MAクロス方向変化",               "fraction"),
        # 出来高3
        (16, "vr520",          "出来高",         "出来高5日/20日比",               "比率"),
        (17, "vr2060",         "出来高",         "出来高20日/60日比",              "比率"),
        (18, "vsurge",         "出来高",         "当日出来高/20日平均比",          "比率"),
        # 日経マクロ3
        (19, "nk5",            "日経マクロ",     "日経225 5日リターン",             "fraction"),
        (20, "nk20",           "日経マクロ",     "日経225 20日リターン",            "fraction"),
        (21, "nk60",           "日経マクロ",     "日経225 60日リターン",            "fraction"),
        # 60日系列要約7 (seq_feat)
        (22, "seq_mean",       "60日系列",       "60日日次リターンの平均",          "fraction"),
        (23, "seq_std",        "60日系列",       "60日日次リターンの標準偏差",      "fraction"),
        (24, "seq_skew",       "60日系列",       "60日日次リターンの歪度",          "-"),
        (25, "seq_max",        "60日系列",       "60日日次リターンの最大値",        "fraction"),
        (26, "seq_min",        "60日系列",       "60日日次リターンの最小値",        "fraction"),
        (27, "seq_pos_ratio",  "60日系列",       "60日中プラス日の割合",           "0〜1"),
        (28, "seq_autocorr",   "60日系列",       "1期ラグ自己相関",                "-1〜1"),
        # 日経相対アルファ4
        (29, "rel5",           "日経相対α",      "5日超過リターン(銘柄-日経)",      "fraction"),
        (30, "rel20",          "日経相対α",      "20日超過リターン",               "fraction"),
        (31, "rel60",          "日経相対α",      "60日超過リターン",               "fraction"),
        (32, "alpha_momentum", "日経相対α",      "アルファ加速度(rel5-rel20/4)",   "fraction"),
        # ファンダメンタル6 (新規追加)
        (33, "per_feat",       "ファンダメンタル", "PER（価格収益率）",              "PER/20-1 → 割安<0"),
        (34, "pbr_feat",       "ファンダメンタル", "PBR（株価純資産倍率）",          "PBR/1.5-1 → 1倍割れ<0"),
        (35, "roe_feat",       "ファンダメンタル", "ROE（自己資本利益率）",          "ROE/15 → 15%=1.0"),
        (36, "earn_feat",      "ファンダメンタル", "次回決算まで日数",              "days/90 (0=直前)"),
        (37, "div_feat",       "ファンダメンタル", "次回配当確定日まで日数",         "days/60 (0=直前)"),
        (38, "yutai_feat",     "ファンダメンタル", "次回優待確定日まで日数",         "days/60 (0=直前)"),
        # クロスセクショナルランク6
        (39, "cs_ret5",        "クロスセクション", "ret5の銘柄ランク百分位",         "0〜1"),
        (40, "cs_ret20",       "クロスセクション", "ret20の銘柄ランク百分位",        "0〜1"),
        (41, "cs_ret60",       "クロスセクション", "ret60の銘柄ランク百分位",        "0〜1"),
        (42, "cs_rsi",         "クロスセクション", "RSIの銘柄ランク百分位",          "0〜1"),
        (43, "cs_vol20",       "クロスセクション", "vol20の銘柄ランク百分位",        "0〜1"),
        (44, "cs_pos52",       "クロスセクション", "pos52の銘柄ランク百分位",        "0〜1"),
    ]
    for f in features:
        rows.append(list(f))
    return rows


# ────────────────────────────────────────────────────────────
# 5. 銘柄ファンダメンタル（PER/PBR/ROE + 各種確定日）
# ────────────────────────────────────────────────────────────

_FUND_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_fund_cache.json")


def build_stock_fundamentals(kabutan_top_n=None):
    """
    全銘柄をDBから取得（net順）、kabutan.jp で ROE/決算/優待を補完する。
    kabutan_top_n=None なら全銘柄、整数なら上位 N 件のみ。
    取得済みデータは _fund_cache.json にキャッシュし、再実行時は再取得しない。
    """
    import calendar as _cal
    import requests as _req
    import time as _time
    import re as _re
    import io as _io
    import json as _json
    from datetime import timedelta as _td
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
    today = date.today()
    db_path = os.path.join(BASE_DIR, "stock_alert.db")

    # ── Step0: 市場区分マップ（JPXから取得）────────────────────
    market_map = {}
    try:
        jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        jr = _req.get(jpx_url, headers=_HEADERS, timeout=30)
        jdf = pd.read_excel(_io.BytesIO(jr.content), dtype=str)
        jdf.columns = jdf.columns.str.strip()
        ccol = [c for c in jdf.columns if "コード" in c][0]
        mcol = [c for c in jdf.columns if "市場" in c][0]
        seg_short = {"プライム（内国株式）": "プライム",
                     "スタンダード（内国株式）": "スタンダード",
                     "グロース（内国株式）": "グロース"}
        for _, jr2 in jdf.iterrows():
            market_map[str(jr2[ccol]).strip()] = seg_short.get(str(jr2[mcol]).strip(),
                                                                str(jr2[mcol]).strip())
        print(f"  JPX市場区分マップ: {len(market_map)}銘柄")
    except Exception as e:
        print(f"  JPX市場区分取得失敗（無視）: {e}")

    # ── Step1: DB から全銘柄取得 ──────────────────────────────
    rows_db = []
    ranking_date = None
    yutai_map = {}
    try:
        from lib.db import get_latest_ranking_date, get_ranking_by_date, get_all_yutai
        ranking_date = get_latest_ranking_date()
        if ranking_date:
            rows_db = get_ranking_by_date(
                ranking_date,
                select="code,name,close,net,rise_prob,drop_prob,vol,recommend,rel20,per,pbr")
            for r in get_all_yutai():
                yutai_map[str(r["code"])] = r.get("yutai_month") or r.get("record_month")
    except Exception as e:
        print(f"  DB読み込みエラー: {e}")
        return [["DBエラー"]]

    print(f"  DB取得: {len(rows_db)}銘柄（最終更新 {ranking_date}）")

    # ── キャッシュ読み込み ────────────────────────────────────
    kabutan_map = {}
    if os.path.exists(_FUND_CACHE_PATH):
        try:
            cache = _json.load(open(_FUND_CACHE_PATH, encoding="utf-8"))
            if cache.get("date") == today.isoformat():
                kabutan_map = cache.get("data", {})
                print(f"  キャッシュ読込: {len(kabutan_map)}銘柄（{today}取得済み）")
        except Exception:
            pass

    # ── Step2: 上位 N 件を kabutan.jp で補完 ─────────────────
    def _days_to_event(months, day=28):
        best = 9999
        for m in months:
            for yr in [today.year, today.year + 1]:
                last = _cal.monthrange(yr, m)[1]
                d = date(yr, m, min(day, last))
                delta = (d - today).days
                if 0 <= delta < best:
                    best = delta
        return best if best < 9999 else None

    def fetch_kabutan(code):
        # 既存キャッシュから優待月を引き継ぐ（優待リクエストをスキップ）
        cached = kabutan_map.get(code, {})
        res = {"code": code, "per": None, "pbr": None, "roe": None,
               "next_earnings": None, "days_earnings": None,
               "yutai_month": cached.get("yutai_month")}
        need_yutai = "yutai_month" not in cached
        try:
            # PER/PBR + ROE + 決算日（finance ページ）
            r2 = _req.get(f"https://kabutan.jp/stock/finance/?code={code}",
                          headers=_HEADERS, timeout=10)
            if r2.status_code == 200:
                # PER / PBR
                tp = r2.text.replace("\n","").replace("\t","")
                idxp = tp.find('data-help="PER"')
                if idxp != -1:
                    pv = _re.findall(r'<td>([\d.-]+)<span', tp[idxp:idxp+600])
                    if len(pv) >= 1:
                        try: res["per"] = float(pv[0])
                        except ValueError: pass
                    if len(pv) >= 2:
                        try: res["pbr"] = float(pv[1])
                        except ValueError: pass
                t2 = r2.text.replace(" ","").replace("\n","").replace("\t","")
                idx2 = t2.find('ROE">')
                if idx2 != -1:
                    tbody = t2.find("<tbody>", idx2)
                    trows = _re.findall(r'<tr><thscope="row".*?</tr>', t2[tbody:tbody+1200])
                    if trows:
                        vs = _re.findall(r'<td[^>]*>([\d,.-]+)</td>', trows[0])
                        if len(vs) >= 3:
                            try: res["roe"] = float(vs[2])
                            except ValueError: pass
                raw_dates = _re.findall(r'(\d{2})/(\d{2})/(\d{2})', r2.text)
                past = []
                for yy, mm, dd in raw_dates:
                    try:
                        dt = datetime.strptime(f"20{yy}/{mm}/{dd}", "%Y/%m/%d").date()
                        if dt <= today: past.append(dt)
                    except ValueError: pass
                if past:
                    est = max(past) + _td(days=91)
                    res["next_earnings"] = f"{est.isoformat()}（推定）"
                    res["days_earnings"] = (est - today).days
            # 優待（キャッシュ未取得の場合のみ）
            if need_yutai:
                r3 = _req.get(f"https://kabutan.jp/stock/yutai?code={code}",
                              headers=_HEADERS, timeout=10)
                if r3.status_code == 200:
                    m3 = _re.search(r'権利確定月は(\d{1,2})月', r3.text)
                    if m3:
                        res["yutai_month"] = int(m3.group(1))
        except Exception:
            pass
        _time.sleep(0.3)
        return res

    if kabutan_top_n is None:
        target_codes = [str(r["code"]) for r in rows_db]
    else:
        target_codes = [str(r["code"]) for r in rows_db[:kabutan_top_n]]
    # キャッシュ済みはスキップ（ただしPER欠損エントリは再取得）
    pending = [c for c in target_codes
               if c not in kabutan_map or kabutan_map[c].get("per") is None]
    print(f"  kabutan.jp 補完中（対象{len(target_codes)} / 未取得{len(pending)}銘柄）...")

    done = [0]
    total = len(pending)

    def _save_cache():
        try:
            _json.dump({"date": today.isoformat(), "data": kabutan_map},
                       open(_FUND_CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass

    if pending:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(fetch_kabutan, c): c for c in pending}
            for future in as_completed(futures):
                res = future.result()
                kabutan_map[res["code"]] = res
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"    {done[0]}/{total} 完了...")
                    _save_cache()  # 進捗を逐次保存（中断対策）
        _save_cache()
    print(f"  kabutan.jp 補完完了（{len(kabutan_map)}銘柄）")

    # ── Step3: シート行を組み立て ─────────────────────────────
    def fmt_days(d):
        if d is None: return "—"
        if d <= 14:   return f"⚠️ {d}日後"
        if d <= 30:   return f"🔔 {d}日後"
        return f"{d}日後"

    # jquants_fin_summary から DPS（年間配当）をバルク読み込み
    div_map = {}  # code -> dps
    try:
        import lib.supabase_client as _sb
        _jq_rows = _sb.select(
            "jquants_fin_summary",
            f"disc_date=lte.{ranking_date}&div_ann=not.is.null"
            "&order=code.asc,disc_date.desc&select=code,div_ann"
        )
        for r in _jq_rows:
            c = str(r["code"])
            if c not in div_map and r.get("div_ann") is not None:
                div_map[c] = r["div_ann"]
    except Exception:
        pass

    header = [
        "コード", "銘柄名", "市場区分", "株価", "Netスコア", "上昇確率(%)", "下落確率(%)",
        "ボラ(%)", "推奨", "日経相対20日",
        "PER", "PBR", "ROE(%)",
        "配当利回り(%)", "1株配当(円)",
        "次回決算日(推定)", "決算まで",
        "配当確定月", "配当まで",
        "優待確定月", "優待まで",
    ]
    result_rows = [header]

    for r in rows_db:
        code = str(r["code"])
        kb   = kabutan_map.get(code, {})
        ym_db   = yutai_map.get(code)
        ym      = kb.get("yutai_month") or ym_db
        div_months = [ym] if ym else [3, 9]

        # 配当利回り計算
        dps   = div_map.get(code)
        close = r["close"] if r["close"] else None
        div_yield_str = "—"
        if dps is not None and close and close > 0:
            dy = dps / close * 100
            flag = " 🔔" if dy >= 3.0 else ""
            div_yield_str = f"{dy:.2f}%{flag}"

        result_rows.append([
            code,
            r["name"],
            market_map.get(code, "—"),
            r["close"]     if r["close"]     is not None else "—",
            round(r["net"], 2) if r["net"] is not None else "—",
            round(r["rise_prob"], 1) if r["rise_prob"] is not None else "—",
            round(r["drop_prob"], 1) if r["drop_prob"] is not None else "—",
            round(r["vol"], 1)       if r["vol"]       is not None else "—",
            r["recommend"] or "—",
            round(r["rel20"], 2)     if r["rel20"]     is not None else "—",
            (kb.get("per") if kb.get("per") is not None else (r["per"] if r["per"] is not None else "—")),
            (kb.get("pbr") if kb.get("pbr") is not None else (r["pbr"] if r["pbr"] is not None else "—")),
            kb.get("roe") if kb.get("roe") is not None else "—",
            div_yield_str,
            round(dps, 1) if dps is not None else "—",
            kb.get("next_earnings") or "—",
            fmt_days(kb.get("days_earnings")),
            ",".join(f"{m}月" for m in div_months),
            fmt_days(_days_to_event(div_months)),
            f"{ym}月" if ym else "なし",
            fmt_days(_days_to_event([ym]) if ym else None),
        ])

    return result_rows


# ────────────────────────────────────────────────────────────
# 6. モデルスペック
# ────────────────────────────────────────────────────────────

def build_model_spec():
    rows = [
        ["項目", "値", "備考"],
        ["モデルタイプ", "XGBoost (2モデルアンサンブル)", "上昇モデル + 下落モデル"],
        ["予測ターゲット", "21日後の株価変動", "21営業日 ≒ 1ヶ月"],
        ["上昇ラベル基準", "21日リターン ≥ +5%", "絶対リターン"],
        ["下落ラベル基準", "21日リターン ≤ -5%", "絶対リターン"],
        ["選択銘柄数", "上位5銘柄", "Net Score上位"],
        ["スコア計算式", "Net Score = 上昇確率 - 下落確率", ""],
        ["特徴量次元数", "44次元", "38基本 + 6クロスセクション"],
        ["", "", ""],
        ["── XGBoostパラメータ ──", "", ""],
        ["n_estimators", "5000", "early_stoppingあり"],
        ["max_depth", "5", ""],
        ["learning_rate", "0.005", ""],
        ["subsample", "0.65", ""],
        ["colsample_bytree", "0.45", ""],
        ["min_child_weight", "60", "過学習抑制"],
        ["reg_alpha (L1)", "1.5", ""],
        ["reg_lambda (L2)", "5", ""],
        ["gamma", "0.5", ""],
        ["early_stopping_rounds", "150", ""],
        ["", "", ""],
        ["── 成績（バックテスト複合）──", "", ""],
        ["平均リターン", "~2.3%/21日", "目標: >2%"],
        ["勝率", "~67%", "目標: >50%"],
        ["大勝率(+8%)", "~10%", "目標: >20% ← 未達"],
        ["日経アルファ", "+0.27%", "目標: >0%"],
        ["投資フェーズ", "Phase 1（少額投資開始可）", "2026-06-04に達成"],
    ]
    return rows


# ────────────────────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────────────────────

def main():
    print("Google Spreadsheetに接続中...")
    creds = Credentials.from_service_account_file(GCP_KEY_PATH, scopes=SCOPE)
    gc    = gspread.authorize(creds)
    sh    = gc.open_by_key(SPREADSHEET_ID)
    print(f"スプレッドシート: {sh.title}")

    # 1. バックテスト成績
    print("\n[1/5] バックテスト成績を取得中...")
    bt_rows, all_rounds = build_backtest_data()
    ws1 = get_or_create_sheet(sh, "📊 バックテスト成績")
    write_sheet(ws1, bt_rows)
    ws1.freeze(rows=1)
    print(f"  → {len(bt_rows)-1}行 書き込み完了")

    # 2. 運用シミュレーション（一括投資）
    print("\n[2/5] 運用シミュレーション（一括）を計算中...")
    sim1, sim2 = build_simulation_data(all_rounds)
    ws2 = get_or_create_sheet(sh, "💰 運用シミュレーション")
    # 一括と積立を同シートに縦並び
    spacer = [[""] * len(sim1[0])]
    label1 = [["【一括投資 300万円スタート】"] + [""] * (len(sim1[0])-1)]
    label2 = [["【毎月10万円積立 300万円スタート】"] + [""] * (len(sim2[0])-1)]
    combined = label1 + sim1 + spacer + label2 + sim2
    ws2.update(range_name="A1", values=combined)
    ws2.format("A1:H1", {"textFormat": {"bold": True, "fontSize": 11},
                          "backgroundColor": {"red":0.13,"green":0.55,"blue":0.13}})
    ws2.format("A1:H1", {"textFormat": {"foregroundColor": {"red":1,"green":1,"blue":1}, "bold": True}})
    label_row2 = len(sim1) + 3
    ws2.format(f"A{label_row2}:H{label_row2}",
               {"textFormat": {"bold": True, "fontSize": 11},
                "backgroundColor": {"red":0.13,"green":0.13,"blue":0.55}})
    ws2.format(f"A{label_row2}:H{label_row2}",
               {"textFormat": {"foregroundColor": {"red":1,"green":1,"blue":1}, "bold": True}})
    ws2.freeze(rows=2)
    print(f"  → 書き込み完了")

    # 3. 特徴量一覧
    print("\n[3/5] 特徴量一覧を書き込み中...")
    feat_rows = build_feature_data()
    ws4 = get_or_create_sheet(sh, "🧬 特徴量一覧(44次元)", rows=60, cols=10)
    write_sheet(ws4, feat_rows)
    ws4.freeze(rows=1)
    print(f"  → {len(feat_rows)-1}行 書き込み完了")

    # 5. モデルスペック
    print("\n[5/5] モデルスペックを書き込み中...")
    spec_rows = build_model_spec()
    ws5 = get_or_create_sheet(sh, "⚙️ モデルスペック", rows=40, cols=5)
    write_sheet(ws5, spec_rows)
    ws5.freeze(rows=1)
    print(f"  → 書き込み完了")

    # 6. 銘柄ファンダメンタル
    print("\n[6/6] 銘柄ファンダメンタルを取得中（kabutan.jp）...")
    fund_rows = build_stock_fundamentals()
    ws6 = get_or_create_sheet(sh, "📋 銘柄ファンダメンタル", rows=4000, cols=20)
    write_sheet(ws6, fund_rows)
    ws6.freeze(rows=1)
    print(f"  → {len(fund_rows)-1}銘柄 書き込み完了")

    print(f"\n✅ 全シート書き込み完了！")
    print(f"   URL: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}")


if __name__ == "__main__":
    main()
