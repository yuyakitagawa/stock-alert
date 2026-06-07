"""
fetch_fundamentals_history.py
kabutan.jp の財務ページから年度別ファンダ（EPS・ROE・BPS・発表日）を取得し
fundamentals_annual テーブルへ格納する。backtest の point-in-time 再構成用。

通期業績テーブル : 決算期 / 修正1株益(EPS) / 発表日
収益性テーブル   : 決算期 / ROE
財務状況テーブル : 決算期 / 1株純資産(BPS) / 発表日
"""
import sys, os, re, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.db import upsert_fundamentals_annual, get_fundamentals_codes_count

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}


def _strip(c):
    return re.sub(r"<[^>]+>", "", c).replace("&nbsp;", "").strip()


def _parse_rows(table_html):
    """テーブルHTMLを [[cell,...], ...] に変換（ヘッダー含む）。"""
    out = []
    for tr in re.findall(r"<tr[^>]*>.*?</tr>", table_html):
        cells = [_strip(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr)]
        out.append(cells)
    return out


def _find_table(tables_parsed, required_headers):
    """ヘッダーに required_headers を全て含む最初のテーブルを返す。"""
    for rows in tables_parsed:
        if not rows:
            continue
        hdr = rows[0]
        if all(any(req in h for h in hdr) for req in required_headers):
            return rows
    return None


def _fy_end(label):
    """'I 2023.03' / 'I予2027.03' → '2023-03'。予想行も拾う。"""
    m = re.search(r"(\d{4})\.(\d{2})", label)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _to_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _announce(label_cell):
    """発表日セル 'YY/MM/DD' → 'YYYY-MM-DD'。"""
    m = re.search(r"(\d{2})/(\d{2})/(\d{2})", label_cell)
    if not m:
        return None
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"


def fetch_one(code):
    """1銘柄の年度別ファンダ行リストを返す。失敗時は []。"""
    try:
        r = requests.get(f"https://kabutan.jp/stock/finance/?code={code}",
                         headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        clean = r.text.replace("\n", "").replace("\t", "")
        tables = re.findall(r"<table[^>]*>.*?</table>", clean)
        parsed = [_parse_rows(t) for t in tables]

        # 通期業績: 決算期 / 修正1株益 / 発表日（最初に一致したものを通期とみなす）
        eps_tbl = _find_table(parsed, ["決算期", "修正1株益", "発表日"])
        # 収益性: 決算期 / ＲＯＥ
        roe_tbl = _find_table(parsed, ["決算期", "ＲＯＥ"])
        # 財務状況: 決算期 / １株純資産
        bps_tbl = _find_table(parsed, ["決算期", "１株純資産"])

        merged = {}  # fy_end -> dict

        if eps_tbl:
            hdr = eps_tbl[0]
            i_eps = next((i for i, h in enumerate(hdr) if "修正1株益" in h), 5)
            i_ann = next((i for i, h in enumerate(hdr) if "発表日" in h), len(hdr) - 1)
            for row in eps_tbl[1:]:
                if len(row) <= max(i_eps, i_ann):
                    continue
                fy = _fy_end(row[0])
                if not fy:
                    continue
                eps = _to_float(row[i_eps])
                ann = _announce(row[i_ann])
                if eps is None and ann is None:
                    continue
                merged.setdefault(fy, {})["eps"] = eps
                if ann:
                    merged[fy]["announce_date"] = ann

        if roe_tbl:
            hdr = roe_tbl[0]
            i_roe = next((i for i, h in enumerate(hdr) if "ＲＯＥ" in h), 4)
            for row in roe_tbl[1:]:
                if len(row) <= i_roe:
                    continue
                fy = _fy_end(row[0])
                if not fy:
                    continue
                roe = _to_float(row[i_roe])
                if roe is not None:
                    merged.setdefault(fy, {})["roe"] = roe

        if bps_tbl:
            hdr = bps_tbl[0]
            i_bps = next((i for i, h in enumerate(hdr) if "１株純資産" in h), 1)
            i_ann = next((i for i, h in enumerate(hdr) if "発表日" in h), len(hdr) - 1)
            for row in bps_tbl[1:]:
                if len(row) <= i_bps:
                    continue
                fy = _fy_end(row[0])
                if not fy:
                    continue
                bps = _to_float(row[i_bps])
                if bps is not None:
                    merged.setdefault(fy, {})["bps"] = bps
                if len(row) > i_ann:
                    ann = _announce(row[i_ann])
                    if ann and "announce_date" not in merged.get(fy, {}):
                        merged.setdefault(fy, {})["announce_date"] = ann

        rows_out = []
        for fy, d in merged.items():
            rows_out.append({
                "fy_end": fy,
                "announce_date": d.get("announce_date"),
                "eps": d.get("eps"),
                "roe": d.get("roe"),
                "bps": d.get("bps"),
            })
        return rows_out
    except Exception:
        return []


def get_tse_codes():
    import io, pandas as pd
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
    df.columns = df.columns.str.strip()
    ccol = [c for c in df.columns if "コード" in c][0]
    mcol = [c for c in df.columns if "市場" in c][0]
    df["code"] = df[ccol].str.strip()
    # 内国株式のみ（ETF/REIT/PRO除外）
    df = df[df[mcol].str.contains("内国株式", na=False)]
    df = df[df["code"].str.match(r"^[1-9]\d{3}$")]
    return df["code"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="先頭N銘柄のみ（テスト用）")
    ap.add_argument("--codes", type=str, default=None, help="カンマ区切りの銘柄コード（テスト用）")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    today = date.today().isoformat()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    else:
        print("TSE銘柄リスト取得中...")
        codes = get_tse_codes()
        if args.limit:
            codes = codes[:args.limit]
    print(f"対象: {len(codes)}銘柄")

    done = [0]
    ok = [0]
    total = len(codes)

    def work(code):
        rows = fetch_one(code)
        time.sleep(0.25)
        if rows:
            upsert_fundamentals_annual(code, rows, today)
            return code, len(rows)
        return code, 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, c): c for c in codes}
        for fut in as_completed(futures):
            code, n = fut.result()
            done[0] += 1
            if n > 0:
                ok[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{total} 完了（取得成功 {ok[0]}）")

    print(f"\n完了: {ok[0]}/{total}銘柄を格納")
    print(f"DB内ファンダ銘柄数: {get_fundamentals_codes_count()}")


if __name__ == "__main__":
    main()
