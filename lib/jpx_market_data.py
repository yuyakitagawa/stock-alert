"""
lib/jpx_market_data.py
JPX公式の無料データ（空売り残高報告 / 個別銘柄信用取引週末残高）を取得するモジュール。

データ源は JPX 公式サイトの公開ファイル（Excel / CSV-zip）。
列名はキーワードで柔軟にマッチさせ、レイアウト変更にある程度耐える。
失敗時は常に [] を返す（例外を伝播させない）。

保存先:
  - jpx_short_selling   : 空売り残高報告（残高割合0.5%以上）
  - jpx_margin_balance  : 個別銘柄信用取引週末残高

⚠️ 注意:
  JPX のファイルレイアウトは予告なく変わりうる。初回実運用時は
  --dry-run で取得列を必ず目視確認すること（tools/fetch_jpx_market.py）。
"""
import io
import re
import zipfile
import requests
import pandas as pd

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-alert/1.0)"}
_TIMEOUT = 30

# 空売り残高報告: 公開ページ（日次でExcelが追加される）
_SHORT_INDEX = "https://www.jpx.co.jp/markets/public/short-selling/index.html"
# 信用取引残高: 公開ページ（週次でCSV-zipが追加される）
_MARGIN_INDEX = "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html"


def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """列名に keywords のいずれかを含む最初の列名を返す。"""
    for col in df.columns:
        name = str(col)
        for kw in keywords:
            if kw in name:
                return col
    return None


def _to_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "−", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_code(raw) -> str | None:
    """銘柄コードを4桁に正規化。5桁末尾0は4桁に。"""
    if raw is None:
        return None
    s = str(raw).strip()
    s = re.sub(r"\.0$", "", s)  # pandas が 7203 → '7203.0' にするケース
    if len(s) == 5 and s.endswith("0"):
        s = s[:4]
    if not re.match(r"^\d{4}", s):
        return None
    return s[:4]


def _list_links(index_url: str, pattern: str) -> list[str]:
    """インデックスページから pattern にマッチするファイルリンクを抽出。"""
    try:
        resp = requests.get(index_url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            print(f"[jpx] index HTTP {resp.status_code}: {index_url}")
            return []
        html = resp.text
    except Exception as e:
        print(f"[jpx] index取得失敗: {e}")
        return []

    links = re.findall(r'href="([^"]+)"', html)
    base = "https://www.jpx.co.jp"
    out = []
    for href in links:
        if re.search(pattern, href, re.IGNORECASE):
            url = href if href.startswith("http") else base + href
            out.append(url)
    # 重複除去・順序維持
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# ── 空売り残高報告 ──────────────────────────────────────────────

def _parse_short_excel(content: bytes) -> list[dict]:
    """空売り残高報告のExcelをパース。列名キーワードで柔軟マッチ。"""
    try:
        df = pd.read_excel(io.BytesIO(content), dtype=str, header=None)
    except Exception as e:
        print(f"[jpx] short Excel読込失敗: {e}")
        return []

    # ヘッダー行を探す（「銘柄コード」を含む行）
    header_row = None
    for i in range(min(15, len(df))):
        row_vals = [str(x) for x in df.iloc[i].tolist()]
        if any("コード" in v for v in row_vals):
            header_row = i
            break
    if header_row is None:
        return []

    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    c_code  = _find_col(df, "コード")
    c_date  = _find_col(df, "計算年月日", "年月日", "日付")
    c_name  = _find_col(df, "商号", "名称", "氏名")
    c_ratio = _find_col(df, "残高割合", "割合")
    c_shr   = _find_col(df, "残高数量", "数量", "株数")
    if not (c_code and c_date):
        return []

    out = []
    for _, r in df.iterrows():
        code = _normalize_code(r.get(c_code))
        calc_date = _parse_jpx_date(r.get(c_date))
        if not code or not calc_date:
            continue
        out.append({
            "code": code,
            "calc_date": calc_date,
            "short_seller": (str(r.get(c_name)).strip() if c_name else "不明")[:200],
            "short_ratio": _to_float(r.get(c_ratio)) if c_ratio else None,
            "short_shares": _to_float(r.get(c_shr)) if c_shr else None,
        })
    return out


def _parse_jpx_date(v) -> str | None:
    """JPXの和暦/西暦様々な日付表現を ISO(YYYY-MM-DD) に。"""
    if v is None:
        return None
    s = str(v).strip()
    m = re.search(r"(\d{4})[/\-年](\d{1,2})[/\-月](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    # Excelシリアル値
    try:
        f = float(s)
        if 30000 < f < 60000:
            ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(f))
            return ts.date().isoformat()
    except ValueError:
        pass
    return None


def fetch_short_selling(max_files: int = 3, persist: bool = True) -> list[dict]:
    """直近の空売り残高報告ファイルを取得して jpx_short_selling に保存。

    Args: max_files: 取得する最新ファイル数
    Returns: 取得レコードのリスト
    """
    links = _list_links(_SHORT_INDEX, r"short-selling.*\.(xls|xlsx)$|balance.*\.(xls|xlsx)$")
    if not links:
        # フォールバック: ページ内の任意のExcelリンク
        links = _list_links(_SHORT_INDEX, r"\.(xls|xlsx)$")
    if not links:
        print("[jpx] 空売り残高ファイルが見つからない")
        return []

    all_recs = []
    for url in links[:max_files]:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code != 200:
                continue
            recs = _parse_short_excel(resp.content)
            all_recs.extend(recs)
            print(f"[jpx] 空売り {url.split('/')[-1]}: {len(recs)}件")
        except Exception as e:
            print(f"[jpx] 空売りファイル取得失敗 {url}: {e}")

    if persist and all_recs:
        _persist("jpx_short_selling", all_recs, "code,calc_date,short_seller")
    return all_recs


# ── mtdailyk 位置ベースパーサー ─────────────────────────────────

# mtdailyk の列レイアウト（0-indexed）
_MT_C_CODE = 6
_MT_C_SELL = 8
_MT_C_BUY  = 11


def _parse_mtdailyk(df: pd.DataFrame, fallback_date: str | None) -> list[dict]:
    """mtdailyk XLS の位置ベースパース。ヘッダー行をスキップしてデータ行を取得。"""
    if df.shape[1] <= _MT_C_BUY:
        print(f"[jpx] mtdailyk 列数不足: {df.shape[1]}列（12列以上必要）")
        return []
    if not fallback_date:
        print("[jpx] mtdailyk 日付推定失敗")
        return []

    # データ開始行を探す（コード列に4-5桁数値が入っている最初の行）
    start = 0
    for i in range(min(20, len(df))):
        val = str(df.iloc[i, _MT_C_CODE]).strip().replace(".0", "")
        if re.match(r"^\d{4,5}$", val):
            start = i
            break

    out = []
    for i in range(start, len(df)):
        row = df.iloc[i]
        code = _normalize_code(row.iloc[_MT_C_CODE])
        if not code:
            continue
        out.append({
            "code": code,
            "record_date": fallback_date,
            "margin_buy": _to_float(row.iloc[_MT_C_BUY]),
            "margin_sell": _to_float(row.iloc[_MT_C_SELL]),
        })
    if out:
        print(f"[jpx] mtdailyk パース成功: {len(out)}件 (日付={fallback_date})")
    return out


# ── 個別銘柄信用取引週末残高 ────────────────────────────────────

def _parse_margin_file(content: bytes, url: str = "") -> list[dict]:
    """信用取引残高ファイルをパース。XLS/XLSX/CSV/ZIP対応。"""
    ext = url.rsplit(".", 1)[-1].lower() if url else ""

    # Excel
    if ext in ("xls", "xlsx") or content[:8].startswith(b"\xd0\xcf\x11\xe0"):
        try:
            df = pd.read_excel(io.BytesIO(content), dtype=str, header=None)
        except Exception as e:
            print(f"[jpx] margin Excel読込失敗: {e}")
            return []
    else:
        # ZIP → CSV展開
        raw = content
        if content[:2] == b"PK":
            try:
                zf = zipfile.ZipFile(io.BytesIO(content))
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    return []
                raw = zf.read(csv_names[0])
            except Exception as e:
                print(f"[jpx] margin zip展開失敗: {e}")
                return []
        df = None
        for enc in ("cp932", "utf-8", "shift_jis"):
            try:
                df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc, header=None)
                break
            except Exception:
                continue

    if df is None or df.empty:
        return []

    # ファイル名から日付を推定（mtdailyk20260625 → 2026-06-25）
    fallback_date = None
    m_url_date = re.search(r"(\d{4})(\d{2})(\d{2})", url)
    if m_url_date:
        fallback_date = f"{m_url_date.group(1)}-{m_url_date.group(2)}-{m_url_date.group(3)}"

    # mtdailyk 形式: 位置ベースの列レイアウト（キーワードヘッダーなし）
    if "mtdailyk" in url.lower():
        return _parse_mtdailyk(df, fallback_date)

    # ヘッダー行を探す（「コード」or「銘柄」を含む行）
    header_row = None
    for i in range(min(20, len(df))):
        row_vals = [str(x) for x in df.iloc[i].tolist()]
        if any("コード" in v or "銘柄コード" in v for v in row_vals):
            header_row = i
            break
    if header_row is None:
        print(f"[jpx] margin ヘッダー行が見つからない。先頭10行:")
        for i in range(min(10, len(df))):
            print(f"  row{i}: {df.iloc[i].tolist()}")
        return []
    df.columns = [str(c).strip() for c in df.iloc[header_row].tolist()]
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    c_code = _find_col(df, "銘柄コード", "コード")
    c_date = _find_col(df, "申込日付", "年月日", "日付", "残高日付")
    c_buy  = _find_col(df, "信用買残", "信用買", "買残", "買い残", "融資残高")
    c_sell = _find_col(df, "信用売残", "信用売", "売残", "売り残", "貸株残高")
    if not c_code:
        print(f"[jpx] margin コード列が見つからない。列名: {list(df.columns)}")
        return []
    if not c_date and not fallback_date:
        print(f"[jpx] margin 日付列なし＆ファイル名日付なし。列名: {list(df.columns)}")
        return []

    out = []
    for _, r in df.iterrows():
        code = _normalize_code(r.get(c_code))
        rec_date = _parse_jpx_date(r.get(c_date)) if c_date else fallback_date
        if not code or not rec_date:
            continue
        out.append({
            "code": code,
            "record_date": rec_date,
            "margin_buy": _to_float(r.get(c_buy)) if c_buy else None,
            "margin_sell": _to_float(r.get(c_sell)) if c_sell else None,
        })
    return out


def fetch_margin_balance(max_files: int = 2, persist: bool = True) -> list[dict]:
    """直近の信用取引週末残高ファイルを取得して jpx_margin_balance に保存。"""
    links = _list_links(_MARGIN_INDEX, r"\.(xls|xlsx|zip|csv)$")
    if not links:
        print("[jpx] 信用残ファイルが見つからない")
        print(f"[jpx] 対象ページ: {_MARGIN_INDEX}")
        all_links = _list_links(_MARGIN_INDEX, r"\.[a-z]{2,4}$")
        if all_links:
            print(f"[jpx] ページ内リンク（先頭5件）: {all_links[:5]}")
        return []

    all_recs = []
    for url in links[:max_files]:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code != 200:
                print(f"[jpx] 信用残ファイル HTTP {resp.status_code}: {url}")
                continue
            recs = _parse_margin_file(resp.content, url)
            all_recs.extend(recs)
            print(f"[jpx] 信用残 {url.split('/')[-1]}: {len(recs)}件")
        except Exception as e:
            print(f"[jpx] 信用残ファイル取得失敗 {url}: {e}")

    if persist and all_recs:
        _persist("jpx_margin_balance", all_recs, "code,record_date")
    return all_recs


def _persist(table: str, records: list[dict], on_conflict: str) -> None:
    try:
        import lib.supabase_client as sb
        sb.upsert(table, records, on_conflict=on_conflict)
        print(f"[jpx] DB保存: {len(records)}件 → {table}")
    except Exception as e:
        print(f"[jpx] DB保存失敗 ({table}): {e}")
