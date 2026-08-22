"""
lib/buyback.py

TDnet適時開示（ext_tdnet_disclosures, category='自社株買い'）のうち「取得枠の決定」開示を
選び、原文PDFから取得枠（上限株数・上限金額・発行済比率・取得期間・方法）を抽出して
tdnet_buybacks に保存する。サイト（/stocks/[code]）・X投稿（web/x_buyback.py）・
ブログ（web/publish_buyback_articles.py）はこのテーブルを共通の事実として参照する。

自社株買い開示の大半（直近2ヶ月で425件中376件）は「自己株式の取得状況に関するお知らせ」
という月次の進捗報告で、ニュース価値があるのは取得枠を新設する「決定」開示だけ。
タイトルで決定/進捗を分け、PDFを読むのは決定のみにする。

PDFはテキスト層を持つ（TDnetはPDF/A提出）ため pypdf でテキスト化できる。取得枠の表記は
各社でバラつく（「取得し得る株式の総数」「取得する株式の総数」「取得価額の総額」、全角数字、
「百万円」単位など）ので正規表現ではなく Claude Haiku にJSONで抜かせる。数値が取れなかった
行も extract_ok=false で保存し、同じPDFを毎日読み直さないようにする。
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-alert/1.0)"}
_PDF_MAX_PAGES = 3        # 取得枠は1〜2ページ目に必ずある
_PDF_MAX_CHARS = 6000

# 「進捗」: 月次の取得状況・買付結果・取得終了の報告（取得枠は既報）
_PROGRESS_RE = re.compile(r"取得状況|取得結果|取得終了|買付け?結果|買付状況|取得の完了|事後調整")
# 「決定」: 取得枠の新設（取締役会決議）・ToSTNeT-3等の立会外買付の実施決定
_DECISION_RE = re.compile(r"決定|決議|取得枠|取得に係る事項|買付けに関する|買付に関する|ToSTNeT|ＴｏＳＴＮｅＴ|N-NET|Ｎ－ＮＥＴ")


def classify_buyback_title(title: str) -> str:
    """'decision' / 'progress' / 'other'。進捗語を含むものは決定語が混ざっていても進捗扱い
    （「取得結果及び取得終了並びに消却」のような報告を決定と誤認しないため）。"""
    t = title or ""
    if _PROGRESS_RE.search(t):
        return "progress"
    if _DECISION_RE.search(t):
        return "decision"
    return "other"


def direct_pdf_url(doc_url: str) -> str:
    """やのしんAPIのリダイレクタ（rd.php?<url>）を外してTDnet原文PDFの直URLにする。
    サイトの「原文」リンクにもこの形で出す。"""
    if not doc_url:
        return ""
    m = re.match(r"https?://webapi\.yanoshin\.jp/rd\.php\?(https?://.+)$", doc_url)
    return m.group(1) if m else doc_url


def fetch_pdf_text(url: str) -> str:
    from pypdf import PdfReader

    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(BytesIO(resp.content))
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:_PDF_MAX_PAGES])
    return text[:_PDF_MAX_CHARS]



_NUM = r"([0-9][0-9,]*)"
_DATE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"


def _normalize_text(text: str) -> str:
    """全角英数・記号をNFKCで半角に寄せ、空白を畳み、改行を取って1本の文字列にする
    （表組みのPDFは項目名と数値が別行に割れるため）。"""
    import unicodedata

    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("⾏", "行").replace("発⾏", "発行")
    return re.sub(r"\s+", "", t)


def _ymd(m, offset: int = 0) -> str:
    return f"{int(m.group(1 + offset)):04d}-{int(m.group(2 + offset)):02d}-{int(m.group(3 + offset)):02d}"


def _yen(num: str, unit: str) -> "int | None":
    n = _to_int(num)
    if n is None:
        return None
    return n * {"億円": 100_000_000, "百万円": 1_000_000, "千円": 1_000, "円": 1}[unit]


def extract_facts_regex(text: str, title: str = "", disclosed_date: str = "") -> dict:
    """TDnetの自己株式取得開示の定型表記から取得枠を正規表現で抜く。
    「取得し得る株式の総数 6,910,000株（上限）」「株式の取得価額の総額 300億円（上限）」
    「（発行済株式総数（自己株式を除く）に対する割合 10.55%）」「取得期間 2026年8月19日から
    2026年11月30日まで」が大半の開示で共通する（2026-08-18〜21の決定開示14本で確認）。
    ToSTNeT-3等の単日買付は「YYYY年M月D日午前8時45分の…買付けの委託」の日付を期間にする。
    取得枠の一部変更（変更前/変更後の2列表）はタイトルに「変更」を含むため、2つ目の値を採る。
    取れない項目はNone。"""
    t = _normalize_text(text)
    is_amend = "変更" in (title or "")
    pick = lambda ms: ms[1] if is_amend and len(ms) > 1 else ms[0]  # noqa: E731

    if is_amend:
        # 変更前/変更後の2列表。項目名は1回しか現れず数値だけが並ぶため「(上限)」付きの値を拾って2つ目を採る
        shares_ms = re.findall(_NUM + r"株\(上限\)", t)
        amount_ms = re.findall(_NUM + r"(億円|百万円|千円|円)\(上限\)", t)
    else:
        shares_ms = re.findall(r"(?:取得し得る|取得する)株式の総数[^0-9]{0,40}?" + _NUM + r"株", t)
        amount_ms = re.findall(r"取得価[額格]の総額[^0-9]{0,40}?" + _NUM + r"(億円|百万円|千円|円)", t)
    # 「…に対する割合 2.43%」が標準だが「発行済株式総数（自己株式を除く）の 13.08%」と書く社もある
    ratio_ms = re.findall(r"(?:割合|発行済株式(?:の)?総数\(自己株式を除く\)の)[^0-9]{0,5}?([0-9]+(?:\.[0-9]+)?)%", t)

    max_shares = _to_int(pick(shares_ms)) if shares_ms else None
    max_amount = _yen(*pick(amount_ms)) if amount_ms else None
    ratio = _to_float(pick(ratio_ms)) if ratio_ms else None

    period_from = period_to = None
    m = re.search(r"取得期間[^0-9]{0,10}?" + _DATE + r"(?:～|〜|から|~|-)" + _DATE, t)
    if m:
        period_from, period_to = _ymd(m), _ymd(m, 3)
    else:
        # 「本日（8/21）の終値で、8/24午前8時45分の…立会外買付取引において買付けの委託」のように
        # 同じ文に開示日と買付日が並ぶため、間に別の日付（年）を挟まない直近の日付を買付日とする
        m = re.search(_DATE + r"(?:午前|午後)?[0-9:時分]*の?[^。年]{0,60}?(?:立会外|ToSTNeT|N-NET)[^。]{0,80}?買付けの委託", t)
        if m:
            period_from = period_to = _ymd(m)

    if re.search(r"ToSTNeT-?3|立会外買付", t):
        method = "ToSTNeT-3（立会外買付）"
    elif "N-NET" in t:
        method = "N-NET3（名証立会外買付）"
    elif "立会外" in t:
        method = "立会外買付"
    elif "取引一任" in t:
        method = "取引一任方式による市場買付"
    elif "信託" in t and "市場買付" in t:
        method = "信託方式による市場買付"
    elif "市場買付" in t:
        method = "市場買付"
    else:
        method = None

    board_date = None
    m = re.search(_DATE + r"(?:付の?|開催の)?(?:当社)?取締役会", t)
    if m:
        board_date = _ymd(m)
    elif disclosed_date:
        # 「本日開催の取締役会」や書面決議（取締役会の語が日付の直後に来ない）は開示日を決議日とみなす
        board_date = disclosed_date[:10]

    will_cancel = ("消却" in (title or "")) or ("消却する株式" in t)

    return {
        "board_date": board_date,
        "max_shares": max_shares,
        "max_amount_yen": max_amount,
        "ratio_pct": ratio,
        "period_from": period_from,
        "period_to": period_to,
        "method": method,
        "purpose": None,
        "will_cancel": will_cancel,
    }


_FACT_KEYS = ("board_date", "max_shares", "max_amount_yen", "ratio_pct",
              "period_from", "period_to", "method", "purpose", "will_cancel")


def extract_facts_llm(text: str, code: str, title: str) -> dict:
    """Claude HaikuでPDF本文から取得枠をJSONで抜く（正規表現で取れなかったときの補助）。
    取れない項目はNone。API未設定・失敗時は全てNone。"""
    empty = {k: None for k in _FACT_KEYS}
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or not text.strip():
        return empty
    import anthropic

    prompt = f"""以下は日本の上場企業（証券コード{code}）の適時開示「{title}」の本文です。
自己株式取得の「今回決議した取得枠」を読み取り、JSONのみを出力してください（他のテキスト・コードフェンス禁止）。

ルール:
- max_shares: 取得し得る株式の総数の上限（株数、整数）。「上限」「（上限）」表記の数字。
- max_amount_yen: 株式の取得価額の総額の上限（円、整数）。「百万円」「億円」単位は円に換算する。
- ratio_pct: 発行済株式総数（自己株式を除く）に対する割合（%、小数）。
- board_date: 取締役会決議日（YYYY-MM-DD）。
- period_from / period_to: 取得期間（YYYY-MM-DD）。ToSTNeT-3等で単日の買付なら両方その日。
- method: 取得方法を短く（例: "東証における市場買付", "ToSTNeT-3", "信託方式による市場買付"）。
- purpose: 取得理由を40字以内で要約。
- will_cancel: 取得した自己株式の消却を同時に決議・予定していれば true、無ければ false。
- 「参考」として載っている過去の取得枠や自己株式の保有状況の数字を今回の枠と混同しない。
- 本文に無い項目は null。数値を推測で埋めない。

出力形式:
{{"board_date": "YYYY-MM-DD", "max_shares": 0, "max_amount_yen": 0, "ratio_pct": 0.0, "period_from": "YYYY-MM-DD", "period_to": "YYYY-MM-DD", "method": "", "purpose": "", "will_cancel": false}}

本文:
{text}
"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0), strict=False) if m else {}
    except Exception as e:
        print(f"[buyback] 抽出失敗 {code}: {e}")
        return empty
    return _coerce_facts(data)


def _to_int(v) -> "int | None":
    if v is None or v == "":
        return None
    try:
        n = int(round(float(str(v).replace(",", ""))))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _to_float(v) -> "float | None":
    if v is None or v == "":
        return None
    try:
        f = float(str(v).replace(",", "").rstrip("%"))
    except (TypeError, ValueError):
        return None
    return round(f, 2) if 0 < f <= 100 else None


def _to_date(v) -> "str | None":
    if not v or not isinstance(v, str):
        return None
    m = re.match(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", v)
    if not m:
        return None
    try:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    except ValueError:
        return None


def _coerce_facts(data: dict) -> dict:
    """Haikuの出力を型と範囲で検証する。桁違い（百万円を円と誤認等）は捨てずに残すが、
    負値・0・100%超の比率は不正としてNoneにする。"""
    return {
        "board_date": _to_date(data.get("board_date")),
        "max_shares": _to_int(data.get("max_shares")),
        "max_amount_yen": _to_int(data.get("max_amount_yen")),
        "ratio_pct": _to_float(data.get("ratio_pct")),
        "period_from": _to_date(data.get("period_from")),
        "period_to": _to_date(data.get("period_to")),
        "method": (str(data.get("method") or "").strip()[:60] or None),
        "purpose": (str(data.get("purpose") or "").strip()[:80] or None),
        "will_cancel": bool(data.get("will_cancel")) if data.get("will_cancel") is not None else None,
    }


def extract_buyback_facts(text: str, code: str, title: str, disclosed_date: str = "") -> dict:
    """正規表現を主、Haikuを補助にして取得枠を抜く。正規表現で上限金額か株数が取れていれば
    LLMは呼ばない（APIの使用量上限で止まっても抽出が続くようにする。2026-08-23に上限到達で
    ブログ生成ごと止まった実例あり）。LLMで補うのは正規表現が空振りした開示だけで、
    その場合も正規表現で取れた項目は残しLLM結果で空欄だけ埋める。"""
    facts = extract_facts_regex(text, title, disclosed_date)
    if facts["max_amount_yen"] is not None or facts["max_shares"] is not None:
        return facts
    llm = extract_facts_llm(text, code, title)
    return {k: (facts[k] if facts[k] is not None else llm[k]) for k in _FACT_KEYS}


def pending_decisions(days: int, retry_failed: bool = False) -> list[dict]:
    """直近days日の自社株買い「決定」開示のうち、tdnet_buybacks 未登録のもの。
    retry_failed=True なら登録済みでも extract_ok=false の行は対象に戻す。"""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    rows = sb.select(
        "ext_tdnet_disclosures",
        f"category=eq.自社株買い&disclosed_at=gte.{since}"
        "&select=code,disclosed_at,title,doc_url&order=disclosed_at.desc",
    )
    done_q = f"disclosed_at=gte.{since}&select=code,disclosed_at,title"
    if retry_failed:
        done_q += "&extract_ok=is.true"  # 抽出失敗行はもう一度読み直す
    done = sb.select("tdnet_buybacks", done_q)
    done_keys = {(d["code"], d["disclosed_at"], d["title"]) for d in done}
    return [
        r for r in rows
        if classify_buyback_title(r["title"]) == "decision"
        and (r["code"], r["disclosed_at"], r["title"]) not in done_keys
        and r.get("doc_url")
    ]


def enrich(days: int = 3, dry_run: bool = False, limit: int = 0, retry_failed: bool = False) -> list[dict]:
    """未処理の決定開示を抽出して保存。戻り値は保存した行（dry_runでも返す）。"""
    targets = pending_decisions(days, retry_failed=retry_failed)
    if limit:
        targets = targets[:limit]
    print(f"[buyback] 未処理の決定開示: {len(targets)}件")
    saved = []
    for r in targets:
        url = direct_pdf_url(r["doc_url"])
        try:
            text = fetch_pdf_text(url)
        except Exception as e:
            print(f"[buyback] PDF取得失敗 {r['code']} {url}: {e}")
            text = ""
        facts = extract_buyback_facts(text, r["code"], r["title"], r["disclosed_at"])
        row = {
            "code": r["code"], "disclosed_at": r["disclosed_at"], "title": r["title"],
            "doc_url": url, **facts,
            "extract_ok": facts["max_amount_yen"] is not None or facts["max_shares"] is not None,
        }
        mark = "✓" if row["extract_ok"] else "✗"
        amt = f"{facts['max_amount_yen']/1e8:.1f}億円" if facts["max_amount_yen"] else "-"
        print(f"  {mark} {r['code']} {r['disclosed_at'][:10]} 上限{amt} {facts['ratio_pct'] or '-'}% {facts['method'] or ''}")
        saved.append(row)
    if saved and not dry_run:
        sb.upsert("tdnet_buybacks", saved, on_conflict="code,disclosed_at,title")
        print(f"[buyback] DB保存: {len(saved)}件 → tdnet_buybacks")
    return saved
