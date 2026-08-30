"""
lib/edinet.py
EDINET API v2 経由で大量保有報告書（5%ルール）を取得するモジュール。

目的（GARP・イベント駆動）:
  「構造的に買収・改革が起きやすい候補（カタリストスクリーン）」× 「実際に誰かが
  5%超を買い集めた事実（大量保有報告書）」を突合し、本物の先回り候補を洗い出す。

必要なもの:
  - 環境変数 EDINET_API_KEY（EDINET API v2 のサブスクリプションキー）
    ※ .env に置けば自動ロード。クラウドは GitHub Secrets に登録。

設計方針（既存 alt_data.py に倣う）:
  - DBキャッシュ前提（日次スキャンで edinet_large_holdings に蓄積）
  - 失敗時は常に [] を返す（例外は伝播しない）
  - documents.json のメタデータ + XBRL本文から保有割合を取得。

docTypeCode:
  350 = 大量保有報告書
  360 = 変更報告書（保有割合の増減・追加取得）
"""
import os
import re
import io
import unicodedata
import html as html_lib
import zipfile
import requests
from datetime import date, timedelta

_API_BASE = "https://api.edinet-fsa.go.jp/api/v2"
_LARGE_HOLDING_TYPES = {"350", "360"}  # 大量保有報告書 / 変更報告書


def _api_key() -> str:
    """EDINET_API_KEY を返す。.env も探索。未設定なら空文字。"""
    key = os.environ.get("EDINET_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("EDINET_API_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip()
                    break
    return key


def _normalize_sec_code(sec_code: str) -> "str | None":
    """EDINET の secCode を 4桁の証券コードに正規化。
    形式: 4桁数字（7794）、5桁末尾0（77940）、英字混じり（268A, 268A0）
    """
    if not sec_code:
        return None
    s = str(sec_code).strip().upper()
    if len(s) == 5 and s.endswith("0"):
        return s[:4]
    if len(s) == 4:
        return s
    return s[:4] if len(s) >= 4 else None


def fetch_documents_list(target_date: "str | date") -> list:
    """指定日に EDINET へ提出された書類メタデータ一覧を返す。

    Returns: documents.json の results 配列（取得失敗時は []）。
    """
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    key = _api_key()
    if not key:
        return []
    try:
        resp = requests.get(
            f"{_API_BASE}/documents.json",
            params={"date": target_date, "type": 2, "Subscription-Key": key},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("results", []) or []
    except Exception:
        return []


def verify_api(target_date: "str | date | None" = None) -> dict:
    """APIキーの有効性を確認する。指定日（既定=直近の平日）の書類一覧を取得し、
    HTTPステータス・総件数・大量保有件数を返す。

    Returns: {'ok': bool, 'reason': str, 'status': int|None, 'total': int, 'large': int, 'date': str}
    """
    if not _api_key():
        return {"ok": False, "reason": "EDINET_API_KEY 未設定", "status": None,
                "total": 0, "large": 0, "date": ""}
    if target_date is None:
        d = date.today()
        while d.weekday() >= 5:  # 土日は遡る
            d -= timedelta(days=1)
        target_date = d
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    try:
        resp = requests.get(
            f"{_API_BASE}/documents.json",
            params={"date": target_date, "type": 2, "Subscription-Key": _api_key()},
            timeout=15,
        )
        status = resp.status_code
        if status != 200:
            reason = "キー無効/権限不足" if status in (401, 403) else f"HTTP {status}"
            return {"ok": False, "reason": reason, "status": status,
                    "total": 0, "large": 0, "date": target_date}
        results = resp.json().get("results", []) or []
        large = sum(1 for r in results
                    if str(r.get("docTypeCode", "")) in _LARGE_HOLDING_TYPES)
        return {"ok": True, "reason": "OK", "status": status,
                "total": len(results), "large": large, "date": target_date}
    except Exception as e:
        return {"ok": False, "reason": f"例外: {e}", "status": None,
                "total": 0, "large": 0, "date": target_date}


def _fetch_xbrl_text(doc_id: str) -> "str | None":
    """XBRL本文（ZIP内 PublicDoc/*.xbrl）をテキストで返す。取得失敗時は None。"""
    key = _api_key()
    if not key or not doc_id:
        return None
    try:
        resp = requests.get(
            f"{_API_BASE}/documents/{doc_id}",
            params={"type": 1, "Subscription-Key": key},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"    XBRL HTTP {resp.status_code}: {doc_id}")
            return None
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        xbrl_names = [n for n in zf.namelist()
                      if "PublicDoc" in n and n.endswith(".xbrl")]
        if not xbrl_names:
            print(f"    XBRL内に.xbrlファイルなし: {doc_id} files={zf.namelist()[:5]}")
            return None
        return zf.read(xbrl_names[0]).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    XBRL例外: {doc_id} {e}")
        return None


def fetch_xbrl_details(doc_id: str) -> dict:
    """XBRL本文から対象銘柄コード(issuer_code)、保有割合(holding_ratio)、
    直前報告時の保有割合(holding_ratio_prior)、発行者名(issuer_name)を抽出する。

    holding_ratio_prior は変更報告書(360)にのみ存在し、holding_ratio との差分から
    増加（買い）/減少（売り）を判定するのに使う（doc_descriptionの文言に依存しない）。

    短期大量譲渡に該当する開示は「譲渡の相手方・単価」の表も持つので併せて返す
    （short_term_transfers。該当しない開示では空リスト）。

    保有目的・取得資金・保有株数・報告義務発生日は parse_holding_details() が返す
    キーをそのまま含める。

    Returns: {"issuer_code": str|None, "holding_ratio": float|None,
              "holding_ratio_prior": float|None, "issuer_name": str|None,
              "short_term_transfers": list,
              "purpose_of_holding": str|None, "important_proposal": str|None,
              "shares_held": int|None, "shares_outstanding": int|None,
              "funding_total": int|None, "funding_own": int|None,
              "funding_borrowings": int|None, "obligation_date": str|None}
    """
    result = {"issuer_code": None, "holding_ratio": None,
              "holding_ratio_prior": None, "issuer_name": None,
              "short_term_transfers": [],
              "purpose_of_holding": None, "important_proposal": None,
              "shares_held": None, "shares_outstanding": None,
              "funding_total": None, "funding_own": None,
              "funding_borrowings": None, "obligation_date": None}
    xbrl_text = _fetch_xbrl_text(doc_id)
    if not xbrl_text:
        print(f"    ⚠ XBRL取得失敗: {doc_id}")
        return result

    # 対象銘柄の証券コード
    # 優先1: SecurityCodeOfIssuer（発行者コード = 対象銘柄）
    # 優先2: SecurityCodeDEI（nil でないもの）
    # コード形式: 4桁数字 or 3桁数字+英字（268A等、2024年以降のIPO）
    sec_patterns = [
        r'<[^>]*SecurityCodeOfIssuer[^>]*>\s*([0-9A-Za-z]{4,5})\s*<',
        r'<[^>]*SecurityCodeDEI[^>]*(?<!nil="true")>\s*([0-9A-Za-z]{4,5})\s*<',
    ]
    for pat in sec_patterns:
        m = re.search(pat, xbrl_text)
        if m:
            result["issuer_code"] = _normalize_sec_code(m.group(1))
            break

    # 発行者名（対象企業名）
    name_patterns = [
        r'<[^>]*IssuerNameDEI[^>]*>\s*(.+?)\s*<',
        r'<[^>]*NameOfIssuer[^>]*>\s*(.+?)\s*<',
    ]
    for pat in name_patterns:
        m = re.search(pat, xbrl_text)
        if m:
            result["issuer_name"] = m.group(1).strip()
            break

    # 保有割合（提出後）と直前報告時の保有割合。
    # 共同保有の報告書では、メンバー無しの合算context（FilingDateInstant）が
    # 「提出者＋共同保有者の合計」＝報告書の見出しに出る数値。保有者ごとのcontextを
    # 先に拾うと筆頭保有者の1枠だけを保有割合として出してしまう。
    # 実測（2026-08-30・1,101件）で656件＝60%がこのズレを起こしていた。
    # 例: 野村證券→トリケミカル研究所を0.46%（実際は10.41%）、
    #     細沼順人→成友興業を23.61%（実際は78.66%）と出していた。
    result["holding_ratio"] = _aggregate_ratio(
        xbrl_text, "HoldingRatioOfShareCertificatesEtc")
    result["holding_ratio_prior"] = _aggregate_ratio(
        xbrl_text, "HoldingRatioOfShareCertificatesEtcPerLastReport")

    # 上のcontext方式は `jplvh_cor:` 名前空間のタグだけを見る。名前空間や様式が違って
    # 1件も取れなかった開示のために、従来の素の正規表現を残しておく（比率はサイトの
    # 主要数値なので、精度を上げる変更で取得できる件数を減らしてはいけない）。
    if result["holding_ratio"] is None:
        m = re.search(
            r'<[^>]*HoldingRatioOfShareCertificatesEtc(?:DEI)?[\s>](?:(?!PerLastReport)[^>])*>'
            r'\s*([0-9]*\.?[0-9]+)\s*<', xbrl_text)
        if m:
            result["holding_ratio"] = _normalize_ratio(m.group(1))
    if result["holding_ratio_prior"] is None:
        m = re.search(
            r'<[^>]*HoldingRatioOfShareCertificatesEtc(?:DEI)?PerLastReport[^>]*>'
            r'\s*([0-9]*\.?[0-9]+)\s*<', xbrl_text)
        if m:
            result["holding_ratio_prior"] = _normalize_ratio(m.group(1))

    # 短期大量譲渡の「譲渡の相手方・単価」（該当しない開示では空リスト）
    result["short_term_transfers"] = parse_short_term_transfers(xbrl_text)

    # 保有目的・取得資金・保有株数・報告義務発生日
    result.update(parse_holding_details(xbrl_text))
    # XBRL本文まで取れた回だけ日付を立てる。値が全部Noneの開示（全部売却・資金の記載なし）と
    # 「そもそもXBRLを取りに行っていない行」を区別するための印で、
    # 未取得行の絞り込み（xbrl_detail_fetched_date is null）に使う。
    result["xbrl_detail_fetched_date"] = date.today().isoformat()

    return result


# ---- 保有目的・取得資金（XBRLの本表。EDINETが構造化して出している）----
# 大量保有報告書のXBRLには、これまで使っていた保有割合のほかに
# 「保有目的」「取得資金の総額／自己資金／借入金」「保有株数」「発行済株式総数」
# 「報告義務発生日」が入っている。ここから
#   - 保有目的（純投資 / 政策保有 / 重要提案行為等 / 経営参加 ...）の判定
#   - 平均取得単価 = 取得資金の総額 ÷ 保有株数（開示ベースの取得コスト）
#   - 借入比率 = 借入金 ÷ 取得資金（レバレッジをかけた買いかどうか）
#   - 報告義務発生日→提出日のラグ（買った日と開示日のズレ）
# が計算できる。株価からの概算に頼っていた金額の裏取りにもなる。
#
# 値は保有者ごとのcontext（...FilerLargeVolumeHolder{N}Member）に入る。
# 株数・発行済株式総数・保有割合はメンバー無しの合算context（FilingDateInstant）も
# あるのでそれを優先し、取得資金系は合算contextが無いのでメンバー分を足し上げる。

# 数値タグは値に子タグを含まない。`[^<]*` で閉じタグまで取り切り、
# 別のcontextの要素まで飲み込まないようにする（`.*?` だと欠損時に暴走する）。
_NUM_TAG_TMPL = r'<jplvh_cor:{name}\b([^>]*)>([^<]*)</jplvh_cor:{name}>'
# テキストタグは中に注記のタグが入りうるので非貪欲で取り、後からタグを落とす。
_TEXT_TAG_TMPL = r'<jplvh_cor:{name}\b([^>]*)>(.*?)</jplvh_cor:{name}>'
_MEMBER_RE = re.compile(r'FilerLargeVolumeHolder(\d+)Member')
_CONTEXT_RE = re.compile(r'contextRef="([^"]+)"')


def _tag_by_context(xbrl_text: str, name: str, text: bool = False) -> dict:
    """タグ名を context ごとに取り出す。{context: 値} を返す（重複は先勝ち）。"""
    tmpl = _TEXT_TAG_TMPL if text else _NUM_TAG_TMPL
    pattern = re.compile(tmpl.format(name=name), re.S)
    out: dict[str, str] = {}
    for m in pattern.finditer(xbrl_text):
        ctx_match = _CONTEXT_RE.search(m.group(1))
        ctx = ctx_match.group(1) if ctx_match else ""
        value = m.group(2)
        if text:
            value = re.sub(r'<[^>]+>', '', _unescape(value))
        value = re.sub(r'\s+', ' ', value).strip()
        if ctx not in out and value:
            out[ctx] = value
    return out


def _sum_over_members(by_context: dict) -> "int | None":
    """保有者ごとのcontextに散った金額を足し上げる（合算contextが無い項目用）。"""
    total = None
    for ctx, value in by_context.items():
        if not _MEMBER_RE.search(ctx):
            continue
        try:
            total = (total or 0) + int(float(value))
        except ValueError:
            continue
    return total


def _aggregate_or_sum(by_context: dict) -> "int | None":
    """メンバー無しの合算contextを優先し、無ければメンバー分を足し上げる。"""
    for ctx, value in by_context.items():
        if not _MEMBER_RE.search(ctx):
            try:
                return int(float(value))
            except ValueError:
                pass
    return _sum_over_members(by_context)


def _normalize_ratio(value: str) -> "float | None":
    """保有割合は小数（0.0778 = 7.78%）とパーセント（7.78）の両方の書き方がある。"""
    try:
        val = float(value)
    except ValueError:
        return None
    if val < 1.0:
        val *= 100
    return round(val, 2)


def _aggregate_ratio(xbrl_text: str, name: str) -> "float | None":
    """保有割合をメンバー無しの合算context優先で取る。

    合算contextが無い様式では保有者ごとの割合を足し上げる（単独保有なら1件なのでそのまま）。
    株数（_aggregate_or_sum）と同じ方針だが、戻り値がintではなくパーセントの小数なので分けている。
    """
    by_context = _tag_by_context(xbrl_text, name)
    for ctx, value in by_context.items():
        if not _MEMBER_RE.search(ctx):
            ratio = _normalize_ratio(value)
            if ratio is not None:
                return ratio
    total = None
    for ctx, value in by_context.items():
        if not _MEMBER_RE.search(ctx):
            continue
        ratio = _normalize_ratio(value)
        if ratio is not None:
            total = (total or 0) + ratio
    return round(total, 2) if total is not None else None


def _join_member_texts(by_context: dict, limit: int = 3) -> "str | None":
    """保有者ごとの文言を重複を除いて連結する（共同保有で目的が違うことがある）。"""
    seen: list[str] = []
    for ctx, value in sorted(by_context.items()):
        if value not in seen:
            seen.append(value)
    if not seen:
        return None
    return "\n".join(seen[:limit])


def parse_holding_details(xbrl_text: "str | None") -> dict:
    """XBRLから保有目的・取得資金・保有株数・報告義務発生日を取り出す。

    Returns: {"purpose_of_holding": str|None, "important_proposal": str|None,
              "shares_held": int|None, "shares_outstanding": int|None,
              "funding_total": int|None, "funding_own": int|None,
              "funding_borrowings": int|None, "obligation_date": str|None}
    """
    empty = {"purpose_of_holding": None, "important_proposal": None,
             "shares_held": None, "shares_outstanding": None,
             "funding_total": None, "funding_own": None,
             "funding_borrowings": None, "obligation_date": None}
    if not xbrl_text:
        return empty

    purpose = _join_member_texts(_tag_by_context(xbrl_text, "PurposeOfHolding", text=True))
    proposal = _join_member_texts(
        _tag_by_context(xbrl_text, "ActOfMakingImportantProposalEtc", text=True))
    obligation = _tag_by_context(xbrl_text, "DateWhenFilingRequirementAroseCoverPage")
    obligation_date = None
    for value in obligation.values():
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            obligation_date = value
            break

    return {
        "purpose_of_holding": purpose,
        "important_proposal": proposal,
        "shares_held": _aggregate_or_sum(_tag_by_context(xbrl_text, "TotalNumberOfStocksEtcHeld")),
        "shares_outstanding": _aggregate_or_sum(
            _tag_by_context(xbrl_text, "TotalNumberOfOutstandingStocksEtc")),
        "funding_total": _sum_over_members(
            _tag_by_context(xbrl_text, "TotalAmountOfFundingForAcquisition")),
        "funding_own": _sum_over_members(_tag_by_context(xbrl_text, "AmountOfOwnFund")),
        "funding_borrowings": _sum_over_members(
            _tag_by_context(xbrl_text, "TotalAmountOfBorrowings")),
        "obligation_date": obligation_date,
    }


# 保有目的の分類。EDINETの保有目的欄は自由記述なので、株主としての姿勢が
# 強い順（重要提案行為等 > 経営参加 > 政策保有 > 純投資 > 安定株主）に判定する。
# 「重要提案行為等を行う可能性もある」という留保付きの記載も、実際に経営へ
# 関与しうる立場を宣言している点で純投資とは区別する。
_PURPOSE_RULES = [
    ("重要提案行為等", re.compile(r'重要提案行為|重要な提案|株主提案|企業価値の向上を目的として')),
    ("経営参加", re.compile(r'経営参加|経営に参画|経営権|子会社化|完全子会社|経営支援|役員として')),
    ("政策保有", re.compile(r'政策投資|政策保有|取引関係|業務提携|資本提携|関係強化|取引の維持|グループ間取引')),
    ("安定株主", re.compile(r'安定株主|長期保有|長期安定|経営の安定|継続保有')),
    # 運用会社・証券会社の「投資一任」「投資信託」「ディーリング」等はどれも
    # 経営に関与しない保有なので純投資に寄せる（直近553件で判定不可を77→36件に減らした）。
    ("純投資", re.compile(r'純投資|投資収益|値上がり|運用を目的|商品在庫|信託契約|投資一任|投資信託'
                       r'|信託財産|ディーリング|トレーディング|一時保有|一時的保有|資産運用')),
]


def classify_purpose(purpose_text: "str | None") -> "str | None":
    """保有目的の自由記述を5区分（重要提案行為等/経営参加/政策保有/安定株主/純投資）に寄せる。

    どれにも当たらなければ None（サイト側で「その他」表示にする）。
    """
    if not purpose_text:
        return None
    for label, pattern in _PURPOSE_RULES:
        if pattern.search(purpose_text):
            return label
    return None


def average_acquisition_price(funding_total: "int | None",
                              shares_held: "int | None") -> "float | None":
    """開示ベースの平均取得単価（円/株）= 取得資金の総額 ÷ 保有株数。

    取得資金の総額は「現在保有している分」の取得原価なので、政策保有株のように
    保有が古いほど現在株価から乖離する（それ自体が含み損益の手がかりになる）。
    株数0（全部売却済み）や資金額の記載が無い開示では None。
    """
    if not funding_total or not shares_held or shares_held <= 0:
        return None
    return round(funding_total / shares_held, 1)


# ---- 短期大量譲渡（法第27条の25第2項）の「譲渡の相手方・単価」----
# 短期大量譲渡に該当する変更報告書には、直近60日間の取得・処分が1件ずつ表になっており、
# 「誰に売ったか（譲渡の相手方）」と「いくらで売ったか（単価）」が開示されている。
# EDINETは保有"比率"しか出さないため金額は株価からの概算に頼っているが、この表がある
# 開示に限っては単価×数量で実額が出せる（実例: 日立製作所→日立建機 9.98%売却は
# 開示日終値ベースだと1,274.9億円だが、実際は5,227円×21,462,310株＝約1,122億円）。
_TRANSFER_BLOCK_RE = re.compile(
    r'ShortTermLargeVolumeTransferTextBlock[^>]*>(.*?)</[^>]*ShortTermLargeVolumeTransferTextBlock>',
    re.S,
)
_TR_RE = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S | re.I)
# 空セルは <td class="COL_L"/> と自己終了で書かれる（相手方・単価が無い行）。
# 拾い落とすと列がずれるので、自己終了タグも1セルとして数える。
_TD_RE = re.compile(r'<t[dh][^>]*/>|<t[dh][^>]*>(.*?)</t[dh]>', re.S | re.I)
_JP_ERA_RE = re.compile(r'(令和|平成)(\d+|元)年(\d+)月(\d+)日')
_WESTERN_DATE_RE = re.compile(r'(\d{4})年(\d+)月(\d+)日')
_ERA_START = {"令和": 2018, "平成": 1988}  # 元年 = start + 1


def _unescape(text: str) -> str:
    return html_lib.unescape(text).replace("\u00a0", " ")


def _cell_text(cell_html: str) -> str:
    """テーブルセルのHTML（入れ子のspan/div/brを含む）からプレーンテキストを取り出す。"""
    return re.sub(r'\s+', ' ', _unescape(re.sub(r'<[^>]*>', ' ', cell_html))).strip()


def _parse_jp_date(text: str) -> "str | None":
    """「2026年8月19日」「令和8年7月15日」を ISO 形式（YYYY-MM-DD）で返す。"""
    m = _WESTERN_DATE_RE.search(text)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        m = _JP_ERA_RE.search(text)
        if not m:
            return None
        era, yy, mo, d = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        y = _ERA_START[era] + (1 if yy == "元" else int(yy))
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def _parse_number(text: str) -> "float | None":
    """「21,462,310」「5,227」「4,730円」を数値で返す。「借株」等の非数値は None。"""
    m = re.search(r'[0-9][0-9,]*(?:\.[0-9]+)?', text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# テキストブロックのヘッダ行はEDINETの様式で固定。列順が変わっても拾えるよう
# ヘッダ文字列→キーの対応で引く（見出しが改行やdivで割れるため部分一致で判定する）。
_TRANSFER_COLUMNS = [
    ("年月日", "date"),
    ("種類", "security_type"),
    ("数量", "shares"),
    ("割合", "ratio"),
    ("市場内外", "venue"),
    ("処分の別", "action"),
    ("相手方", "counterparty"),
    ("単価", "unit_price"),
]


def parse_short_term_transfers(xbrl_text: "str | None") -> list:
    """短期大量譲渡の「最近60日間の取得又は処分の状況」表を行ごとの dict にして返す。

    Returns: [{"date": "2026-08-19", "security_type": "普通株式", "shares": 21462310,
               "ratio": 9.98, "venue": "市場外", "action": "処分",
               "counterparty": "SMBC日興証券株式会社", "unit_price": 5227.0,
               "unit_price_note": None}, ...]
    表が無い開示（短期大量譲渡以外）では [] を返す。
    unit_price_note は「借株」のように単価欄が金額でない場合の原文。
    """
    if not xbrl_text:
        return []
    m = _TRANSFER_BLOCK_RE.search(xbrl_text)
    if not m:
        return []
    # テキストブロックの中身はエスケープされたHTML（&lt;tr&gt;...）で入っている
    block = _unescape(m.group(1))
    rows = []
    col_index = None
    for tr in _TR_RE.findall(block):
        cells = [_cell_text(mt.group(1) or "") for mt in _TD_RE.finditer(tr)]
        if not cells:
            continue
        if any("年月日" in c for c in cells) and any("単価" in c for c in cells):
            # ヘッダ行。列位置を確定する
            col_index = {}
            for i, cell in enumerate(cells):
                for needle, key in _TRANSFER_COLUMNS:
                    if needle in cell and key not in col_index:
                        col_index[key] = i
            continue
        if col_index is None:
            continue
        get = lambda key: (cells[col_index[key]] if key in col_index and col_index[key] < len(cells) else "")
        unit_price_raw = get("unit_price")
        unit_price = _parse_number(unit_price_raw)
        counterparty = get("counterparty") or None
        row = {
            "date": _parse_jp_date(get("date")),
            "security_type": get("security_type") or None,
            "shares": int(_parse_number(get("shares")) or 0) or None,
            "ratio": _parse_number(get("ratio")),
            "venue": get("venue") or None,
            "action": get("action") or None,
            "counterparty": counterparty,
            "unit_price": unit_price,
            # 「借株」「貸株」等、単価欄に金額でない語が入る行がある（Evo Fund等の空売り絡み）
            "unit_price_note": unit_price_raw if (unit_price is None and unit_price_raw) else None,
        }
        if row["shares"] or row["counterparty"] or row["unit_price"]:
            rows.append(row)
    return rows


# 単価が「株価」と言えるのは株券（株式）の行だけ。新株予約権証券・社債券・預託証券等は
# 1個あたりの価格なので、株式の売却金額として扱わない。
_NON_EQUITY_RE = re.compile(r'新株予約権|社債|預託証券|受益証券|カバードワラント|転換')
_EQUITY_RE = re.compile(r'株券|株式|普通株')


def _is_equity_row(row: dict) -> bool:
    kind = row.get("security_type") or ""
    return bool(_EQUITY_RE.search(kind)) and not _NON_EQUITY_RE.search(kind)


def _is_unknown_counterparty(name: "str | None") -> bool:
    """「市場内取引のため不明」「単元未満株式の売却のため不明」「該当なし」等、
    相手方が特定できない表記。"""
    return not name or "不明" in name or "該当なし" in name


def summarize_disposals(transfers: list, ratio_change: "float | None" = None) -> dict:
    """短期大量譲渡の行から「誰にいくらで売ったか」を集計する。

    Args:
        transfers:    parse_short_term_transfers() の戻り値。
        ratio_change: 今回の開示で動いた保有比率（ポイント、絶対値）。渡すと
                      「表の処分だけで今回の変化を説明できるか」を検証し、
                      説明できる場合のみ amount_oku（実額）を返す。

    Returns: {"counterparties": [str], "amount_oku": float|None, "shares": int,
              "unit_price": float|None, "security_type": str|None, "is_equity": bool,
              "date": str|None, "venue": str|None, "rows": [dict]}
             rows は相手方が特定できた処分行のみ。
    """
    disposals = [t for t in transfers if t.get("action") == "処分"]
    acquisitions = [t for t in transfers if t.get("action") == "取得"]
    priced = [t for t in disposals if t.get("unit_price") and t.get("shares")]
    named = [t for t in disposals if not _is_unknown_counterparty(t.get("counterparty"))]

    # 実額に使えるのは株式（株券）の譲渡だけ。新株予約権・社債等は単価が株価ではないため、
    # 株式の売却金額として並べると桁が違う（実例: シェアレコ→日本文化数寄財団は新株予約権
    # 191万個×1円で0.0億円になり、株式換算の概算20.1億円とは別物）。
    equity = [t for t in priced if _is_equity_row(t)]
    shares_sum = sum(t["shares"] for t in equity)
    # 実額として採用するのは「表の処分行だけで今回の比率変化が説明できる」ときだけ。
    # 60日間の売買が細かく並ぶ開示（取得と処分が混在）は差引きが表から復元できないため、
    # 従来どおり株価からの概算を使う。
    amount_oku = None
    used = []
    if equity and not acquisitions:
        ratio_sum = sum(t.get("ratio") or 0 for t in equity)
        if ratio_change is None or abs(ratio_sum - abs(ratio_change)) <= 0.5:
            used = equity
        else:
            # 表は直近60日ぶんなので、連続して売った提出者の2枚目以降には前回開示済みの行も
            # 並ぶ。今回の変化幅と一致する行が1つだけならその行が今回の譲渡（実例: 三井金属は
            # 8/3に18.85%、8/4に7.3%を別々の相手へ譲渡し、2枚目の表に両方載る）。
            match = [t for t in equity if abs((t.get("ratio") or 0) - abs(ratio_change)) <= 0.2]
            if len(match) == 1:
                used = match
        amount_yen = sum(t["unit_price"] * t["shares"] for t in used)
        # 0.05億円未満は四捨五入で「0.0億円」になり取引が無かったように見えるため採らない
        if amount_yen >= 5e6:
            amount_oku = round(amount_yen / 1e8, 1)
        else:
            used = []

    # 単価・日付・市場内外は最大株数の行（＝その開示の主役の取引）を代表値にする。
    # 単元未満株の端数処分が同じ表に並ぶため、先頭行を採るとそちらを拾ってしまう。
    main = max(used or priced or named or [{}], key=lambda t: t.get("shares") or 0)
    # 実額の内訳が1行に確定したときは、その相手方だけを見せる（表に載る過去の相手方を
    # 今回の譲渡先として並べない）
    shown = [t for t in (used or named) if not _is_unknown_counterparty(t.get("counterparty"))] or named
    return {
        "counterparties": list(dict.fromkeys(t["counterparty"] for t in shown)),
        "amount_oku": amount_oku,
        "shares": sum(t["shares"] for t in used) if used else shares_sum,
        "unit_price": main.get("unit_price"),
        # 新株予約権・社債等は「1株いくら」ではないので、表示側で単位を出し分けるために返す
        "security_type": main.get("security_type"),
        "is_equity": _is_equity_row(main),
        "date": main.get("date"),
        "venue": main.get("venue"),
        "rows": shown,
    }


# 報告書種別はdoc_type_codeだけでは判別できない（350は新規・変更の両方を含み、360は訂正）。
# doc_descriptionの接頭辞で判定し、descriptionが無い行のみdoc_type_codeにフォールバックする。
# kujira-watch/src/lib/disclosures.ts の disclosureKindLabel() と同一ロジック。
_DOC_LABEL_BY_KIND = {"新規": "大量保有報告書", "変更": "変更報告書", "訂正": "訂正報告書"}


def disclosure_kind_label(doc_description: "str | None", doc_type_code: str) -> str:
    """報告書種別を「新規」「変更」「訂正」のいずれかで返す。"""
    desc = doc_description or ""
    if desc.startswith("訂正"):
        return "訂正"
    if desc.startswith("変更報告書"):
        return "変更"
    if desc.startswith("大量保有報告書"):
        return "新規"
    return "訂正" if str(doc_type_code) == "360" else "新規"


def disclosure_doc_label(doc_description: "str | None", doc_type_code: str) -> str:
    """報告書種別の正式名称（大量保有報告書/変更報告書/訂正報告書）を返す。"""
    return _DOC_LABEL_BY_KIND[disclosure_kind_label(doc_description, doc_type_code)]


def extract_large_holdings(results: list, disc_date: str) -> list:
    """documents.json の results から大量保有報告書（350/360）のみ抽出して整形。

    Returns: list of dict（doc_id, filer_name, doc_type_code,
             doc_description, submit_date, disc_date）。
    """
    records = []
    for r in results:
        if str(r.get("docTypeCode", "")) not in _LARGE_HOLDING_TYPES:
            continue
        records.append({
            "doc_id": r.get("docID"),
            "filer_name": r.get("filerName"),
            "doc_type_code": str(r.get("docTypeCode", "")),
            "doc_description": r.get("docDescription"),
            "submit_date": r.get("submitDateTime"),
            "disc_date": disc_date,
            "holding_ratio": None,        # XBRL取得後に埋める
            "holding_ratio_prior": None,  # XBRL取得後に埋める
            "issuer_code": None,          # XBRL取得後に埋める
            "short_term_transfers": [],   # XBRL取得後に埋める（短期大量譲渡のみ）
        })
    return [x for x in records if x["doc_id"]]


def scan_large_holdings(days_back: int = 7, persist: bool = True,
                        start_date: "str | None" = None, end_date: "str | None" = None,
                        skip_weekends: bool = True, sleep_sec: float = 0.0,
                        fetch_xbrl: bool = True) -> list:
    """直近 days_back 日分（または start_date 以降）の大量保有報告書をスキャンしてDB蓄積。

    Args:
        days_back:     何日前まで遡るか（当日含む）。start_date 指定時は無視。
        persist:       True なら edinet_large_holdings テーブルへ upsert。
        start_date:    'YYYY-MM-DD'。指定するとこの日から end_date（既定は当日）までを全て走査（バックフィル用）。
        end_date:      'YYYY-MM-DD'。走査の終端。既に蓄積済みの期間を再走査しないために使う。
        skip_weekends: 土日はEDINET提出が無いのでスキップ（API呼び出し削減）。
        sleep_sec:     各日リクエスト間の待機（バックフィルでEDINETに優しく）。
        fetch_xbrl:    True なら XBRL本文から保有割合を取得。Falseならメタデータのみ。

    Returns: 取得した全レコード（dict のリスト）。
    """
    import time
    from lib.db import HOLDING_DETAIL_COLUMNS, upsert_edinet_large_holdings

    today = date.today()
    if start_date:
        d0 = date.fromisoformat(start_date)
        d1 = date.fromisoformat(end_date) if end_date else today
        dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    else:
        dates = [today - timedelta(days=i) for i in range(days_back)]
    if skip_weekends:
        dates = [d for d in dates if d.weekday() < 5]

    all_records = []
    for d in dates:
        ds = d.isoformat()
        results = fetch_documents_list(ds)
        recs = extract_large_holdings(results, disc_date=ds)
        if fetch_xbrl:
            for rec in recs:
                details = fetch_xbrl_details(rec["doc_id"])
                rec["holding_ratio"] = details["holding_ratio"]
                rec["holding_ratio_prior"] = details["holding_ratio_prior"]
                rec["issuer_code"] = details["issuer_code"]
                rec["issuer_name"] = details.get("issuer_name")
                rec["short_term_transfers"] = details.get("short_term_transfers") or []
                for key in HOLDING_DETAIL_COLUMNS:
                    rec[key] = details.get(key)
                if not rec["issuer_code"]:
                    print(f"    ⚠ issuer_code取得失敗: {rec['doc_id']} filer={rec.get('filer_name')}")
                if sleep_sec:
                    time.sleep(sleep_sec)
            recs = [r for r in recs if r.get("issuer_code")]
        if recs and persist:
            upsert_edinet_large_holdings(recs)
        all_records.extend(recs)
        if sleep_sec:
            time.sleep(sleep_sec)
    return all_records


# ---------------------------------------------------------------------------
# 記事とEDINET開示の突合（提出者の一意化）
# ---------------------------------------------------------------------------
# 記事側のツール（事実カード書き出し・薄い記事のリライト・誤報の是正）が共通で使う。
# tools配下に置くと、互いにimportし合って循環参照になるためここに集約している。

def _norm(name: str) -> str:
    """提出者名の突合用の正規化。EDINETのXBRLは提出者名を全角（Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…）で
    保持する一方、記事側のfilerNameや本文は半角で入ることがあるため、NFKC正規化して空白を落とす。"""
    return unicodedata.normalize("NFKC", name or "").replace(" ", "").replace("\u3000", "").lower()


def resolve_filer(article: dict, rows: list) -> "str | None":
    """同一銘柄・同一開示日に提出者が複数いる場合の一意化。

    銘柄コード×開示日だけでは絞れない記事が全体の18%（実測2026-08-25: 999件中182件）あり、
    そのままではリライトの材料が作れない。記事側が持つ情報で候補を絞る:
      1. microCMSのfilerNameが候補と一致すればそれを採る（154件がこの経路で解決する）
      2. 記事タイトルに候補の提出者名が含まれていればそれを採る
    どちらでも決まらなければNone（＝材料を作らずスキップ）。誤った提出者で記事を書き直すと
    別の投資家の取引として公開されることになるため、曖昧なままでは進めない。
    """
    names = {r["filer_name"] for r in rows if r.get("filer_name")}
    if len(names) == 1:
        return names.pop()
    if not names:
        return None

    by_norm = {_norm(n): n for n in names}
    filer_name = (article.get("filerName") or "").strip()
    if filer_name:
        hit = by_norm.get(_norm(filer_name))
        if hit:
            return hit

    title = _norm(article.get("title") or "")
    matches = [n for norm, n in by_norm.items() if norm and norm in title]
    if len(matches) == 1:
        return matches[0]
    return None


def _load_ledger(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
