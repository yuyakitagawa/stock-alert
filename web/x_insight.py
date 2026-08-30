"""
web/x_insight.py

X投稿の「解釈行」（提出者の文脈を1行足す）に使うデータを取る。

事実（誰が何%取得した）だけの自動投稿はbot扱いされてフォローされない、というのが
インフルエンサー1000人コンサル（docs/x_post_improvement_1000.md 施策9）の指摘。
一方、以前あった「乗っかり実績」(filer_win_rate)は推定損益の算出が誤っていたため
2026-08-18にテーブルごと廃止しているので、そこは使わない。代わりに検証の要らない
事実（この提出者のEDINET開示件数・この銘柄での報告回数）だけで文脈を作る。

もう1本、開示日時点のバリュエーション（PBR・ROE・配当性向）を出す関数を持つ。
アクティビスト・資本コストのテーマで読まれているアカウントに対して、こちらが常に
勝てるのは「全開示に必ず開示時点の数字が付く」ことだけなので、そこを型にする。

Supabase未設定・取得失敗時は空dictを返し、呼び出し側はその行なしで投稿する。
"""
from urllib.parse import quote

from lib import supabase_client as sb


def fetch_filer_context(filer_name: str, issuer_code: str) -> dict:
    """{filer_disclosures, stock_disclosures} を返す。取得できない場合は空dict。
    filer_disclosures: その提出者のEDINET大量保有関連の開示件数（edinet_filer_summary）
    stock_disclosures: その提出者×この銘柄の開示件数（今回を含む）"""
    if not filer_name or not sb.is_configured():
        return {}
    try:
        summary = sb.select_one(
            "edinet_filer_summary",
            f"filer_name=eq.{quote(filer_name, safe='')}&select=holding_count",
        )
        rows = sb.select(
            "edinet_large_holdings",
            f"filer_name=eq.{quote(filer_name, safe='')}"
            f"&issuer_code=eq.{quote(issuer_code or '', safe='')}&select=doc_id",
        ) if issuer_code else []
        return {
            "filer_disclosures": int((summary or {}).get("holding_count") or 0),
            "stock_disclosures": len(rows),
        }
    except Exception as e:
        print(f"  ⚠ 解釈行用の提出者データ取得に失敗: {e}")
        return {}


def build_insight_line(stock_name: str, context: dict) -> str:
    """解釈行を組み立てる。文脈が薄い（データが無い）場合は空文字＝行を出さない。"""
    if not context:
        return ""
    filer_n = context.get("filer_disclosures") or 0
    stock_n = context.get("stock_disclosures") or 0
    if filer_n <= 1:
        return "この提出者がEDINETに登場するのは初"
    if stock_n >= 2:
        return f"この提出者の開示は過去{filer_n}件、{stock_name}では{stock_n}回目"
    return f"この提出者の開示は過去{filer_n}件、{stock_name}は初"


# ---------------------------------------------------------------- 開示時点のバリュエーション
# 「誰が何%買った」だけの投稿は事実の羅列で、同じテーマ（アクティビスト・資本コスト・
# 非公開化）を扱う既存アカウントに対して読む理由が作れない。こちらの強みは全開示に
# point-in-timeの数字を必ず添えられることなので、開示日時点のPBR・ROE・配当性向を1行足す。
#
# PIT規律（CLAUDE.md）: 株価指標は開示日以前で最新の gen_rankings 行（記事本文の
# build_context_facts と同じ取り方）、ROEは開示日以前で最新の**本決算(FY)**行から
# np/equity で算出する。四半期行のnpは累計利益なのでROEとして出すと過小になるため使わない。
# 「割安」「狙われやすい」等の評価的な語は付けない（docs/x_operation_rules.md 2）。

# 異常値を投稿に出さないための範囲。範囲外は「取れなかった」扱いにする。
PBR_RANGE = (0.01, 50.0)
ROE_RANGE = (-99.9, 99.9)
PAYOUT_MAX_PCT = 300.0


def fetch_valuation_context(issuer_code: str, disc_date: str) -> dict:
    """開示日時点の {pbr, roe, payout_pct} を返す。取れない項目はキーごと落とす。
    disc_dateは 'YYYY-MM-DD'（先頭10文字だけ使う）。Supabase未設定・失敗時は空dict。"""
    code = (issuer_code or "").strip()
    day = (disc_date or "")[:10]
    if not code or not day or not sb.is_configured():
        return {}
    out = {}
    try:
        px = sb.select_one(
            "gen_rankings",
            f"select=date,pbr&code=eq.{quote(code, safe='')}&pbr=not.is.null"
            f"&date=lte.{quote(day, safe='')}&order=date.desc",
        )
        pbr = (px or {}).get("pbr")
        if pbr is not None and PBR_RANGE[0] <= float(pbr) <= PBR_RANGE[1]:
            out["pbr"] = float(pbr)

        fy = sb.select_one(
            "jquants_fin_summary",
            f"select=disc_date,np,equity,payout_ratio&code=eq.{quote(code, safe='')}"
            f"&doc_type=eq.FY&disc_date=lte.{quote(day, safe='')}&order=disc_date.desc",
        ) or {}
        np_v, equity = fy.get("np"), fy.get("equity")
        if np_v is not None and equity and float(equity) > 0:
            roe = float(np_v) / float(equity) * 100
            if ROE_RANGE[0] <= roe <= ROE_RANGE[1]:
                out["roe"] = roe
        pr = fy.get("payout_ratio")
        if pr is not None:
            # payout_ratioは行によって比率(0.2)と%(20.0)が混在する。lib/fundamentalsと同じ基準で揃える。
            pct = float(pr) * 100 if float(pr) <= 1.5 else float(pr)
            if 0 <= pct <= PAYOUT_MAX_PCT:
                out["payout_pct"] = pct
    except Exception as e:
        print(f"  ⚠ 開示時点のバリュエーション取得に失敗（投稿は続行）: {e}")
        return {}
    return out


def build_valuation_line(valuation: dict) -> str:
    """開示時点のバリュエーション1行。PBRが取れない銘柄（REIT等）は行を出さない
    ——ROE単独だと資本コストの文脈にならず、行を足す意味が無いため。"""
    if not valuation or valuation.get("pbr") is None:
        return ""
    parts = [f"PBR{valuation['pbr']:.2f}倍"]
    if valuation.get("roe") is not None:
        parts.append(f"ROE{valuation['roe']:.1f}%")
    if valuation.get("payout_pct") is not None:
        parts.append(f"配当性向{valuation['payout_pct']:.0f}%")
    return "開示時点 " + "・".join(parts)
