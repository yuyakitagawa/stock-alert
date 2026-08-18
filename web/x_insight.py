"""
web/x_insight.py

X投稿の「解釈行」（提出者の文脈を1行足す）に使うデータを取る。

事実（誰が何%取得した）だけの自動投稿はbot扱いされてフォローされない、というのが
インフルエンサー1000人コンサル（docs/x_post_improvement_1000.md 施策9）の指摘。
一方、以前あった「乗っかり実績」(filer_win_rate)は推定損益の算出が誤っていたため
2026-08-18にテーブルごと廃止しているので、そこは使わない。代わりに検証の要らない
事実（この提出者のEDINET開示件数・この銘柄での報告回数）だけで文脈を作る。

Supabase未設定・取得失敗時は空dictを返し、呼び出し側は解釈行なしで投稿する。
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
