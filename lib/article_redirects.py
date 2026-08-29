"""削除した記事のURLを、内容の近い生存ページへ引き継ぐためのリダイレクト表。

なぜ必要か（2026-08-29のGSC実測）:
  検索結果に出ているURL 924件のうち194件が本番で404を返しており、そこに28日で25クリック
  （全クリックの18%）が着地していた。うち124件は削除済みの記事URL。低価値・重複・誤報の記事を
  消すたびに、順位が付いていたURLが「ページが見つかりません」に変わっていた。
  削除自体は続けるべき（薄い記事はテンプレート全体の評価を下げる）が、**URLは捨てない**。

引き継ぎ先:
  - 重複削除 … 残した方の記事 `/articles/<残ったid>`（同じ開示を扱っているので等価）
  - それ以外 … その銘柄のページ `/stocks/<証券コード>`（同じ開示の一覧・大株主・会社情報がある）
  銘柄コードが取れない記事はリダイレクト先が無いので記録しない（404のままにする）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client

TABLE = "deleted_article_redirects"


def stock_target(article: dict) -> "str | None":
    """記事の銘柄ページへのパス。銘柄コードが無ければNone。"""
    code = str(article.get("stockCode") or article.get("stock_code") or "").strip()
    return f"/stocks/{code}" if code else None


def article_target(surviving_id: str) -> str:
    """重複削除で残した方の記事へのパス。"""
    return f"/articles/{surviving_id}"


def record(article_id: str, target_path: str, reason: str) -> bool:
    """削除した記事1件のリダイレクトを登録する。"""
    return record_many([{"article_id": article_id, "target_path": target_path, "reason": reason}])


def record_many(rows: list) -> bool:
    """まとめて登録する。article_idが空・target_pathが空の行は捨てる。"""
    cleaned = [
        {"article_id": str(r["article_id"]), "target_path": r["target_path"], "reason": r.get("reason", "")}
        for r in rows
        if r.get("article_id") and r.get("target_path")
    ]
    if not cleaned:
        return True
    ok = supabase_client.upsert(TABLE, cleaned, on_conflict="article_id")
    # 今回消した記事を指していた既存の行を、新しい行き先に付け替える（301の連鎖を作らない）。
    # 例: A→B（重複削除）の後にBを消すと、A→B→C の2ホップになりGoogleは評価を減衰させる。
    for r in cleaned:
        _flatten_chain_to(r["article_id"], r["target_path"])
    return ok


def _flatten_chain_to(deleted_id: str, new_target: str) -> None:
    stale = f"/articles/{deleted_id}"
    supabase_client.update(TABLE, f"target_path=eq.{stale}", {"target_path": new_target})


def lookup(article_id: str) -> "str | None":
    """登録済みのリダイレクト先。無ければNone（Web側の実装はTypeScript側にある。
    こちらはバックフィルの重複確認・調査用）。"""
    row = supabase_client.select_one(TABLE, f"article_id=eq.{article_id}&select=target_path")
    return (row or {}).get("target_path")
