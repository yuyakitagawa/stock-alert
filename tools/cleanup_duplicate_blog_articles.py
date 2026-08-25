"""microCMSブログの重複記事クリーンアップ。

同一銘柄・同一開示日・同一提出者(filerName)・同一比率変化幅(ratioChangePct)の記事が
複数あれば、最初に公開された1件を残して後発を削除する（先発はX投稿等で既にリンクされて
いる可能性が高いため）。突き合わせキーは publish_blog_articles.already_published() と
同一にする（同一提出者が同日に複数の報告書を出す実例＝2936 2025-08-13 橋本舜2件 を
別イベントとして残すため、ratioChangePctまで一致した場合のみ重複と見なす）。
重複は概算金額の株価再計算ブレでalready_published()をすり抜けた場合に発生する
（実運用で発生: 2026-08-17、開示当日の終値が価格キャッシュに入る前後で金額が変わり
同一開示が多数二重投稿された）。filerName送信開始(2026-08-15)前の旧記事は
filerNameが空で突き合わせできないため対象外。

edinet_blog.ymlの投稿ステップ後に毎回 --delete で実行され、すり抜けた重複を
自動回収する（microCMS API失敗時にalready_published()がFalseを返す設計のため、
重複は今後も稀に発生しうる）。

実行: python3 tools/cleanup_duplicate_blog_articles.py [--days 3] [--delete]
      --delete 無しはdry-run（削除対象の表示のみ）
"""
import argparse
import os
import sys
from datetime import date, timedelta

import requests

BUYBACK_TAG = "自社株買い"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.publish_blog_articles import _microcms_base_url, _microcms_headers, MICROCMS_DOMAIN, MICROCMS_KEY


def fetch_recent_articles(days: int) -> list:
    """直近days日にdealDateを持つ記事を全件取得（100件ずつページング）。"""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    articles, offset = [], 0
    while True:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"dealDate[greater_than]{cutoff}",
                "fields": "id,title,stockCode,dealDate,dealAmount,filerName,ratioChangePct,createdAt,tags",
                "limit": 100,
                "offset": offset,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        articles.extend(data.get("contents", []))
        offset += 100
        if offset >= data.get("totalCount", 0):
            return articles


def is_buyback(article: dict) -> bool:
    """自社株買い記事（web/publish_buyback_articles.py が投稿）かどうか。

    dealTypeでは判定できない。microCMSのdealTypeはセレクト型で、選択肢に無い値をPOSTしても
    エラーにならず空配列で保存されるため、「自社株買い」は全記事で空になっている。
    テキスト型のtagsには保存されているので、そちらで見る。
    """
    return BUYBACK_TAG in (article.get("tags") or "")


def find_duplicates(articles: list) -> list:
    """同一開示から作られた記事をグループ化し、各グループの先発1件を除いた削除対象を返す。

    突き合わせキーは記事の種類で分ける:
      - 大量保有報告書: (stockCode, 開示日, filerName, ratioChangePct)
        filerNameが空の記事（filerName送信開始=2026-08-15より前の旧仕様）は判定不能のため対象外。
      - 自社株買い: (stockCode, 開示日)
        提出者は発行体自身でfilerNameを持たないため、旧実装では丸ごと対象外になっていた。
        その結果 already_published() のすり抜け（dealTypeで既報判定していた不具合）を
        回収できず、同一開示から13本の重複記事が公開された（2026-08-25、コンヴァノ6574）。
        同じ銘柄が同じ日に決定開示を2本出すことは実務上ないので、銘柄×日付で重複と見なす。
    """
    groups = {}
    for a in articles:
        if is_buyback(a):
            key = ("buyback", a.get("stockCode"), str(a.get("dealDate", ""))[:10])
        elif a.get("filerName"):
            ratio = a.get("ratioChangePct")
            key = (a.get("stockCode"), str(a.get("dealDate", ""))[:10], a["filerName"],
                   round(ratio, 2) if ratio is not None else None)
        else:
            continue
        groups.setdefault(key, []).append(a)
    to_delete = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda a: a.get("createdAt", ""))
        to_delete.extend(members[1:])
    return to_delete


def delete_article(content_id: str) -> bool:
    resp = requests.delete(f"{_microcms_base_url()}/{content_id}", headers=_microcms_headers(), timeout=20)
    return resp.status_code in (200, 202)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3, help="dealDateを遡る日数")
    p.add_argument("--delete", action="store_true", help="実際に削除する（無指定はdry-run）")
    args = p.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("[cleanup_duplicate_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return

    articles = fetch_recent_articles(args.days)
    dups = find_duplicates(articles)
    if not dups:
        print(f"重複なし（直近{args.days}日・{len(articles)}記事を確認）")
        return
    for a in dups:
        label = (f"{a.get('stockCode')} {str(a.get('dealDate', ''))[:10]} {a.get('filerName')} "
                 f"({a.get('ratioChangePct')}pt) → id={a['id']}")
        if not args.delete:
            print(f"  [dry-run] 削除対象: {label}")
        elif delete_article(a["id"]):
            print(f"  🗑 削除: {label}")
        else:
            print(f"  ⚠ 削除失敗: {label}")
    print(f"重複{len(dups)}件（直近{args.days}日・{len(articles)}記事中）{'を削除しました' if args.delete else '。--delete で削除実行'}")


if __name__ == "__main__":
    main()
