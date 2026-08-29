"""microCMSブログの重複記事クリーンアップ。

同一開示から二重に投稿された記事を、最初に公開された1件だけ残して削除する
（先発はX投稿等で既にリンクされている可能性が高いため）。

突き合わせキーは記事の世代で分ける:

- filerName入り（2026-08-15以降の記事）
  銘柄コード＋開示日＋提出者名＋比率変化幅(ratioChangePct)。
  publish_blog_articles.already_published() と同一キーにする（同一提出者が同日に
  複数の報告書を出す実例＝2936 2025-08-13 橋本舜2件 を別イベントとして残すため、
  ratioChangePctまで一致した場合のみ重複と見なす）。
- filerName無し（2026-08-15より前の旧記事）
  銘柄コード＋開示日＋タイトル。旧記事は突き合わせキーになるフィールドを持たないが、
  タイトルに提出者名と比率が入るため、同一銘柄・同一開示日でタイトルまで一致すれば
  同一開示の重複と判定できる。この世代は already_published() が概算金額(dealAmount)
  ±0.05億円でしか突き合わせておらず、開示当日の終値が価格キャッシュに入る前後で金額が
  変わるたびに同じ開示が再投稿されたため、1開示あたり10件を超える重複が残っている
  （実例: 9706 日本空港ビルデングの /stocks/9706 に同一記事が11件）。
- 自社株買い（tagsに「自社株買い」を含む記事）
  銘柄コード＋開示日。提出者は発行体自身でfilerNameを持たないため、世代に関わらず
  この形で突き合わせる（同じ銘柄が同じ日に決定開示を2本出すことは実務上ない）。

重複は概算金額の株価再計算ブレでalready_published()をすり抜けた場合に発生する
（実運用で発生: 2026-08-17、開示当日の終値が価格キャッシュに入る前後で金額が変わり
同一開示が多数二重投稿された）。

edinet_blog.ymlの投稿ステップ後に毎回 --delete で実行され、すり抜けた重複を
自動回収する（microCMS API失敗時にalready_published()がFalseを返す設計のため、
重複は今後も稀に発生しうる）。日次のCIは直近30日だけを見るので、それより古い世代の
重複は --all を付けた手動実行で回収する。

実行: python3 tools/cleanup_duplicate_blog_articles.py [--days 3 | --all] [--code 9706] [--delete]
      --delete 無しはdry-run（削除対象の表示のみ）
"""
import argparse
import os
import sys
from datetime import date, timedelta

import requests

BUYBACK_TAG = "自社株買い"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import article_redirects
from web.publish_blog_articles import _microcms_base_url, _microcms_headers, MICROCMS_DOMAIN, MICROCMS_KEY


ARTICLE_FIELDS = "id,title,stockCode,dealDate,dealAmount,filerName,ratioChangePct,createdAt,tags"


def fetch_articles(days: "int | None", code: "str | None" = None) -> list:
    """記事を全件取得（100件ずつページング）。

    days=None は全期間。dealDateでの絞り込みを付けないので、dealDateが空の記事も拾える。
    daysを指定すると直近days日にdealDateを持つ記事だけを見る（日次CI用）。"""
    filters = []
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        filters.append(f"dealDate[greater_than]{cutoff}")
    if code:
        filters.append(f"stockCode[equals]{code}")

    articles, offset = [], 0
    while True:
        params = {"fields": ARTICLE_FIELDS, "limit": 100, "offset": offset}
        if filters:
            params["filters"] = "[and]".join(filters)
        resp = requests.get(
            _microcms_base_url(), headers=_microcms_headers(), params=params, timeout=20,
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


def duplicate_key(article: dict) -> tuple:
    """記事の種類・世代に応じた重複判定キー。自社株買いは銘柄コード＋開示日、
    filerNameを持つ新記事は開示データそのものの値で、持たない旧記事はタイトルで
    突き合わせる（モジュールdocstring参照）。"""
    stock_code = article.get("stockCode")
    deal_date = str(article.get("dealDate", ""))[:10]
    if is_buyback(article):
        return ("buyback", stock_code, deal_date)
    filer = article.get("filerName")
    if filer:
        ratio = article.get("ratioChangePct")
        return ("filer", stock_code, deal_date, filer,
                round(ratio, 2) if ratio is not None else None)
    return ("legacy", stock_code, deal_date, (article.get("title") or "").strip())


def find_duplicate_pairs(articles: list) -> list:
    """重複ごとに (残す記事, 消す記事) の組を返す。残す方はリダイレクト先に使う
    （同じ開示を扱っているので、消した記事のURLは残った記事へ引き継げる）。
    キーの一部が欠けている記事（銘柄コードか開示日が空）は突き合わせできないため対象外。"""
    groups = {}
    for a in articles:
        key = duplicate_key(a)
        if not a.get("stockCode") or not str(a.get("dealDate", ""))[:10]:
            continue
        if key[0] == "legacy" and not key[3]:
            continue
        groups.setdefault(key, []).append(a)
    pairs = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda a: a.get("createdAt", ""))
        pairs.extend((members[0], dup) for dup in members[1:])
    return pairs


def find_duplicates(articles: list) -> list:
    """削除対象の記事だけを返す（重複の組は find_duplicate_pairs）。"""
    return [dup for _, dup in find_duplicate_pairs(articles)]


def delete_article(content_id: str) -> bool:
    resp = requests.delete(f"{_microcms_base_url()}/{content_id}", headers=_microcms_headers(), timeout=20)
    return resp.status_code in (200, 202)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3, help="dealDateを遡る日数")
    p.add_argument("--all", action="store_true",
                   help="全期間を見る（--days を無視。旧記事の重複回収に使う）")
    p.add_argument("--code", help="銘柄コードを1つに絞る（例: 9706）")
    p.add_argument("--delete", action="store_true", help="実際に削除する（無指定はdry-run）")
    args = p.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("[cleanup_duplicate_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return

    days = None if args.all else args.days
    scope = "全期間" if days is None else f"直近{days}日"
    if args.code:
        scope += f"・{args.code}のみ"

    articles = fetch_articles(days, args.code)
    pairs = find_duplicate_pairs(articles)
    dups = [dup for _, dup in pairs]
    if not dups:
        print(f"重複なし（{scope}・{len(articles)}記事を確認）")
        return
    redirects = []
    for survivor, a in pairs:
        filer = a.get("filerName") or f"（旧記事・提出者未保存）{a.get('title', '')}"
        label = (f"{a.get('stockCode')} {str(a.get('dealDate', ''))[:10]} {filer} "
                 f"({a.get('ratioChangePct')}pt) → id={a['id']}")
        if not args.delete:
            print(f"  [dry-run] 削除対象: {label}")
        elif delete_article(a["id"]):
            print(f"  🗑 削除: {label}")
            # 消したURLは残した方の記事へ引き継ぐ（404にすると順位ごと捨てることになる）。
            redirects.append({"article_id": a["id"],
                              "target_path": article_redirects.article_target(survivor["id"]),
                              "reason": "duplicate"})
        else:
            print(f"  ⚠ 削除失敗: {label}")
    if redirects:
        article_redirects.record_many(redirects)
        print(f"  ↪ リダイレクトを登録: {len(redirects)}件")
    print(f"重複{len(dups)}件（{scope}・{len(articles)}記事中）"
          f"{'を削除しました' if args.delete else '。--delete で削除実行'}")


if __name__ == "__main__":
    main()
