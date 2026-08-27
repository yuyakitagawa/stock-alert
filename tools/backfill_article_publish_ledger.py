"""開示側の「記事を作ったことがある」台帳（article_published_at）を実績から埋め直す。

なぜ必要か:
  記事の有無をmicroCMSだけで判定していると、**意図的に削除した記事**を「取りこぼし」と
  誤認して作り直してしまう。実績として 2026-08-18に低品質129件、08-25にリライト不能74件と
  重複12件、08-27に誤報12件を削除しており、web/publish_blog_articles.py --backfill は
  そのうち72件を復活させる状態だった。記事を消しても消えない台帳を開示側に持たせて塞ぐ。

台帳の元にする実績は2つ:
  1. microCMSに今ある記事（stockCode + dealDate + filerName）
  2. 削除時のバックアップ logs/deleted_*.json（削除ツールが全フィールドを退避している）
     ※このファイルはリポジトリに入っていない（オーナーのローカルのみ）。CIでは 1 だけになる。

突き合わせのキー:
  - edinet_large_holdings: (issuer_code, disc_date, filer_name) → doc_id。
    同じ提出者が同じ日に複数の報告書を出していれば ratioChangePct と保有比率の変化幅で絞る。
    filerName未保存の旧記事（2026-08-16以前）は提出者で絞れないため、
    (issuer_code, disc_date) がその日その銘柄で1件しか無いときだけ紐づける。
    それでも一意に決まらない開示には印を付けない（1本の記事で複数の開示を「作成済み」に
    すると、まだ記事の無い開示を永久に作れなくなる）。
  - tdnet_buybacks: (code, disclosed_at[:10]) ※自社株買い記事は filerName を持たない

実行:
    python3 tools/backfill_article_publish_ledger.py            # dry-run（件数の表示のみ）
    python3 tools/backfill_article_publish_ledger.py --apply    # 実際に書き込む
    python3 tools/backfill_article_publish_ledger.py --apply --logs-dir /path/to/logs
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import lib.supabase_client as sb  # noqa: E402
from web.publish_blog_articles import fetch_published_index  # noqa: E402

DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
# 台帳を埋める対象の開示日の下限。これより前の開示はbackfillの窓（30日）に二度と入らない。
SINCE_DATE = "2026-01-01"


def article_keys(logs_dir: str) -> list[dict]:
    """記事の実績（今ある記事＋削除済みの記事）を {stockCode, dealDate, filerName, at} で返す。"""
    out: list[dict] = []
    rows = fetch_published_index(SINCE_DATE, fields="stockCode,dealDate,filerName,ratioChangePct,publishedAt,tags")
    if rows is None:
        print("⚠ microCMSの記事一覧を取得できませんでした（削除ログぶんだけで続行）")
        rows = []
    for a in rows:
        out.append({"stockCode": str(a.get("stockCode") or ""),
                    "dealDate": str(a.get("dealDate") or "")[:10],
                    "filerName": a.get("filerName") or "",
                    "ratioChangePct": a.get("ratioChangePct"),
                    "tags": a.get("tags") or "",
                    "at": str(a.get("publishedAt") or "")[:19] or None})
    live = len(out)

    for path in sorted(glob.glob(os.path.join(logs_dir, "deleted_*.json"))):
        try:
            deleted = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)} を読めません: {e}")
            continue
        n = 0
        for a in deleted:
            if not isinstance(a, dict) or not a.get("stockCode"):
                continue
            out.append({"stockCode": str(a["stockCode"]),
                        "dealDate": str(a.get("dealDate") or "")[:10],
                        "filerName": a.get("filerName") or "",
                        "ratioChangePct": a.get("ratioChangePct"),
                        "tags": a.get("tags") or "",
                        "at": str(a.get("publishedAt") or "")[:19] or None})
            n += 1
        print(f"  削除ログ {os.path.basename(path)}: {n}件")
    print(f"記事の実績: 公開中{live}件 + 削除済み{len(out) - live}件 = {len(out)}件")
    return out


def _patch_in_batches(table: str, key_col: str, by_ts: dict, apply: bool) -> int:
    """{タイムスタンプ: [キー...]} をまとめてPATCHする。戻り値は書き込んだ行数。"""
    done = 0
    for ts, keys in sorted(by_ts.items()):
        keys = sorted(set(keys))
        for i in range(0, len(keys), 100):
            chunk = keys[i:i + 100]
            if apply:
                ok = sb.update(table, f"{key_col}=in.({','.join(chunk)})",
                               {"article_published_at": ts})
                if not ok:
                    print(f"  ⚠ {table} の更新に失敗（{len(chunk)}件・{ts}）")
                    continue
            done += len(chunk)
    return done


def backfill_holdings(records: list[dict], apply: bool) -> int:
    """大量保有報告書の開示に article_published_at を立てる。

    同じ銘柄・同じ日・同じ提出者が複数の報告書を出す実例がある（2936 2025-08-13に
    橋本舜が9:50と10:07に別々の変更報告書）ため、提出者まで一致しても doc_id が1つに
    決まらないことがある。そこは記事の ratioChangePct と開示の保有比率変化幅で突き合わせ、
    **それでも決まらない開示には印を付けない**（1本の記事で複数の開示を「作成済み」に
    してしまうと、まだ記事の無い開示を永久に作れなくする＝今回直している不具合そのもの）。"""
    rows = sb.select("edinet_large_holdings",
                     f"disc_date=gte.{SINCE_DATE}&article_published_at=is.null"
                     "&select=doc_id,issuer_code,disc_date,filer_name,holding_ratio,holding_ratio_prior")
    by_triple, by_pair = defaultdict(list), defaultdict(list)
    for r in rows:
        code, d = str(r.get("issuer_code") or ""), str(r.get("disc_date") or "")[:10]
        by_triple[(code, d, r.get("filer_name") or "")].append(r)
        by_pair[(code, d)].append(r)

    def change_of(r: dict) -> "float | None":
        if r.get("holding_ratio") is None or r.get("holding_ratio_prior") is None:
            return None
        return abs(float(r["holding_ratio"]) - float(r["holding_ratio_prior"]))

    by_ts, unmatched, ambiguous = defaultdict(list), 0, 0
    for a in records:
        if "自社株買い" in (a.get("tags") or ""):
            continue                                   # 発行体の自社株買い記事は開示元が別
        hits = by_triple.get((a["stockCode"], a["dealDate"], a["filerName"])) or []
        if not hits and not a["filerName"]:
            # filerName未保存の旧記事。その日その銘柄の開示が1件だけなら一意に決まる
            pair = by_pair.get((a["stockCode"], a["dealDate"]), [])
            hits = pair if len(pair) == 1 else []
        if len(hits) > 1 and a.get("ratioChangePct") is not None:
            target = abs(float(a["ratioChangePct"]))
            narrowed = [r for r in hits if change_of(r) is not None
                        and abs(change_of(r) - target) < 0.01]
            hits = narrowed or hits
        if not hits:
            unmatched += 1
            continue
        if len(hits) > 1:
            ambiguous += 1
            continue
        by_ts[a["at"] or f"{a['dealDate']}T00:00:00Z"].append(hits[0]["doc_id"])
    n = _patch_in_batches("edinet_large_holdings", "doc_id", by_ts, apply)
    print(f"edinet_large_holdings: {n}行に記録{'' if apply else '（dry-run）'}"
          f" / 紐づけられなかった記事 {unmatched}件 / 開示を一意に決められず見送り {ambiguous}件")
    return n


def backfill_buybacks(records: list[dict], apply: bool) -> int:
    """自社株買い決定の開示に article_published_at を立てる。"""
    rows = sb.select("tdnet_buybacks",
                     f"disclosed_at=gte.{SINCE_DATE}&article_published_at=is.null"
                     "&select=code,disclosed_at")
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(str(r["code"]), str(r["disclosed_at"])[:10])].append(r["disclosed_at"][:19])

    by_ts, unmatched = defaultdict(list), 0
    for a in records:
        if "自社株買い" not in (a.get("tags") or ""):
            continue
        hit = by_pair.get((a["stockCode"], a["dealDate"]))
        if not hit:
            unmatched += 1
            continue
        ts = a["at"] or f"{a['dealDate']}T00:00:00Z"
        for disclosed_at in hit:
            by_ts[ts].append(f"{a['stockCode']}|{disclosed_at}")
    # code と disclosed_at の複合キーなので in.() ではまとめられない。1件ずつPATCHする
    done = 0
    for ts, keys in by_ts.items():
        for k in sorted(set(keys)):
            code, disclosed_at = k.split("|", 1)
            if apply:
                if not sb.update("tdnet_buybacks",
                                 f"code=eq.{code}&disclosed_at=eq.{disclosed_at}",
                                 {"article_published_at": ts}):
                    continue
            done += 1
    print(f"tdnet_buybacks: {done}行に記録{'' if apply else '（dry-run）'}"
          f" / 開示に紐づけられなかった記事 {unmatched}件")
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="実際に書き込む（既定はdry-run）")
    p.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR,
                   help="削除時バックアップ deleted_*.json の置き場所")
    args = p.parse_args()

    records = article_keys(args.logs_dir)
    if not records:
        print("記事の実績が1件も取れなかったため中止します")
        return 1
    backfill_holdings(records, args.apply)
    backfill_buybacks(records, args.apply)
    if not args.apply:
        print("\n--apply を付けると書き込みます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
