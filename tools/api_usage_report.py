"""
tools/api_usage_report.py

`api_usage` テーブルから Anthropic API の利用実績を集計して表示する（手動実行専用）。

なぜ必要か:
  2026-08-23の月次上限到達では「どの用途がいくら使ったか」を示す記録が無く、
  バックフィルのログを後からgrepして犯人を推定するしかなかった。lib/api_usage.py が
  呼び出しごとに残す実績を、ここで用途別・日別に読む。

注意:
  コストは公開単価からの**推定値**であり、Anthropicの請求額そのものではない。
  記録開始（2026-08-29）より前の消費は入っていない。

実行:
  python3 tools/api_usage_report.py              # 直近30日
  python3 tools/api_usage_report.py --days 7
  python3 tools/api_usage_report.py --by job     # 日別+ジョブ別
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402

DEFAULT_DAYS = 30


def fetch_rows(days: int) -> list[dict]:
    since = (datetime.now(timezone.utc).date() - timedelta(days=days - 1)).isoformat()
    return sb.select("api_usage", f"usage_date=gte.{since}&order=usage_date.asc")


def _sum(rows: list[dict], key_of) -> dict:
    """key_of(row) 単位で合計する。"""
    out: dict = defaultdict(lambda: dict(calls=0, input_tokens=0, output_tokens=0,
                                         cache_write_tokens=0, cache_read_tokens=0,
                                         web_search_requests=0, cost_usd=0.0))
    for r in rows:
        agg = out[key_of(r)]
        for k in agg:
            agg[k] += float(r.get(k) or 0) if k == "cost_usd" else int(r.get(k) or 0)
    return dict(out)


def _print_table(title: str, agg: dict, sort_by_cost: bool = False) -> None:
    if not agg:
        return
    keys = sorted(agg, key=lambda k: -agg[k]["cost_usd"]) if sort_by_cost else sorted(agg)
    # 見出しは半角に揃える。全角は表示幅2なのに1文字と数えられ、桁がずれる。
    print(f"\n■ {title}")
    print(f"{'':<26}{'calls':>7}{'input':>12}{'output':>10}{'srch':>6}{'est.$':>9}")
    print("-" * 70)
    for k in keys:
        a = agg[k]
        print(f"{str(k):<26}{a['calls']:>7,}{a['input_tokens']:>12,}"
              f"{a['output_tokens']:>10,}{a['web_search_requests']:>6,}"
              f"{a['cost_usd']:>9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="対象日数（UTC基準）")
    ap.add_argument("--by", choices=["task", "job", "model", "all"], default="task",
                    help="日別に加えて出す内訳（既定: task）")
    args = ap.parse_args()

    if not sb.is_configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です。")
        return 1

    rows = fetch_rows(args.days)
    if not rows:
        print(f"直近{args.days}日の記録がありません（記録開始は2026-08-29）。")
        return 0

    total = _sum(rows, lambda r: "合計")["合計"]
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    month_rows = [r for r in rows if str(r["usage_date"]).startswith(month)]
    month_cost = sum(float(r.get("cost_usd") or 0) for r in month_rows)

    print(f"=== Anthropic API 利用実績（直近{args.days}日 / UTC）===")
    print(f"呼び出し {total['calls']:,}回 / 入力 {total['input_tokens']:,}tk / "
          f"出力 {total['output_tokens']:,}tk / Web検索 {total['web_search_requests']:,}回")
    print(f"推定コスト 合計 ${total['cost_usd']:.2f}（うち{month}月分 ${month_cost:.2f}）")
    if total["cache_read_tokens"]:
        print(f"キャッシュ 書込 {total['cache_write_tokens']:,}tk / "
              f"読出 {total['cache_read_tokens']:,}tk")

    _print_table("日別", _sum(rows, lambda r: r["usage_date"]))
    wanted = ["task", "job", "model"] if args.by == "all" else [args.by]
    labels = {"task": "タスク", "job": "ジョブ", "model": "モデル"}
    for col in wanted:
        _print_table(f"{labels[col]}別", _sum(rows, lambda r, c=col: r[c]),
                     sort_by_cost=True)

    print("\n※ コストは公開単価からの推定値。Anthropicの請求額とは一致しない。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
