"""lib/article_redirects.py と tools/backfill_article_redirects.py のユニットテスト
（SupabaseとファイルI/Oはモック）。"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import article_redirects as ar  # noqa: E402
from tools import backfill_article_redirects as backfill  # noqa: E402


def test_stock_target_builds_stock_page_path():
    assert ar.stock_target({"stockCode": "6574"}) == "/stocks/6574"
    assert ar.stock_target({"stock_code": "378A"}) == "/stocks/378A"


def test_stock_target_is_none_without_code():
    """銘柄コードが無い記事は引き継ぎ先が決められないので登録しない（404のままにする）。"""
    assert ar.stock_target({"stockCode": ""}) is None
    assert ar.stock_target({}) is None


def test_article_target_points_to_surviving_article():
    assert ar.article_target("abc123") == "/articles/abc123"


def test_record_many_drops_rows_without_id_or_target():
    captured = {}

    def fake_upsert(table, rows, on_conflict=""):
        captured["table"], captured["rows"], captured["on_conflict"] = table, rows, on_conflict
        return True

    with mock.patch.object(ar.supabase_client, "upsert", side_effect=fake_upsert), \
            mock.patch.object(ar.supabase_client, "update"):
        ok = ar.record_many([
            {"article_id": "a1", "target_path": "/stocks/1332", "reason": "low_value"},
            {"article_id": "", "target_path": "/stocks/9999", "reason": "low_value"},
            {"article_id": "a2", "target_path": None, "reason": "low_value"},
        ])
    assert ok
    assert captured["table"] == ar.TABLE and captured["on_conflict"] == "article_id"
    assert captured["rows"] == [{"article_id": "a1", "target_path": "/stocks/1332", "reason": "low_value"}]


def test_record_many_with_nothing_to_write_does_not_touch_supabase():
    with mock.patch.object(ar.supabase_client, "upsert") as upsert:
        assert ar.record_many([]) is True
    upsert.assert_not_called()


def test_record_flattens_existing_chain_to_the_new_target():
    """A→B のあとにBを消すと A→B→C の2ホップになる。Googleは多段リダイレクトの評価を
    減衰させるので、Bを指していた行をCへ付け替える。"""
    updates = []
    with mock.patch.object(ar.supabase_client, "upsert", return_value=True), \
            mock.patch.object(ar.supabase_client, "update",
                              side_effect=lambda t, q, patch: updates.append((t, q, patch))):
        ar.record("b_id", "/stocks/1332", "low_value")
    assert updates == [(ar.TABLE, "target_path=eq./articles/b_id", {"target_path": "/stocks/1332"})]


def test_lookup_returns_target_path():
    with mock.patch.object(ar.supabase_client, "select_one",
                           return_value={"target_path": "/stocks/1332"}):
        assert ar.lookup("a1") == "/stocks/1332"
    with mock.patch.object(ar.supabase_client, "select_one", return_value=None):
        assert ar.lookup("a1") is None


def _write_log(dir_path: str, name: str, items) -> str:
    path = os.path.join(dir_path, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    return path


def test_collect_reads_deleted_backups_and_skips_broken_ones():
    with tempfile.TemporaryDirectory() as d:
        good = _write_log(d, "deleted_articles_1.json",
                          [{"id": "a1", "stockCode": "1332"}, {"id": "a2", "stockCode": ""}])
        dupe = _write_log(d, "deleted_articles_2.json", [{"id": "a1", "stockCode": "9999"}])
        not_list = _write_log(d, "deleted_articles_3.json", {"id": "a3", "stockCode": "1111"})
        broken = os.path.join(d, "deleted_articles_4.json")
        with open(broken, "w") as f:
            f.write("{ broken json")
        rows = backfill.collect([good, dupe, not_list, broken])
    # a2は銘柄コードが無いので対象外、a1は先に見つかった方（1332）を採る、
    # リストでないJSONと壊れたJSONは黙って飛ばす。
    assert rows == [{"article_id": "a1", "target_path": "/stocks/1332",
                     "reason": "backfill:deleted_articles_1.json"}]


def test_backfill_dry_run_does_not_write():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d, "deleted_articles_1.json", [{"id": "a1", "stockCode": "1332"}])
        with mock.patch.object(backfill, "LOG_GLOB", os.path.join(d, "deleted_*.json")), \
                mock.patch.object(backfill.article_redirects, "record_many") as record, \
                mock.patch.object(sys, "argv", ["backfill_article_redirects.py"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = backfill.main()
    assert rc == 0 and "1件のリダイレクトを検出" in buf.getvalue()
    record.assert_not_called()


def test_backfill_write_registers_rows():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d, "deleted_articles_1.json", [{"id": "a1", "stockCode": "1332"}])
        with mock.patch.object(backfill, "LOG_GLOB", os.path.join(d, "deleted_*.json")), \
                mock.patch.object(backfill.article_redirects, "record_many",
                                  return_value=True) as record, \
                mock.patch.object(sys, "argv", ["backfill_article_redirects.py", "--write"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = backfill.main()
    assert rc == 0 and "登録完了: 1件" in buf.getvalue()
    assert record.call_args[0][0][0]["article_id"] == "a1"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'FAILED' if fails else 'PASSED'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)
