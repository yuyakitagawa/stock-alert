#!/usr/bin/env python3
"""tools/fix_misreported_blog_articles.py の非課金経路（--fix-body-numbers）のテスト。

本文の再生成は Anthropic API を呼ぶため、保有比率の合算バグ（2026-08-30）の是正では
「本文中の数字だけを置換する」経路を使う。置換の取りこぼしと、置換しても直らない
「規模を語る記述」の検出が主な関心事。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fix_misreported_blog_articles import (  # noqa: E402
    _title_ratio,
    rewrite_body_numbers,
    scale_phrase_conflicts,
)

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got={got!r} want={want!r}")
        print(f"  NG {name}: got={got!r} want={want!r}")
    else:
        print(f"  ok {name}")


def test_replaces_ratio_change_and_amount():
    body = ("<p>保有比率を1.74%まで引き上げたことが分かりました。"
            "前回開示から0.5ポイント増加し、推定取得金額は0.7億円でした。</p>")
    new_body, missed = rewrite_body_numbers(
        body,
        {"ratio": 1.74, "change": 0.5, "amount": 0.7},
        {"ratio": 8.24, "change": 2.1, "amount": 3.4},
    )
    check("置換漏れなし", missed, [])
    check("比率", "8.24%" in new_body and "1.74%" not in new_body, True)
    check("変化幅", "2.1ポイント" in new_body, True)
    check("金額", "3.4億円" in new_body, True)


def test_reports_missing_values():
    """本文が旧値を書いていない（言い換えている）記事は置換できない。
    黙って通すと数字が古いまま残るので、必ず missed に出す。"""
    body = "<p>保有比率を大きく引き上げました。</p>"
    _, missed = rewrite_body_numbers(
        body, {"ratio": 1.74, "change": 0.5, "amount": 0.7},
        {"ratio": 8.24, "change": 2.1, "amount": 3.4},
    )
    check("3項目とも置換できない", sorted(missed), ["amount", "change", "ratio"])


def test_unchanged_values_are_not_reported():
    """値が変わっていない項目は置換対象でもなければ失敗でもない。"""
    body = "<p>保有比率8.24%、推定取得金額は3.4億円です。</p>"
    new_body, missed = rewrite_body_numbers(
        body, {"ratio": 8.24, "change": 2.1, "amount": 3.4},
        {"ratio": 8.24, "change": 2.1, "amount": 3.4},
    )
    check("変化なしは失敗にしない", missed, [])
    check("本文は不変", new_body, body)


def test_sign_is_ignored_for_change():
    """本文の変化幅は符号を持たない（「1.22ポイント低下」）。
    ratioChangePct は売りが負値なので絶対値で突き合わせる。"""
    body = "<p>前回開示から1.22ポイント低下しました。</p>"
    new_body, missed = rewrite_body_numbers(
        body, {"ratio": None, "change": -1.22, "amount": None},
        {"ratio": None, "change": -3.4, "amount": None},
    )
    check("符号を無視して置換", "3.4ポイント低下" in new_body, True)
    check("失敗なし", missed, [])


def test_pt_notation():
    body = "<p>前回から1.22pt低下。</p>"
    new_body, missed = rewrite_body_numbers(
        body, {"change": -1.22}, {"change": -3.4},
    )
    check("pt表記も置換", "3.4pt" in new_body, True)
    check("pt表記で失敗なし", missed, [])


def test_scale_phrase_conflict_detected():
    """1.74%→8.24%のように比率が動くと「約半分を占める」が嘘になる。
    数字を置換しても文章は直らないので、本文の作り直しが要る記事として検出する。"""
    body = "<p>対象企業の株式の約半分を占める筆頭大株主です。</p>"
    check("矛盾を検出", scale_phrase_conflicts(body, 8.24), ["約半分"])
    check("範囲内なら検出しない", scale_phrase_conflicts(body, 49.54), [])


def test_scale_phrase_ignores_tags():
    """タグ属性に数字や語が入っていても本文の記述として数えない。"""
    body = '<p class="3割">保有比率は8.24%です。</p>'
    check("タグは除外", scale_phrase_conflicts(body, 8.24), [])


def test_title_ratio_both_templates():
    check("引き上げ型",
          _title_ratio({"title": "日本製麻（3306）、Ａが保有比率33.99%に引き上げ｜大量保有報告書"}),
          33.99)
    check("新規保有型",
          _title_ratio({"title": "日本製麻（3306）、Ａが1.74%を新規保有｜大量保有報告書"}),
          1.74)
    check("読めないタイトルはNone", _title_ratio({"title": "見出しのない記事"}), None)


for fn in [
    test_replaces_ratio_change_and_amount,
    test_reports_missing_values,
    test_unchanged_values_are_not_reported,
    test_sign_is_ignored_for_change,
    test_pt_notation,
    test_scale_phrase_conflict_detected,
    test_scale_phrase_ignores_tags,
    test_title_ratio_both_templates,
]:
    fn()

if failures:
    print(f"\n❌ {len(failures)} 件失敗")
    sys.exit(1)
print("\n✅ all tests passed")
