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
    en_needs_rewrite,
    rebuild_en_title,
    rewrite_body_numbers,
    rewrite_en_body_numbers,
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



# ---- 英語版（titleEn / bodyEn）----
# 2026-09-10: 是正が title/body だけに入り、英語版324本中82本が和文と矛盾していた。

def _sheet(ratio, direction="buy", doc="変更報告書", prior=10.0, change=1.0):
    return {"stock_code": "9201", "holding_ratio": ratio, "direction": direction,
            "doc_type_label": doc, "prior_ratio": prior, "ratio_change_pct": change,
            "is_correction": False}


def test_en_title_flips_direction_and_ratio():
    old = "Nomura Securities Cuts Stake in Japan Airlines (9201) to 0% | Large Shareholding Report"
    got = rebuild_en_title(old, _sheet(8.14, "buy", prior=7.0, change=1.14))
    check("方向と比率を組み直す", got,
          "Nomura Securities Raises Stake in Japan Airlines (9201) to 8.14% | Large Shareholding Report")


def test_en_title_new_holding():
    old = "Arne Deussen Takes 3.42% Stake in Kobayashi Paper (3944) | Large Shareholding Report"
    got = rebuild_en_title(old, _sheet(5.12, "buy", doc="大量保有報告書", prior=0, change=5.12) | {"stock_code": "3944"})
    check("新規保有の英題", got, "Arne Deussen Takes 5.12% Stake in Kobayashi Paper (3944) | Large Shareholding Report")


def test_en_title_keeps_parentheses_in_company_name():
    old = "Evo Fund Cuts Stake in Tokyo Kiraboshi (Holdings) (7173) to 4.2% | Large Shareholding Report"
    got = rebuild_en_title(old, _sheet(3.9, "sell", change=-0.3) | {"stock_code": "7173"})
    check("社名の括弧を残す", got,
          "Evo Fund Cuts Stake in Tokyo Kiraboshi (Holdings) (7173) to 3.9% | Large Shareholding Report")


def test_en_title_unknown_template_is_none():
    check("テンプレート外はNone", rebuild_en_title("Something else entirely", _sheet(5.0)), None)


def test_en_body_numbers():
    body = ("<p>raised its stake to 11.64%, a 1.16 percentage point increase, "
            "with an estimated ¥2.33 billion purchase.</p>")
    new_body, missed = rewrite_en_body_numbers(
        body, {"ratio": 11.64, "change": 1.16, "amount": 23.3}, {"ratio": 12.03, "change": 0.38, "amount": 7.7})
    check("英語: 置換漏れなし", missed, [])
    check("英語: 比率", "12.03%" in new_body and "11.64%" not in new_body, True)
    check("英語: 変化幅", "0.38 percentage point" in new_body, True)
    check("英語: 金額（10億円未満はmillion）", "¥770 million" in new_body, True)


def test_en_body_reports_missing():
    _, missed = rewrite_en_body_numbers("<p>no numbers here</p>", {"ratio": 5.0, "change": 1.0, "amount": 3.0},
                                        {"ratio": 6.0, "change": 2.0, "amount": 4.0})
    check("英語: 旧値が無い項目を報告", missed, ["ratio", "change", "amount"])


def test_en_needs_rewrite_on_direction_flip():
    check("方向が反転したら書き直し", en_needs_rewrite({"ratioChangePct": -0.13}, {"signed_change": 1.14}, []), True)
    check("同方向・置換済みなら不要", en_needs_rewrite({"ratioChangePct": 1.16}, {"signed_change": 0.38}, []), False)
    check("旧値が残るなら書き直し", en_needs_rewrite({"ratioChangePct": 1.16}, {"signed_change": 0.38}, ["amount"]), True)

for fn in [
    test_replaces_ratio_change_and_amount,
    test_reports_missing_values,
    test_unchanged_values_are_not_reported,
    test_sign_is_ignored_for_change,
    test_pt_notation,
    test_scale_phrase_conflict_detected,
    test_scale_phrase_ignores_tags,
    test_title_ratio_both_templates,
    test_en_title_flips_direction_and_ratio,
    test_en_title_new_holding,
    test_en_title_keeps_parentheses_in_company_name,
    test_en_title_unknown_template_is_none,
    test_en_body_numbers,
    test_en_body_reports_missing,
    test_en_needs_rewrite_on_direction_flip,
]:
    fn()

if failures:
    print(f"\n❌ {len(failures)} 件失敗")
    sys.exit(1)
print("\n✅ all tests passed")
