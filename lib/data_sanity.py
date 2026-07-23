"""
data_sanity.py — Quality Assurance (QA) ロールの中核

役割: リリースのたびに、出力データの「定義上必ず成り立つはず」の不変条件をランタイムで検証する。

check_ranking(rows): ランキング単体の不変条件（下落確率レンジ・多様性・欠損等の行レベル検査）。
下落モデルのみに一本化済み（上昇モデル・netスコアは廃止）のため、下落確率を中心に検査する。

使い方:
    from lib.data_sanity import run_gate
    run_gate(ranking_rows, source="rank_stocks")          # リリース前ゲート

各チェックは純粋関数。重大度 critical / warning を付与して Violation のリストを返す。
alert-only: critical でも例外を投げず、呼び出し側の処理は継続させる（通知のみ）。
"""
from __future__ import annotations
import os
import smtplib
import logging
from collections import Counter
from dataclasses import dataclass
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)
NOTIFY_TO = os.getenv("ALERT_TO", "dosankoure@gmail.com")

# 既知の推奨ラベル（絵文字付き・無し両対応）
KNOWN_RECOMMEND = {
    "💎 買い", "—",
    "🥇 S買い", "🥈 A買い", "⏳ 方向感なし",
    "S買い", "A買い", "方向感なし",
    "買い継続", "買い増し", "高値警戒",
}

# しきい値
MIN_ROWS         = 3000    # 想定ユニバース下限
MAX_ROWS         = 4500    # 想定ユニバース上限
DIVERSITY_WARN   = 20      # 下落確率ユニーク値がこれ未満なら warning（バグ時=18）
DIVERSITY_CRIT   = 8       # これ未満なら critical（ほぼ縮退）
TOP1_SHARE_WARN  = 0.40    # 最頻値の占有率がこれ超なら warning（バグ時=0.41 / 健全=0.26）


@dataclass
class Violation:
    severity: str   # "critical" | "warning"
    check:    str   # チェック名
    detail:   str   # 人間可読の説明


def _to_rows(data) -> list[dict]:
    """DataFrame でも list[dict] でも受け取れるよう正規化。列名は英/日どちらも許容。"""
    if hasattr(data, "to_dict"):  # pandas DataFrame
        recs = data.to_dict("records")
    else:
        recs = list(data)

    out = []
    for r in recs:
        out.append({
            "code":      r.get("code", r.get("銘柄コード")),
            "drop":      _num(r.get("drop_prob", r.get("下落確率(%)"))),
            "recommend": r.get("recommend", r.get("推奨")),
        })
    return out


def _num(v):
    """数値化。"-"・None・空文字・NaN は None に。"""
    if v is None or v == "-" or v == "":
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN 除外
    except (ValueError, TypeError):
        return None


def check_ranking(data) -> list[Violation]:
    """ランキング結果の不変条件を検査して Violation のリストを返す。"""
    rows = _to_rows(data)
    v: list[Violation] = []

    if not rows:
        return [Violation("critical", "empty", "ランキング結果が0件")]

    n = len(rows)

    # ── ① 確率レンジ: 0<=drop<=100 ─────────────────────────────────────
    bad_range = [
        r["code"] for r in rows
        if r["drop"] is not None and not (0 <= r["drop"] <= 100)
    ]
    if bad_range:
        v.append(Violation(
            "critical", "prob_range",
            f"確率が0〜100%の範囲外: {len(bad_range)}件 (例 {bad_range[:3]})"
        ))

    # ── ② 主要列の欠損 ─────────────────────────────────────────────────
    miss_code = sum(1 for r in rows if r["code"] in (None, ""))
    miss_drop = sum(1 for r in rows if r["drop"] is None)
    if miss_code or miss_drop:
        v.append(Violation(
            "critical", "missing_fields",
            f"主要列に欠損: code={miss_code}, drop={miss_drop}"
        ))

    # ── ③ 予測多様性（縮退検知）──────────────────────────────────────────
    drop_vals = [round(r["drop"], 1) for r in rows if r["drop"] is not None]
    if drop_vals:
        uniq = len(set(drop_vals))
        top1_share = Counter(drop_vals).most_common(1)[0][1] / len(drop_vals)
        if uniq < DIVERSITY_CRIT:
            v.append(Violation(
                "critical", "prediction_collapse",
                f"下落確率のユニーク値が{uniq}種のみ（モデル出力がほぼ縮退）"
            ))
        elif uniq < DIVERSITY_WARN or top1_share > TOP1_SHARE_WARN:
            v.append(Violation(
                "warning", "low_diversity",
                f"下落確率の多様性が低い: ユニーク{uniq}種 / 最頻値が{top1_share*100:.0f}%占有"
            ))

    # ── ④ 件数 ────────────────────────────────────────────────────────
    if not (MIN_ROWS <= n <= MAX_ROWS):
        v.append(Violation(
            "warning", "row_count",
            f"件数が想定範囲外: {n}件 (想定 {MIN_ROWS}〜{MAX_ROWS})"
        ))

    # ── ⑤ recommend語彙 ───────────────────────────────────────────────
    unknown = {r["recommend"] for r in rows
               if r["recommend"] and r["recommend"] not in KNOWN_RECOMMEND}
    if unknown:
        v.append(Violation(
            "warning", "recommend_vocab",
            f"未知の推奨ラベル: {sorted(unknown)[:5]}"
        ))

    return v


def has_critical(violations: list[Violation]) -> bool:
    return any(x.severity == "critical" for x in violations)


def format_violations(violations: list[Violation]) -> str:
    """違反リストを人間可読サマリに整形。"""
    if not violations:
        return "✅ データ整合性チェック: 違反なし"
    lines = [f"🚨 データ整合性チェック: {len(violations)}件の違反"]
    for x in violations:
        mark = "🔴 critical" if x.severity == "critical" else "🟡 warning"
        lines.append(f"  [{mark}] {x.check}: {x.detail}")
    return "\n".join(lines)


def send_qa_alert(violations: list[Violation], source: str = "") -> bool:
    """違反をメール通知する（alert-only: 処理は止めない）。送信成否を返す。"""
    # ユーザー依頼でデータ整合性アラートメール停止(2026-06-20)。復活はこのreturnを削除。
    return False
    if not violations:
        return False
    gmail_addr = os.getenv("GMAIL_ADDRESS")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_addr or not gmail_pass:
        logger.warning("[data_sanity] GMAIL未設定のためQAアラート送信スキップ")
        return False

    n_crit = sum(1 for x in violations if x.severity == "critical")
    icon   = "🔴" if n_crit else "🟡"
    subject = f"{icon} データ整合性アラート — {source or 'ランキング'} ({n_crit}件critical)"
    body = (
        f"出力データの不変条件チェックで違反を検知しました。\n"
        f"発生源: {source}\n\n"
        f"{format_violations(violations)}\n\n"
        "━━━━━━━━━━━━━━━━\n"
        "※ alert-only 設定のため、データ更新は継続しています。\n"
        "  critical の場合は web/メールに誤データが出ている可能性があるため確認してください。\n"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = gmail_addr
    msg["To"]      = NOTIFY_TO
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_addr, gmail_pass)
            server.sendmail(gmail_addr, NOTIFY_TO, msg.as_string())
        logger.info("📧 QAアラート送信: %s", subject)
        return True
    except Exception as e:
        logger.error("[data_sanity] QAアラート送信失敗: %s", e)
        return False


# 銘柄の価格が全期間で同一値のまま = 凍結（更新パイプラインが古いデータを掴んだまま
# 上書きし続けているバグ）とみなす閾値。
FROZEN_PRICE_RATIO_CRIT = 0.3


def check_price_freshness(history: dict) -> list[Violation]:
    """銘柄ごとの直近close値の時系列が凍結（同一値が続く）していないか検査する（純粋関数）。

    きっかけ: backfill_history.pyが「既存日付はスキップ」するため、価格データ更新前に
    生成された古いgen_rankings行が、価格を修正した後の再実行でも再計算されず
    古い値のまま残り続けるバグがあった（例: 三菱商事8058のcloseが2026-06-02時点の
    値のまま6/19〜7/17まで固まっていた）。単日のスナップショット検査（check_ranking）
    では検出できないため、複数日にまたがる時系列を直接検査する。

    Args:
        history: {code: [close1, close2, ...]}。各リストは時系列順、2件以上を想定。
                 1件以下の銘柄は判定対象外（比較不能なため）。
    """
    v: list[Violation] = []
    if not history:
        return v
    checkable = {code: closes for code, closes in history.items() if len(closes) >= 2}
    if not checkable:
        return v
    frozen = [code for code, closes in checkable.items() if len(set(closes)) == 1]
    if frozen:
        ratio = len(frozen) / len(checkable)
        sev = "critical" if ratio > FROZEN_PRICE_RATIO_CRIT else "warning"
        v.append(Violation(sev, "frozen_price",
            f"価格が全期間で同一値のまま凍結: {len(frozen)}/{len(checkable)}件 "
            f"(例 {frozen[:5]})"))
    return v


def run_price_freshness_gate(history: dict, source: str = "", alert: bool = True) -> list[Violation]:
    """価格凍結チェック→ログ→（違反あれば）メール通知。alert-only。"""
    violations = check_price_freshness(history)
    summary = format_violations(violations)
    if violations:
        logger.warning("[%s] %s", source or "price_freshness", summary)
        print(summary)
        if alert:
            send_qa_alert(violations, source)
    else:
        logger.info("[%s] %s", source or "price_freshness", summary)
        print(summary)
    return violations


def run_gate(data, source: str = "", alert: bool = True) -> list[Violation]:
    """検査→ログ出力→（違反あれば）メール通知 をまとめて行うパイプライン用ゲート。
       alert-only: critical でも例外を投げず、呼び出し側の処理は継続させる。"""
    violations = check_ranking(data)
    summary = format_violations(violations)
    if violations:
        logger.warning("[%s] %s", source or "data_sanity", summary)
        print(summary)
        if alert:
            send_qa_alert(violations, source)
    else:
        logger.info("[%s] %s", source or "data_sanity", summary)
        print(summary)
    return violations
