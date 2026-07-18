"""
data_sanity.py — Quality Assurance (QA) ロールの中核

役割: リリースのたびに、出力データの「定義上必ず成り立つはず」の不変条件と、
サイト全体（Supabase各テーブル）のデータ欠損・整合性をランタイムで検証する。

きっかけ: ensemble分岐で net=rise（下落確率を引かない）バグが、コードを読んでも
気づけずリリースされた。コードレビューでは防げないので出力データを直接検査する。

2系統のチェック:
  - check_ranking(rows): ランキング単体の不変条件（net=rise-drop 等の行レベル検査）
  - check_site(context): サイト全体の完全性（テーブル横断のカバレッジ・鮮度・欠損）

使い方:
    from lib.data_sanity import run_gate, run_site_gate
    run_gate(ranking_rows, source="rank_stocks")          # リリース前ゲート
    run_site_gate(site_context, source="export_to_web")   # サイト全体チェック

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
NET_TOL          = 0.2     # |net - (rise-drop)| 許容誤差
MIN_ROWS         = 3000    # 想定ユニバース下限
MAX_ROWS         = 4500    # 想定ユニバース上限
DIVERSITY_WARN   = 20      # 上昇確率ユニーク値がこれ未満なら warning（バグ時=18）
DIVERSITY_CRIT   = 8       # これ未満なら critical（ほぼ縮退）
TOP1_SHARE_WARN  = 0.40    # 最頻値の占有率がこれ超なら warning（バグ時=0.41 / 健全=0.26）

# ── ページ・スモーク検査用 ───────────────────────────────────────────────
# 注: Next.jsは not-found / error boundary を全ページのRSCペイロードに埋め込むため
# 「ページが見つかりません」等の文字列は健全ページにも現れる。よって 404/欠落の判定は
# 文字列ではなく HTTPステータス＋期待文言(expect)の有無で行う。
# エラー画面マーカーは error boundary が実際に描画された時のみ現れるので信頼できる。
PAGE_ERROR_MARKERS = ["エラーが発生しました", "データの取得中に問題が発生しました"]
PAGE_MIN_BODY      = 800   # HTML本文がこれ未満なら空ページ疑い


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
            "rise":      _num(r.get("rise_prob", r.get("上昇確率(%)"))),
            "drop":      _num(r.get("drop_prob", r.get("下落確率(%)"))),
            "net":       _num(r.get("net", r.get("ネット(%)"))),
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

    # ── ① net整合: net == rise - drop（このバグの決定的シグネチャ）──────────
    mismatch = 0
    examples = []
    for r in rows:
        if r["rise"] is None or r["drop"] is None or r["net"] is None:
            continue
        expected = round(r["rise"] - r["drop"], 1)
        if abs(r["net"] - expected) > NET_TOL:
            mismatch += 1
            if len(examples) < 3:
                examples.append(
                    f"{r['code']}: net={r['net']} だが rise-drop={expected} "
                    f"(rise={r['rise']}, drop={r['drop']})"
                )
    if mismatch:
        # net==rise になっていないか（ドロップ未減算バグの典型）も診断
        net_eq_rise = sum(
            1 for r in rows
            if r["rise"] is not None and r["net"] is not None and r["drop"]
            and abs(r["net"] - r["rise"]) < 0.01
        )
        hint = "（net==上昇確率＝下落確率の未減算バグの疑い）" if net_eq_rise > n * 0.5 else ""
        v.append(Violation(
            "critical", "net_integrity",
            f"net≠rise-drop が {mismatch}/{n}件{hint}。例: " + " / ".join(examples)
        ))

    # ── ② 確率レンジ: 0<=rise,drop<=100 ─────────────────────────────────
    bad_range = [
        r["code"] for r in rows
        if (r["rise"] is not None and not (0 <= r["rise"] <= 100))
        or (r["drop"] is not None and not (0 <= r["drop"] <= 100))
    ]
    if bad_range:
        v.append(Violation(
            "critical", "prob_range",
            f"確率が0〜100%の範囲外: {len(bad_range)}件 (例 {bad_range[:3]})"
        ))

    # ── ③ 主要列の欠損 ─────────────────────────────────────────────────
    miss_code = sum(1 for r in rows if r["code"] in (None, ""))
    miss_rise = sum(1 for r in rows if r["rise"] is None)
    miss_net  = sum(1 for r in rows if r["net"] is None)
    if miss_code or miss_rise or miss_net:
        v.append(Violation(
            "critical", "missing_fields",
            f"主要列に欠損: code={miss_code}, rise={miss_rise}, net={miss_net}"
        ))

    # ── ④ 予測多様性（縮退検知）──────────────────────────────────────────
    rise_vals = [round(r["rise"], 1) for r in rows if r["rise"] is not None]
    if rise_vals:
        uniq = len(set(rise_vals))
        top1_share = Counter(rise_vals).most_common(1)[0][1] / len(rise_vals)
        if uniq < DIVERSITY_CRIT:
            v.append(Violation(
                "critical", "prediction_collapse",
                f"上昇確率のユニーク値が{uniq}種のみ（モデル出力がほぼ縮退）"
            ))
        elif uniq < DIVERSITY_WARN or top1_share > TOP1_SHARE_WARN:
            v.append(Violation(
                "warning", "low_diversity",
                f"上昇確率の多様性が低い: ユニーク{uniq}種 / 最頻値が{top1_share*100:.0f}%占有"
            ))

    # ── ⑤ 件数 ────────────────────────────────────────────────────────
    if not (MIN_ROWS <= n <= MAX_ROWS):
        v.append(Violation(
            "warning", "row_count",
            f"件数が想定範囲外: {n}件 (想定 {MIN_ROWS}〜{MAX_ROWS})"
        ))

    # ── ⑥ recommend語彙 ───────────────────────────────────────────────
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


def check_site(context: dict) -> list[Violation]:
    """サイト全体のデータ完全性・整合性を検査する。

    context（提供された項目だけ検査。未提供のキーはスキップ）:
      - date:         想定する最新日 "YYYY-MM-DD"
      - rankings:     gen_rankings の行（date/code/rise_prob/drop_prob/net/recommend）
      - stock_meta:   jpx_stock_list の行（code/name/sector）
      - gen_ai_analyses:  gen_ai_analyses の行（code/summary/verdict）
      - earnings:     (廃止済み・互換性のため残存)
      - expected_ai:  AI解析が存在すべき件数（上位N）
      - descriptions: gen_ai_analyses(company-desc-v1) の行（code/summary）＝会社説明
      - desc_targets: 会社説明があるべき銘柄コード（ウォッチリスト＋保有株）
    """
    v: list[Violation] = []
    expected_date = context.get("date")
    rankings = context.get("rankings")

    # ── ランキング本体（行レベル不変条件を再利用）+ 鮮度 ──────────────────
    if "rankings" in context:
        if not rankings:
            v.append(Violation("critical", "rankings_empty",
                               "gen_rankings が空（サイトに本日データが出ない）"))
        else:
            v.extend(check_ranking(rankings))
            if expected_date:
                dates = {r.get("date") for r in rankings if r.get("date")}
                if dates and expected_date not in dates:
                    v.append(Violation("critical", "stale_rankings",
                        f"gen_rankings に本日({expected_date})のデータがない（最新={max(dates)}）"))

    # ── stock_meta カバレッジ（銘柄名・セクターの欠損）───────────────────
    meta = context.get("stock_meta")
    if rankings and meta is not None:
        meta_codes = {str(m.get("code")) for m in meta}
        rank_codes = {str(r.get("code")) for r in rankings}
        missing = rank_codes - meta_codes
        if missing:
            sev = "critical" if len(missing) > len(rank_codes) * 0.2 else "warning"
            v.append(Violation(sev, "meta_coverage",
                f"stock_meta未登録の銘柄 {len(missing)}/{len(rank_codes)}件 (例 {list(missing)[:3]})"))
        # セクター欠損
        no_sector = [str(m.get("code")) for m in meta if not (m.get("sector") or "").strip()]
        if len(no_sector) > len(meta) * 0.3:
            v.append(Violation("warning", "sector_missing",
                f"セクター未設定が{len(no_sector)}/{len(meta)}件（業種別成績が崩れる）"))

    # ── AI解析カバレッジ・空欠損 ─────────────────────────────────────────
    ai = context.get("gen_ai_analyses")
    expected_ai = context.get("expected_ai", 0)
    if ai is not None:
        if expected_ai and len(ai) < expected_ai:
            v.append(Violation("warning", "ai_coverage",
                f"AI解析が{len(ai)}件のみ（想定{expected_ai}件）"))
        empty = [a.get("code") for a in ai if not str(a.get("summary") or "").strip()]
        if empty:
            v.append(Violation("warning", "ai_empty",
                f"AI解析のsummaryが空: {len(empty)}件 (例 {empty[:3]})"))
        if expected_date:
            stale_ai = [a.get("code") for a in ai
                        if a.get("date") and a.get("date") != expected_date]
            if stale_ai and len(stale_ai) == len(ai):
                v.append(Violation("warning", "ai_stale",
                    f"AI解析が古い日付のみ（本日{expected_date}分なし）"))

    # ── 会社説明カバレッジ（詳細ページ「この会社について」）─────────────────
    # descriptions: gen_ai_analyses(model_version=company-desc-v1) の {code, summary}
    # desc_targets: 説明があるべき銘柄コード（ウォッチリスト＋保有株など）
    descriptions = context.get("descriptions")
    desc_targets = context.get("desc_targets")
    if descriptions is not None and desc_targets:
        have = {str(d.get("code")) for d in descriptions
                if str(d.get("summary") or "").strip()}
        targets = [str(c) for c in desc_targets]
        missing = [c for c in targets if c not in have]
        if missing:
            # 対象に説明が1件も無い＝説明パイプライン全体の故障 → critical。
            # 一部欠損はコンテンツ不足 → warning（スプシに追記すれば解消）。
            present = [c for c in targets if c in have]
            sev = "critical" if not present else "warning"
            v.append(Violation(sev, "description_coverage",
                f"会社説明が未登録の銘柄 {len(missing)}/{len(targets)}件"
                f"（詳細ページで『概要情報を取得できませんでした』表示）(例 {missing[:5]})"))

    return v


# 銘柄の価格が全期間で同一値のまま = 凍結（更新パイプラインが古いデータを掴んだまま
# 上書きし続けているバグ）とみなす閾値。
FROZEN_PRICE_RATIO_CRIT = 0.3


def check_price_freshness(history: dict) -> list[Violation]:
    """銘柄ごとの直近close値の時系列が凍結（同一値が続く）していないか検査する（純粋関数）。

    きっかけ: backfill_history.pyが「既存日付はスキップ」するため、価格データ更新前に
    生成された古いgen_rankings行が、価格を修正した後の再実行でも再計算されず
    古い値のまま残り続けるバグがあった（例: 三菱商事8058のcloseが2026-06-02時点の
    値のまま6/19〜7/17まで固まっていた）。単日のスナップショット検査（check_ranking/
    check_site）では検出できないため、複数日にまたがる時系列を直接検査する。

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


def check_pages(results: list[dict]) -> list[Violation]:
    """Webページの巡回結果を検査する（純粋関数：取得は呼び出し側で行う）。

    results: 各ページの取得結果リスト。各要素:
      - route:  ページ識別子（"/", "/rankings", "/stocks/7203" など）
      - status: HTTPステータス（int）
      - body:   レスポンスHTML本文（str）
      - error:  取得時の例外メッセージ（任意。あれば取得失敗扱い）
      - expect: 本文に含まれるべき文言（任意。これが無ければ内容欠落＝404/描画失敗の疑い）
    """
    v: list[Violation] = []
    for r in results:
        route = str(r.get("route") or r.get("url") or "?")
        if r.get("error"):
            v.append(Violation("critical", "page_fetch",
                f"{route}: 取得失敗（{r.get('error')}）"))
            continue
        status = r.get("status")
        body   = str(r.get("body") or "")
        if status != 200:
            v.append(Violation("critical", "page_status",
                f"{route}: HTTP {status}（ページが開けない）"))
            continue
        if any(m in body for m in PAGE_ERROR_MARKERS):
            v.append(Violation("critical", "page_error",
                f"{route}: エラー画面を表示中（描画失敗）"))
            continue
        if len(body) < PAGE_MIN_BODY:
            v.append(Violation("critical", "page_empty",
                f"{route}: 本文が極端に短い（{len(body)}字・空ページ/描画失敗の疑い）"))
            continue
        expect = r.get("expect")
        if expect and expect not in body:
            v.append(Violation("critical", "page_content",
                f"{route}: 期待文言『{expect}』が無い（404/内容欠落の疑い）"))
    return v


def run_pages_gate(results: list[dict], source: str = "", alert: bool = True) -> list[Violation]:
    """全ページ巡回チェック→ログ→（違反あれば）メール通知。alert-only。"""
    violations = check_pages(results)
    summary = format_violations(violations)
    if violations:
        logger.warning("[%s] %s", source or "pages_qa", summary)
        print(summary)
        if alert:
            send_qa_alert(violations, source)
    else:
        logger.info("[%s] %s", source or "pages_qa", summary)
        print(summary)
    return violations


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


def run_site_gate(context: dict, source: str = "", alert: bool = True) -> list[Violation]:
    """サイト全体チェック→ログ→（違反あれば）メール通知。alert-only。"""
    violations = check_site(context)
    summary = format_violations(violations)
    if violations:
        logger.warning("[%s] %s", source or "site_qa", summary)
        print(summary)
        if alert:
            send_qa_alert(violations, source)
    else:
        logger.info("[%s] %s", source or "site_qa", summary)
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
