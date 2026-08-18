"""
lib/attention_score.py
クジラ注目度スコア: EDINET大量保有報告の開示内容から0〜100点の「期待リターンスコア」を
算出する。「保有比率が高い」「急増した」「アクティビスト」は見た目のインパクトはあるが、
実際の63営業日後リターンとは弱い負の相関〜無相関だった（買い開示4,232件を
開示日エントリー・63営業日保有で検証、2026-08-15）。そのため直感ではなく実績データで
較正したスコアカード方式を採用する。

算出方法:
  1. 保有比率・保有比率の変化幅(pt)・推定取引金額(億円)を、それぞれ5分位ビンの実績平均
     リターンに変換する（HOLDING_RATIO_*, RATIO_CHANGE_*, DEAL_AMOUNT_* の各定数）。
  2. 投資家分類(13分類)は縮小推定(shrinkage, k=20)した実績平均リターンに変換する
     （CATEGORY_SHRUNK_RETURN、サンプルの少ない分類が極端な値にならないよう全体平均に
     引き寄せる）。
  3. 上記4値をRidge回帰の重みで線形結合し「期待リターン」を予測する。
  4. 期待リターンを、学習時点の予測値分布のパーセンタイルへ変換して0〜100点にする
     （p50=50点が典型的な開示、p90=90点は上位10%）。

「過去の取得回数」は検証したが有意な関係が見られなかったため採用していない
（CLAUDE.md 7章「不採用機能はコードからも消す」の方針に従いスコアに含めない）。

再学習: サンプル・分布が古くなったら同じ手法（edinet_large_holdingsから買い開示を抽出し
開示日エントリー・63営業日保有のリターンを本ファイルのビンで集計）で定数を再計算し、
下記の定数を差し替える。なお当時のイベント抽出は holding_ratio_prior がNULLの開示を
「新規取得」として買い扱いしており、売り開示が一部混入している（2026-08-18に判明）。
再学習時は prior がNULLの開示を除外すること。
"""
import math

# ── 1) 分位ビン: 実績平均リターン(小数, 例 0.07=7%) ─────────────────────────
# edges は [下限, ..., 上限] で長さ = len(means) + 1、両端は ±inf。

HOLDING_RATIO_EDGES = [-math.inf, 1.99, 5.06, 7.0, 11.69, math.inf]
HOLDING_RATIO_MEANS = [0.10033, 0.07519, 0.04050, 0.02234, -0.01680]

RATIO_CHANGE_EDGES = [-math.inf, 0.28, 1.02, 1.52, 5.078, math.inf]
RATIO_CHANGE_MEANS = [0.07205, 0.05915, 0.04777, 0.05298, -0.01019]

DEAL_AMOUNT_EDGES = [-math.inf, 1.0153, 3.1975, 8.667, 31.5228, math.inf]
DEAL_AMOUNT_MEANS = [0.01380, 0.03380, 0.03757, 0.05970, 0.07722]

# ── 2) 投資家分類: 縮小推定後の実績平均リターン ──────────────────────────
CATEGORY_SHRUNK_RETURN = {
    "プライムブローカー": 0.13408,
    "国内アセットマネジメント": 0.08534,
    "日系証券銀行": 0.07944,
    "外資系伝統運用会社": 0.06796,
    "公益/一般財団法人": 0.06491,
    "創業家の資産管理会社": 0.04791,
    "アクティビスト": 0.03222,
    "PE・メザニンファンド": 0.02255,
    "独立系ブティックAM": 0.01393,
    "VC": 0.01237,
    "事業会社": 0.00886,
    "その他": 0.00091,
    "個人": -0.00815,
}
OVERALL_MEAN_RETURN = 0.04438  # カテゴリ不明時のフォールバック

# ── 3) Ridge回帰の重み ───────────────────────────────────────────────────
RIDGE_COEF = {
    "holding_ratio": 0.49987,
    "ratio_change": 0.76521,
    "deal_amount": 0.98234,
    "category": 0.46913,
}
RIDGE_INTERCEPT = -0.07628

# ── 4) 期待リターン → 0-100点のパーセンタイル変換テーブル ───────────────
# (percentile, expected_return) を学習時点の予測値分布から算出。区間は線形補間。
PERCENTILE_TABLE = [
    (0, -0.06317), (5, -0.03445), (10, -0.02089), (20, -0.00243),
    (30, 0.00953), (40, 0.02663), (50, 0.04299), (60, 0.06125),
    (70, 0.07983), (80, 0.09696), (90, 0.11496), (95, 0.12331),
    (99, 0.13502), (100, 0.15789),
]

# 取得回数(prior_count)はXGBoost特徴量重要度・単純相関いずれでも実質無関係
# (OOF R^2がprior_count追加前後でほぼ同一)だったためスコアに含めない。


def _bin_score(value: float, edges: list, means: list) -> float:
    for i, mean in enumerate(means):
        if value <= edges[i + 1]:
            return mean
    return means[-1]


def _percentile_score(expected_return: float) -> int:
    xs = [p[1] for p in PERCENTILE_TABLE]
    ys = [p[0] for p in PERCENTILE_TABLE]
    if expected_return <= xs[0]:
        return 0
    if expected_return >= xs[-1]:
        return 100
    for i in range(len(xs) - 1):
        if xs[i] <= expected_return <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            frac = (expected_return - xs[i]) / span if span else 0.0
            return round(ys[i] + frac * (ys[i + 1] - ys[i]))
    return 100


def _stars(score: int) -> int:
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 55:
        return 3
    if score >= 35:
        return 2
    return 1


def _build_reasons(holding_ratio, ratio_change_pct, deal_amount_oku, category,
                    hr_score, rc_score, da_score, cat_score) -> list:
    candidates = []
    if deal_amount_oku >= 31.52:
        candidates.append((da_score, f"推定取引規模{deal_amount_oku:.1f}億円の大口取引"))
    elif deal_amount_oku >= 8.67:
        candidates.append((da_score, f"推定取引規模{deal_amount_oku:.1f}億円"))
    if holding_ratio <= 1.99:
        candidates.append((hr_score, f"保有比率{holding_ratio:.2f}%の低水準からの新規参入"))
    elif holding_ratio <= 5.06:
        candidates.append((hr_score, f"保有比率{holding_ratio:.2f}%とまだ小さい段階での取得"))
    if abs(ratio_change_pct) <= 1.02:
        candidates.append((rc_score, "急激な買い増しではなく緩やかな積み増し"))
    if category in ("プライムブローカー", "国内アセットマネジメント", "日系証券銀行", "外資系伝統運用会社"):
        candidates.append((cat_score, f"{category}による取得"))
    candidates.sort(key=lambda x: -x[0])
    return [text for _, text in candidates[:3]]


def compute_attention_score(holding_ratio: float, ratio_change_pct: float,
                             deal_amount_oku: float, category: str) -> dict:
    """EDINET大量保有報告1件のクジラ注目度を算出する。
    holding_ratio: 今回の保有比率(%)
    ratio_change_pct: 前回開示からの保有比率変化(pt, プラスが買い増し)
    deal_amount_oku: 推定取引金額(億円)
    category: 投資家分類(13分類のいずれか。未知の値はその他扱いのフォールバック)
    """
    hr_score = _bin_score(holding_ratio, HOLDING_RATIO_EDGES, HOLDING_RATIO_MEANS)
    rc_score = _bin_score(ratio_change_pct, RATIO_CHANGE_EDGES, RATIO_CHANGE_MEANS)
    da_score = _bin_score(deal_amount_oku, DEAL_AMOUNT_EDGES, DEAL_AMOUNT_MEANS)
    cat_score = CATEGORY_SHRUNK_RETURN.get(category, OVERALL_MEAN_RETURN)

    expected_return = (
        RIDGE_COEF["holding_ratio"] * hr_score
        + RIDGE_COEF["ratio_change"] * rc_score
        + RIDGE_COEF["deal_amount"] * da_score
        + RIDGE_COEF["category"] * cat_score
        + RIDGE_INTERCEPT
    )
    score = _percentile_score(expected_return)
    reasons = _build_reasons(
        holding_ratio, ratio_change_pct, deal_amount_oku, category,
        hr_score, rc_score, da_score, cat_score,
    )
    return {
        "score": score,
        "stars": _stars(score),
        "expected_return_pct": round(expected_return * 100, 2),
        "reasons": reasons,
    }
