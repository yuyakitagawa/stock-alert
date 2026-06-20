"""
risk_regime.py — 相場リスク管制官（Market Risk Officer）の中核

マクロ指標（日経20日・VIX・ドル円・S&P500）から、その日の「リスクオン/オフ」を
多面的に判定する。単一指標（日経20日だけ）より頑健で、買いを出してよい地合いかを
具体的なアクションに翻訳する。

出力（assess の戻り値）:
  {
    "regime":  "risk_on" | "caution" | "risk_off",
    "score":   リスク点（0=安全 … 大きいほど危険）,
    "action":  "通常" | "買い縮小" | "買い見送り",
    "label":   "🟢 リスクオン" 等の表示用ラベル,
    "reasons": ["日経20日 -6.2%（弱気水準）", ...],
    "suppress_buy": bool,   # True なら S買いを抑制すべき地合い
  }
"""
from __future__ import annotations

# しきい値（config の BEAR_MARKET_THRESHOLD=-5 と整合）
NK20_CAUTION = -5.0    # 日経20日がこれ以下で警戒
NK20_RISKOFF = -8.0    # これ以下でリスクオフ
VIX_CAUTION  = 22.0    # VIXがこれ超で警戒
VIX_RISKOFF  = 30.0    # これ超でリスクオフ（恐怖相場）
JPY5_SPIKE   = -3.0    # ドル円5日がこれ以下=円急騰（輸出株にリスクオフ）
US20_CAUTION = -5.0    # S&P500 20日がこれ以下で警戒


def assess(nk20=None, vix=None, jpy5=None, us20=None, us5=None) -> dict:
    """マクロ指標から当日のリスク地合いを判定する。"""
    score = 0
    reasons: list[str] = []

    # 日経20日トレンド（最重要）
    if nk20 is not None:
        if nk20 <= NK20_RISKOFF:
            score += 2
            reasons.append(f"日経20日 {nk20:+.1f}%（明確な下落トレンド）")
        elif nk20 <= NK20_CAUTION:
            score += 1
            reasons.append(f"日経20日 {nk20:+.1f}%（弱含み）")

    # VIX（恐怖指数）
    if vix is not None:
        if vix >= VIX_RISKOFF:
            score += 2
            reasons.append(f"VIX {vix:.0f}（恐怖相場）")
        elif vix >= VIX_CAUTION:
            score += 1
            reasons.append(f"VIX {vix:.0f}（不安定）")

    # ドル円急変（円急騰はリスクオフの典型）
    if jpy5 is not None and jpy5 <= JPY5_SPIKE:
        score += 1
        reasons.append(f"ドル円5日 {jpy5:+.1f}%（円急騰=リスクオフ）")

    # 米国市場の地合い
    if us20 is not None and us20 <= US20_CAUTION:
        score += 1
        reasons.append(f"S&P500 20日 {us20:+.1f}%（米国も軟調）")

    # 総合判定
    if score >= 4:
        regime, action, label, suppress = "risk_off", "買い見送り", "🔴 リスクオフ", True
    elif score >= 2:
        regime, action, label, suppress = "caution", "買い縮小", "🟡 警戒", False
    else:
        regime, action, label, suppress = "risk_on", "通常", "🟢 リスクオン", False

    if not reasons:
        reasons.append("マクロに大きな警戒材料なし")

    return {
        "regime":       regime,
        "score":        score,
        "action":       action,
        "label":        label,
        "reasons":      reasons,
        "suppress_buy": suppress,
    }


def summary_line(verdict: dict) -> str:
    """1行サマリー（ログ・メール見出し用）。"""
    return f"{verdict['label']}（リスク点{verdict['score']}）→ {verdict['action']}: " + " / ".join(verdict["reasons"])
